"""#7 集成层：把前 6 项优化接进 BasicsTransformerLM，且**不改** cs336-basics/model.py。

三件事:
1. flash_sdpa: 用 Triton Flash attention (#3/#4/#5) 替换模型里的 naive scaled_dot_product_attention。
   通过 monkey-patch cs336_basics.model.scaled_dot_product_attention 实现——模型 attention 层
   内部调用的就是这个模块级函数, patch 掉即可全局生效, 无需改模型代码。
2. forward_hidden: 跑 embedding → layers → ln_final, 返回**隐藏态**(不过 lm_head, 不 materialize
   [N, vocab] 全量 logits)。外部再用 #1 的 fused_linear_cross_entropy(h, lm_head.weight, targets)。
3. apply_activation_checkpointing (#6, 从 activation_ckpt 复用)。

FSDP(#7 切分) / FusedAdamW(#2) 直接在 leaderboard.py 里组装, 这里只提供模型侧的 hook。
"""

from __future__ import annotations

import contextlib

import torch

import cs336_basics.model as basics_model
from cs336_systems.opt_kernel import FlashAttnFuncTriton


def flash_sdpa(Q, K, V, mask=None):
    """替换 model.scaled_dot_product_attention 的 Flash 版本。

    模型传进来的 Q/K/V 是 4D [batch, heads, seq, d_head], 且传了一个 causal bool mask。
    我们的 Triton kernel 要求 3D [bs, seq, d] 且 causal 内建(不吃 mask), 所以:
      - 把 [B, H, S, d] 折成 [B*H, S, d];
      - is_causal=True 交给 kernel(它自己按下三角算, 比传 mask 更省);
      - 结果折回 [B, H, S, d]。

    注意: 模型总是以 causal mask 调这个函数(CausalMultiHeadSelfAttention 里构造的下三角),
    所以这里恒定 is_causal=True; 传入的 mask 忽略。
    """
    lead = Q.shape[:-2]          # (batch, heads) 等前导维
    seq, d = Q.shape[-2], Q.shape[-1]

    # 折成 3D [prod(lead), seq, d]，kernel 要求内存连续
    q3 = Q.reshape(-1, seq, d).contiguous()
    k3 = K.reshape(-1, K.shape[-2], d).contiguous()
    v3 = V.reshape(-1, V.shape[-2], V.shape[-1]).contiguous()

    o3 = FlashAttnFuncTriton.apply(q3, k3, v3, True)

    return o3.reshape(*lead, seq, o3.shape[-1])


@contextlib.contextmanager
def use_flash_attention():
    """上下文内把模型的 scaled_dot_product_attention 换成 Triton Flash, 退出还原。"""
    orig = basics_model.scaled_dot_product_attention
    basics_model.scaled_dot_product_attention = flash_sdpa
    try:
        yield
    finally:
        basics_model.scaled_dot_product_attention = orig


def patch_flash_attention():
    """永久 patch(不还原)。分布式 worker 里用这个更省事。"""
    basics_model.scaled_dot_product_attention = flash_sdpa


def forward_hidden(model, x):
    """跑到 ln_final 为止, 返回隐藏态 [B, S, d_model], 不过 lm_head。

    复制自 BasicsTransformerLM.forward 的前半段(embedding → layers → ln_final),
    但**不算 logits**——把 lm_head 留给 fused_linear_cross_entropy 在 loss 里按 chunk 算,
    从而不 materialize [B, S, vocab] 全量 logits(#1 的核心收益)。

    layers 可能已被 #6 的 CheckpointedBlock 包过, 调用方式不变(都是 layer(x))。
    """
    x = model.token_embeddings(x)
    for layer in model.layers:
        x = layer(x)
    x = model.ln_final(x)
    return x


class HiddenModel(torch.nn.Module):
    """把 forward_hidden 变成一个 nn.Module 的 forward, 好让它作为 FSDP 包裹的顶层 module。

    为什么需要它(overlap 的关键):
      FSDP 的 prefetch(异步 all-gather 下几层 weight 与计算 overlap)依赖 inflight 队列,
      而该队列在**顶层 FSDP.forward** 里初始化。若直接调 forward_hidden(绕过顶层 forward),
      prefetch 初始化不执行 → 只能退回同步 gather(prefetch=0)。
      把 forward_hidden 包成本 module 的 forward 后, `fsdp(x)` 会先跑 FSDP.forward
      (触发 prefetch 初始化), 再进 self.base 的逐层前向 → 每层 _pre_forward 走 prefetch 分支,
      overlap 打开。

    lm_head 依旧不在这里过(留给外部 fused CE 手算), 所以 base.lm_head 仍应从 FSDP skip_modules 排除。
    """

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        return forward_hidden(self.base, x)
