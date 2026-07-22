"""Activation checkpointing 包装（#6）。

用法（#7 集成阶段在 leaderboard 训练步里调）:
    from cs336_systems.activation_ckpt import apply_activation_checkpointing
    model = BasicsTransformerLM(...)
    apply_activation_checkpointing(model)   # 就地把每个 TransformerBlock 换成 checkpointed 版本

原理:
    - 不 checkpoint 时, 反向要用的每层中间激活 (attn 里的 QKV proj 输出、softmax 概率、
      SwiGLU 里的 silu(w1x)*w3x 等) 全部驻留 HBM, 34 层累起来 200+GB, 8B 模型塞不下 seq=32768。
    - checkpoint 后, 前向只保存每个 block 的 **输入张量** ([B, S, d_model] bf16 = 512MB @ 2×32k×4096),
      block 内所有中间激活全丢弃; 反向时对该 block 重新前向一遍(在 no_grad 模式下建计算图),
      再走标准 autograd, 用完立刻释放。
    - 34 层输入张量共 34 × 512MB ≈ 17GB, 反向时同一时刻只有 1 层的完整激活活着(~10GB),
      峰值大幅降低。代价是一次额外前向, 但相比"塞不下 → OOM"完全可接受。

设计:
    - use_reentrant=False: 新 API, 不走 autograd re-entry, 支持任意输入/输出结构、
      自动保存 RNG state、和 torch.compile 兼容。老的 True 版本对 kwargs/non-tensor 输入敏感,
      且不能与 compile 良好配合, 直接淘汰。
    - 通过 nn.Module 包装(而非 monkey-patch forward): 保留原 block 作为 self.block 子模块,
      state_dict、参数遍历、named_parameters 全都自然工作, 不用特殊处理。
    - 只包 TransformerBlock, 不碰 embedding / final norm / lm_head: 前者输入是 int64
      (checkpoint 不能处理 int 张量 requires_grad), 后者算完即弃或后接 fused CE 本身就省了。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CheckpointedBlock(nn.Module):
    """把一个 TransformerBlock 包成 activation-checkpoint 版。"""

    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use_reentrant=False 用新 autograd 图机制, 兼容 compile / 无 kwargs 限制
        return checkpoint(self.block, x, use_reentrant=False)


def apply_activation_checkpointing(model: nn.Module) -> nn.Module:
    """就地把 model.layers 里每个 TransformerBlock 换成 CheckpointedBlock。

    Args:
        model: BasicsTransformerLM 实例（必须有 .layers: nn.ModuleList）。

    Returns:
        model 本身（就地修改, 返回值仅方便链式调用）。
    """
    assert hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList), (
        "expect BasicsTransformerLM with .layers: nn.ModuleList"
    )
    for i, block in enumerate(model.layers):
        # 如果已经被包过(重复调用), 跳过
        if isinstance(block, CheckpointedBlock):
            continue
        model.layers[i] = CheckpointedBlock(block)
    return model
