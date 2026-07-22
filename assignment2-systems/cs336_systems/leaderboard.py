"""#7 端到端集成 + full-step 计时（8B 模型, 2×GPU）。

集成全部 6 项优化:
  #1 Fused CE + LM head   : forward 只出隐藏态, loss 用 fused_linear_cross_entropy 分块算, 不 materialize [N, vocab] logits
  #2 Fused AdamW          : 优化器换 FusedAdamW（单 Triton kernel 逐元素更新）
  #3/#4/#5 Flash attention: monkey-patch model.scaled_dot_product_attention 走 Triton Flash kernel
  #6 Activation ckpt      : 每个 TransformerBlock 包 checkpoint, 反向重算激活
  #7 FSDP(prefetch overlap): 参/梯/优化器状态按 2 卡切分; 前向走顶层 FSDP.forward, 异步 all-gather
                            后 2 层 weight 与本层计算 overlap(靠 HiddenModel 包装触发 prefetch 初始化)

运行(H20/B200, 2 卡):
    uv run python -m cs336_systems.leaderboard
可选环境变量:
    LB_REP=<次数>  计时迭代数(默认 5)   LB_WARMUP=<次数>  预热数(默认 3)
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.activation_ckpt import apply_activation_checkpointing
from cs336_systems.dist_train import FSDP
from cs336_systems.fused_adamw import FusedAdamW
from cs336_systems.fused_ce import fused_linear_cross_entropy
from cs336_systems.leaderboard_integration import (
    HiddenModel,
    patch_flash_attention,
)


class Config:
    vocab_size = 151936
    context_length = 32768
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    rope_theta = 10000.0


CFG = Config()
DTYPE = torch.bfloat16
GPUS = 2
GLOBAL_BS = 2
MASTER_PORT = "12358"


def build_model():
    return BasicsTransformerLM(
        vocab_size=CFG.vocab_size,
        context_length=CFG.context_length,
        d_model=CFG.d_model,
        num_layers=CFG.num_layers,
        num_heads=CFG.num_heads,
        d_ff=CFG.d_ff,
        rope_theta=CFG.rope_theta,
    )


def set_up(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def worker(rank):
    set_up(rank, GPUS)
    device = f"cuda:{rank}"
    local_bs = GLOBAL_BS // GPUS

    # #3/#4/#5: 全局把 attention 换成 Triton Flash（在建模型/前向之前 patch 即可）
    patch_flash_attention()

    # 建模型 → bf16 → 上卡
    base = build_model().to(device=device, dtype=DTYPE)

    # #6: 每个 TransformerBlock 包 activation checkpoint（放在 FSDP 之前, 让 FSDP 见到已包好的 block）
    apply_activation_checkpointing(base)

    # 包成 HiddenModel：前向出隐藏态。这样 fsdp(x) 会走顶层 FSDP.forward → 触发 prefetch 初始化 → overlap。
    model = HiddenModel(base)

    # #7: FSDP 切分静态状态。lm_head 不切——fused CE 手动读它的完整 weight, 绕过 forward hook。
    #     prefetch=2: 处理第 i 层时异步 all-gather 后面 2 层的 weight, 与本层计算 overlap。
    #     依赖前向走顶层 FSDP.forward(HiddenModel.forward)来初始化 inflight 队列。
    fsdp = FSDP(
        model,
        compute_dtype=None,
        prefetch=2,
        skip_modules={base.lm_head},
    )

    # #2: Fused AdamW（接口与 AdamW 一致, 只切到本 rank 的参数由 FSDP 保证 grad 已 reduce）
    optimizer = FusedAdamW(model.parameters(), lr=1e-4)

    # 造一份固定输入（labels 是输入 token, targets 是预测目标）
    torch.manual_seed(rank)
    labels = torch.randint(0, CFG.vocab_size, (local_bs, CFG.context_length), device=device)
    targets = torch.randint(0, CFG.vocab_size, (local_bs, CFG.context_length), device=device)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        # forward: 走顶层 FSDP.forward → HiddenModel.forward → forward_hidden, 出隐藏态
        #          [local_bs, S, d_model]（不 materialize 全量 logits）。prefetch overlap 在此生效。
        hidden = fsdp(labels)
        # loss: fused LM head + CE, 分块算, 峰值 [chunk, vocab]。lm_head.weight 完整(未被切)。
        loss = fused_linear_cross_entropy(hidden, base.lm_head.weight, targets)
        loss.backward()
        # FSDP: reduce-scatter 梯度到各分片
        fsdp.finish_gradient_synchronization()
        optimizer.step()
        return loss

    warmup = int(os.environ.get("LB_WARMUP", "3"))
    rep = int(os.environ.get("LB_REP", "5"))

    # 预热（触发 autotune / compile / cuBLAS 选 kernel）
    for _ in range(warmup):
        train_step()
    torch.cuda.synchronize()
    dist.barrier()

    # 端到端 full-step 计时
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(rep):
        train_step()
    torch.cuda.synchronize()
    dist.barrier()
    t1 = time.perf_counter()

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    step_ms = (t1 - t0) / rep * 1e3
    if rank == 0:
        print(
            f"[rank0] full-step (fwd+loss+bwd+AdamW) = {step_ms:.1f} ms/step "
            f"| peak mem = {peak_gb:.2f} GB | prefetch=2 | rep={rep}",
            flush=True,
        )
    else:
        print(f"[rank{rank}] peak mem = {peak_gb:.2f} GB", flush=True)

    dist.destroy_process_group()


def main():
    assert torch.cuda.is_available(), "需要 GPU"
    assert torch.cuda.device_count() >= GPUS, f"需要 >= {GPUS} 张卡"
    mp.spawn(worker, nprocs=GPUS, join=True)


if __name__ == "__main__":
    main()
