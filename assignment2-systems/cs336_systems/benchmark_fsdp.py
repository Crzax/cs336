"""Profile FSDP weight all-gather on the xl model across 2 GPUs.

Run under Nsight Systems:
  uv run nsys profile -o fsdp_xl --trace=cuda,nvtx,osrt --force-overwrite true \
      python -m cs336_systems.benchmark_fsdp

Open fsdp_xl.nsys-rep and, per layer, compare the "fsdp.all_gather_weight"
NVTX range (or the NCCL AllGather kernel) with that layer's forward matmul
kernels to see whether the gather finishes in time / overlaps with compute.
"""

import os

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.dist_train import FSDP

XL = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
VOCAB, CTX, GLOBAL_BS, GPUS = 10000, 512, 4, 2
WARMUP, ITERS = 3, 5
COMPUTE_DTYPE = None  # or torch.bfloat16 to profile mixed precision
PREFETCH = 2          # 0 = synchronous (no overlap); >0 = gather this many layers ahead


def set_up(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def worker(rank):
    set_up(rank, GPUS)
    device = f"cuda:{rank}"
    local_bs = GLOBAL_BS // GPUS

    model = BasicsTransformerLM(VOCAB, CTX, **XL).to(device)
    model = FSDP(model, compute_dtype=COMPUTE_DTYPE, prefetch=PREFETCH)
    opt = torch.optim.AdamW(model.parameters())

    X = torch.randint(VOCAB, (local_bs, CTX), device=device)
    Y = torch.randint(VOCAB, (local_bs, CTX), device=device)

    def one_step(tag=None):
        opt.zero_grad(set_to_none=True)
        if tag:
            nvtx.range_push(f"{tag}.forward")
        loss = cross_entropy(model(X), Y)
        if tag:
            nvtx.range_pop()
            nvtx.range_push(f"{tag}.backward")
        loss.backward()
        if tag:
            nvtx.range_pop()
        model.finish_gradient_synchronization()
        opt.step()

    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()

    for i in range(ITERS):
        nvtx.range_push(f"step{i}")
        one_step(tag=f"step{i}")
        nvtx.range_pop()
    torch.cuda.synchronize()

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, nprocs=GPUS, join=True)
