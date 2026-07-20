"""对比 alternate_all_reduce vs torch 内置 all-reduce 的耗时。

用法:
    # CPU/gloo 冒烟
    python assignment2-systems/scripts/bench_all_reduce.py --world-size 4 --size-mb 16 --backend gloo

    # 单机多卡 NCCL
    python assignment2-systems/scripts/bench_all_reduce.py --world-size 4 --size-mb 256 --backend nccl

理论预期：world_size=N 时，naive 版应该比内置版慢约 N/2 倍。
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.tools import alternate_all_reduce


def _setup(rank: int, world_size: int, backend: str, port: int) -> str:
    """自己实现，绕开 tests.common 里硬编码的端口 12390。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    if backend == "nccl":
        assert torch.cuda.is_available(), "nccl backend requires CUDA"
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def _sync(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _time_fn(fn, x_template: torch.Tensor, iters: int, device: str, name: str, rank: int):
    # warmup
    for _ in range(3):
        buf = x_template.clone()
        fn(buf)
    _sync(device)
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        buf = x_template.clone()
        fn(buf)
    _sync(device)
    dist.barrier()
    dt = (time.perf_counter() - t0) / iters * 1000  # ms/iter

    if rank == 0:
        print(f"  {name:30s}  {dt:8.2f} ms/iter")


def _bench(rank: int, world_size: int, size_mb: int, iters: int, backend: str, port: int):
    import sys, traceback

    try:
        device = _setup(rank, world_size, backend, port)
        dist.barrier()

        numel = size_mb * 1024 * 1024 // 4  # float32
        x = torch.randn(numel, device=device)

        if rank == 0:
            print(f"[world_size={world_size}, size={size_mb} MB, backend={backend}, port={port}]")

        _time_fn(alternate_all_reduce, x, iters, device, "alternate_all_reduce (mine)", rank)
        _time_fn(
            lambda t: dist.all_reduce(t, op=dist.ReduceOp.SUM),
            x,
            iters,
            device,
            "dist.all_reduce (builtin)",
            rank,
        )

        dist.barrier()
        dist.destroy_process_group()
    except BaseException:
        print(f"\n===== rank {rank} traceback =====", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--size-mb", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backend", default="gloo", choices=["gloo", "nccl"])
    p.add_argument("--port", type=int, default=0, help="0 = random")
    args = p.parse_args()

    port = args.port or random.randint(20000, 40000)

    mp.spawn(
        _bench,
        args=(args.world_size, args.size_mb, args.iters, args.backend, port),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
