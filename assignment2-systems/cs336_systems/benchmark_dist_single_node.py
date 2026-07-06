import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DATA_SIZES = [1, 10, 100, 1000]  # 单位 MB
NGPUS = [2, 4, 6]

WARMUP = 5
ITERS = 20


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def dist_test(rank, world_size, data_size, return_dict):
    setup(rank, world_size)

    # float32 -> 每个元素 4 字节
    numel = data_size * 1024 * 1024 // 4
    data = torch.randn(numel, device=f"cuda:{rank}")

    # ---- warmup ----
    for _ in range(WARMUP):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    torch.cuda.synchronize()

    # ---- 计时 ----
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # barrier 保证所有 rank 从同一起点开始计时
    dist.barrier()
    start.record()
    for _ in range(ITERS):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / ITERS  # 单次平均耗时 (ms)

    # 跨 rank 取最大值：集合操作的耗时由最慢的 rank 决定
    t = torch.tensor([elapsed_ms], device=f"cuda:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    max_ms = t.item()

    if rank == 0:
        return_dict[(world_size, data_size)] = max_ms
        print(f"world_size={world_size:>2}  size={data_size:>5}MB  "
              f"all_reduce={max_ms:8.3f} ms")

    cleanup()


if __name__ == "__main__":
    manager = mp.Manager()
    return_dict = manager.dict()

    for world_size in NGPUS:
        for data_size in DATA_SIZES:
            mp.spawn(
                dist_test,
                args=(world_size, data_size, return_dict),
                nprocs=world_size,
                join=True,
            )

    # ---- 汇总成表格 ----
    print("\n===== All-Reduce Benchmark (ms, avg over iters, max over ranks) =====")
    header = "size(MB)\\GPUs | " + " | ".join(f"{w:>10}" for w in NGPUS)
    print(header)
    print("-" * len(header))
    for size in DATA_SIZES:
        row = f"{size:>12} | "
        row += " | ".join(
            f"{return_dict.get((w, size), float('nan')):>10.3f}" for w in NGPUS
        )
        print(row)
