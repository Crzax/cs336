"""alternate_all_reduce 正确性测试：与 torch.distributed.all_reduce 对拍。"""
import os
import random

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_systems.tools import alternate_all_reduce

from .common import _cleanup_process_group


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("numel", [1, 7, 1024, 10_000])
def test_alternate_all_reduce_matches_builtin(world_size: int, numel: int):
    """在 CPU / gloo 上，与 dist.all_reduce 的结果对拍。"""
    # 每次用不同端口，规避端口占用/上一次测试没清理干净
    port = random.randint(20000, 40000)
    mp.spawn(
        _worker,
        args=(world_size, numel, port),
        nprocs=world_size,
        join=True,
    )


def _worker(rank: int, world_size: int, numel: int, port: int):
    import sys
    import traceback

    try:
        # 直接自己 init，绕过 _setup_process_group 里硬编码的端口 12390
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = "cpu"
        dist.barrier()

        # 每个 rank 造一份不同的数据
        torch.manual_seed(1000 + rank)
        x = torch.randn(numel, device=device)

        mine = x.clone()
        ref = x.clone()

        alternate_all_reduce(mine)  # 待测：原地
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)  # 参考：原地

        max_err = (mine - ref).abs().max().item()
        assert torch.allclose(mine, ref, atol=1e-5), (
            f"rank={rank}: max_err={max_err}\nmine={mine}\nref ={ref}"
        )

        _cleanup_process_group()
    except BaseException:
        print(f"\n===== rank {rank} traceback =====", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        raise
