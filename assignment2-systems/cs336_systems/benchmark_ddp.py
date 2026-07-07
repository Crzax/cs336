import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.dist_train import DDP, AsyncDDP

XL = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
VOCAB, CTX, GLOBAL_BS = 10000, 512, 4
WARMUP, ITERS = 5, 10

# NSYS=1 时：开启 profiler 门控（配合 nsys --capture-range=cudaProfilerApi），并少跑几步
NSYS = os.environ.get("NSYS", "0") == "1"
# DDP_MODE 指定单个模式（per_param / flat / overlap）；不设则依次跑全部
ENV_MODE = os.environ.get("DDP_MODE", "")


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, ret_dict, mode):
    setup(rank, world_size)

    device = f"cuda:{rank}"
    _model = BasicsTransformerLM(VOCAB, CTX, **XL).to(device)
    if mode == "overlap":
        model = AsyncDDP(_model)          
    else:
        model = DDP(_model, flat=(mode == "flat"))  
    opt = AdamW(model.parameters())

    measure_comm = mode != "overlap"

    local_bs = GLOBAL_BS // world_size
    x = torch.randint(VOCAB, (local_bs, CTX), device=device)
    y = torch.randint(VOCAB, (local_bs, CTX), device=device)

    def one_step():
        step_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True) if measure_comm else None
        comm_end = torch.cuda.Event(enable_timing=True) if measure_comm else None

        step_start.record()
        opt.zero_grad(set_to_none=True)
        with nvtx.range("forward"):
            loss = cross_entropy(model(x), y)
        with nvtx.range("backward"):
            loss.backward()
        with nvtx.range("grad_sync"):
            if measure_comm:
                model.finish_gradient_synchronization(comm_start, comm_end)
            else:
                model.finish_gradient_synchronization()
        with nvtx.range("optimizer"):
            opt.step()
        step_end.record()
        torch.cuda.synchronize()

        step_ms = step_start.elapsed_time(step_end)
        comm_ms = comm_start.elapsed_time(comm_end) if measure_comm else float("nan")
        return step_ms, comm_ms

    # ---- warmup ----
    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()

    # ---- 计时 ----
    iters = 3 if NSYS else ITERS
    if NSYS:
        torch.cuda.profiler.start()   
    step_list, comm_list = [], []
    for i in range(iters):
        with nvtx.range(f"step_{i}"):
            s, c = one_step()
        step_list.append(s)
        comm_list.append(c)
    if NSYS:
        torch.cuda.profiler.stop()

    step_avg = sum(step_list) / iters
    comm_avg = sum(comm_list) / iters

    t = torch.tensor([step_avg, comm_avg], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    step_max, comm_max = t[0].item(), t[1].item()
    if rank == 0:
        ret_dict['ans'] = (step_max, comm_max)
    cleanup()


if __name__ == "__main__":
    manager = mp.Manager()

    modes = (ENV_MODE,) if ENV_MODE else ("per_param", "flat", "overlap")
    for mode in modes:
        return_dict = manager.dict()
        mp.spawn(
            worker,
            args=(2, return_dict, mode),
            nprocs=2,
            join=True,
        )
        step_max, comm_max = return_dict["ans"]
        if math.isnan(comm_max):
            print(f'[{mode:>9}] step_max={step_max:.3f} ms, comm=N/A (overlapped)')
        else:
            print(f'[{mode:>9}] step_max={step_max:.3f} ms, comm_max={comm_max:.3f} ms, '
                  f'comm_ratio={comm_max/step_max:.3f}')
