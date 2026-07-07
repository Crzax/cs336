import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.dist_train import DDP

XL = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
VOCAB, CTX, GLOBAL_BS = 10000, 512, 4
WARMUP, ITERS = 5, 10

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, ret_dict):
    setup(rank, world_size)

    device = f"cuda:{rank}"
    _model = BasicsTransformerLM(VOCAB, CTX, **XL).to(device)
    model = DDP(_model)
    opt = AdamW(model.parameters())

    local_bs = GLOBAL_BS // world_size
    x = torch.randint(VOCAB, (local_bs, CTX), device=device)
    y = torch.randint(VOCAB, (local_bs, CTX), device=device)

    # ---- warmup ----
    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(model(x), y)
        loss.backward()
        model.finish_gradient_synchronization()
        opt.step()
    torch.cuda.synchronize()

    def step():
        step_start = torch.cuda.Event(enable_timing=True)
        comm_start = torch.cuda.Event(enable_timing=True)
        step_end = torch.cuda.Event(enable_timing=True)
        comm_end = torch.cuda.Event(enable_timing=True)
        step_start.record()
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(model(x), y)
        loss.backward()
        comm_start.record()
        model.finish_gradient_synchronization()
        comm_end.record()

        opt.step()
        step_end.record()
        torch.cuda.synchronize()
        step_ms = step_start.elapsed_time(step_end)
        comm_ms = comm_start.elapsed_time(comm_end)
        return step_ms, comm_ms
    # ---- 计时 ----
    step_list, comm_list = [], []
    for _ in range(ITERS):
        dist.barrier()
        s, c = step()
        step_list.append(s)
        comm_list.append(c)
    step_avg = sum(step_list) / ITERS
    comm_avg = sum(comm_list) / ITERS

    t = torch.tensor([step_avg, comm_avg], device=f"cuda:{rank}")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    step_max, comm_max = t[0].item(), t[1].item()
    if rank == 0:
        ret_dict['ans']=(step_max, comm_max)
    cleanup()


if __name__ == "__main__":
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        worker,
        args=(2, return_dict),
        nprocs=2,
        join=True,
    )

    print(f'step_max={return_dict["ans"][0]:.3f} ms, comm_max={return_dict["ans"][1]:.3f} ms, '
          f'comm_ratio={return_dict["ans"][1]/return_dict["ans"][0]:.3f}')
    
