from cs336_systems.dist_train import StateShardingOptimizer, DDP
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
import torch.distributed as dist
import torch.multiprocessing as mp
import os

XL = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
VOCAB, CTX, GLOBAL_BS = 10000, 512, 4
GPUS = 2
MB = 1024 ** 2
WARMUP, ITERS = 5, 10

def set_up(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def clean_up():
    dist.destroy_process_group()


def reduce_max(value, device):
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def worker(rank, is_sharding, ret_dict):
    set_up(rank, GPUS)
    device = f'cuda:{rank}'

    torch.cuda.reset_peak_memory_stats(device)
    model = BasicsTransformerLM(VOCAB, CTX, **XL).to(device)
    torch.cuda.synchronize()
    N = sum(p.numel() for p in model.parameters())
    peak_init = torch.cuda.max_memory_allocated(device)   
    cur_init = torch.cuda.memory_allocated(device)        

    if is_sharding:
        opt = StateShardingOptimizer(model.parameters(), AdamW)
    else:
        opt = AdamW(model.parameters())

    X = torch.randint(VOCAB, (GLOBAL_BS, CTX), device=device)
    Y = torch.randint(VOCAB, (GLOBAL_BS, CTX), device=device)

    torch.cuda.reset_peak_memory_stats(device)
    opt.zero_grad()
    loss = cross_entropy(model(X), Y)
    loss.backward()
    torch.cuda.synchronize()
    peak_before = torch.cuda.max_memory_allocated(device)  
    cur_before = torch.cuda.memory_allocated(device)       

    torch.cuda.reset_peak_memory_stats(device)
    opt.step()
    torch.cuda.synchronize()
    peak_after = torch.cuda.max_memory_allocated(device)   
    cur_after = torch.cuda.memory_allocated(device)        

    grad_mem = cur_before - cur_init         
    opt_state_mem = cur_after - cur_before   

    result = {
        'N': N,
        'peak_after_init': reduce_max(peak_init, device),
        'peak_before_step': reduce_max(peak_before, device),
        'peak_after_step': reduce_max(peak_after, device),
        'cur_after_init': reduce_max(cur_init, device),
        'cur_before_step': reduce_max(cur_before, device),
        'cur_after_step': reduce_max(cur_after, device),
        'grad_mem': reduce_max(grad_mem, device),
        'opt_state_mem': reduce_max(opt_state_mem, device),
    }
    if rank == 0:
        ret_dict.update(result)
    clean_up()

def worker_measure_time(rank, is_sharding, ret_dict):
    set_up(rank, GPUS)
    device = f'cuda:{rank}'

    # 每卡分到 global batch 的一份（DDP 数据并行）
    local_bs = GLOBAL_BS // GPUS
    _model = BasicsTransformerLM(VOCAB, CTX, **XL).to(device)
    model = DDP(_model, flat=True)          # 两种优化器都套同样的 DDP，差异只来自 optimizer
    opt = StateShardingOptimizer(model.parameters(), AdamW) if is_sharding else AdamW(model.parameters())

    X = torch.randint(VOCAB, (local_bs, CTX), device=device)
    Y = torch.randint(VOCAB, (local_bs, CTX), device=device)

    step_start = torch.cuda.Event(enable_timing=True)
    step_end = torch.cuda.Event(enable_timing=True)
    opt_start = torch.cuda.Event(enable_timing=True)
    opt_end = torch.cuda.Event(enable_timing=True)

    def one_step():
        step_start.record()
        opt.zero_grad()
        loss = cross_entropy(model(X), Y)
        loss.backward()
        model.finish_gradient_synchronization()   # 梯度 all-reduce
        opt_start.record()
        opt.step()
        opt_end.record()
        step_end.record()
        torch.cuda.synchronize()                  # 等 GPU 跑完再读 elapsed_time
        return step_start.elapsed_time(step_end), opt_start.elapsed_time(opt_end)

    # ---- warmup ----
    for _ in range(WARMUP):
        one_step()
    torch.cuda.synchronize()
    dist.barrier()

    # ---- 计时 ----
    step_times, opt_times = [], []
    for _ in range(ITERS):
        s, o = one_step()
        step_times.append(s)
        opt_times.append(o)

    t = torch.tensor([sum(step_times) / ITERS, sum(opt_times) / ITERS], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        ret_dict['avg_step_ms'] = t[0].item()
        ret_dict['avg_opt_ms'] = t[1].item()
    clean_up()

def fmt(ret):
    N = ret['N']
    params_theory = 4 * N
    print(f"  N (params)         : {N:,}")
    print(f"  --- peak (MB, includes activation) ---")
    print(f"  peak_after_init    : {ret['peak_after_init'] / MB:8.2f}")
    print(f"  peak_before_step   : {ret['peak_before_step'] / MB:8.2f}  (fwd+bwd peak)")
    print(f"  peak_after_step    : {ret['peak_after_step'] / MB:8.2f}")
    print(f"  --- current (MB) ---")
    print(f"  cur_after_init     : {ret['cur_after_init'] / MB:8.2f}  (params 4N = {params_theory / MB:.2f})")
    print(f"  cur_before_step    : {ret['cur_before_step'] / MB:8.2f}  (params+grad 8N = {2 * params_theory / MB:.2f})")
    print(f"  cur_after_step     : {ret['cur_after_step'] / MB:8.2f}")
    print(f"  --- breakdown (MB) ---")
    print(f"  grad_mem           : {ret['grad_mem'] / MB:8.2f}  (4N = {params_theory / MB:.2f})")
    print(f"  opt_state_mem      : {ret['opt_state_mem'] / MB:8.2f}  (8N = {2 * params_theory / MB:.2f})")

def print_mem(all_results):
    for is_sharding, ret in all_results.items():
        print(f"\n===== {'sharding' if is_sharding else 'non-sharding'} =====")
        fmt(ret)

    N = all_results[False]['N']
    print(f"\n[theory] optimizer states: full per rank 8N = {8 * N / MB:.2f} MB, "
          f"sharding per rank 8N/{GPUS} = {8 * N / GPUS / MB:.2f} MB")

def print_time(all_results):
    for is_sharding, ret in all_results.items():
        print(f"\n===== {'sharding' if is_sharding else 'non-sharding'} =====")
        print(f"  avg_step_ms        : {ret['avg_step_ms']:.3f} ms  (整步 fwd+bwd+sync+step)")
        print(f"  avg_opt_ms         : {ret['avg_opt_ms']:.3f} ms  (仅 optimizer.step)")

if __name__ == '__main__':
    all_results = {}
    for is_sharding in [False, True]:
        manager = mp.Manager()
        ret_dict = manager.dict()
        mp.spawn(worker_measure_time, args=(is_sharding, ret_dict), nprocs=GPUS, join=True)
        all_results[is_sharding] = dict(ret_dict)
    # print_mem(all_results)
    print_time(all_results)
   