import argparse
from ast import main
import torch, itertools, time, gc
from einops import einsum
from cs336_basics.nn_utils import softmax
BATCH = 8
D_MODELS = [16, 32, 64, 128]
SEQ_LENS = [256, 1024, 4096, 8192, 16384, 32768]
N_ITERS = 100
WARMUP = 3
device = 'cuda'

def get_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--compile", action="store_true")
    return p.parse_args()

def _attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = einsum(Q, K, '... seq_q d_k, ... seq_k d_k -> ... seq_q seq_k') / d_k**0.5
    scores = softmax(scores, dim=-1)
    return einsum(scores, V, '... seq_q seq_k, ... seq_k d_v -> ... seq_q d_v')

def main():
    compile = get_parse().compile
    
    if compile:
        attention = torch.compile(_attention)
    else:
        attention = _attention
        
    results = []
    for d_model, seq_len in itertools.product(D_MODELS, SEQ_LENS):
        try:
            # 创建输入，requires_grad=True 才能 backward
            Q = torch.randn(BATCH, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(BATCH, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(BATCH, seq_len, d_model, device=device, requires_grad=True)

            # ---- warmup ----
            for _ in range(WARMUP):
                out = attention(Q, K, V)
                loss = out.sum()
                loss.backward()
            
            torch.cuda.synchronize()
            # 清理 warmup 残留
            Q.grad = None
            K.grad = None
            V.grad = None
            torch.cuda.empty_cache()

            t0 = time.time()
            for _ in range(N_ITERS):
                out = attention(Q, K, V)
                torch.cuda.synchronize()
            fwd_time = (time.time() - t0) / N_ITERS

            mem_before_bwd = torch.cuda.memory_allocated() / 1024**2  # MB

            # ---- backward 100 次 ----
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(N_ITERS):
                out = attention(Q, K, V)
                loss = out.sum()
                loss.backward()
                torch.cuda.synchronize()
            bwd_time = (time.time() - t0) / N_ITERS - fwd_time  # 减掉 fwd 部分

            results.append((d_model, seq_len, fwd_time*1000, bwd_time*1000, mem_before_bwd))

        except torch.cuda.OutOfMemoryError:
            results.append((d_model, seq_len, 'OOM', 'OOM', 'OOM'))
        finally:
            # 清理，否则下一轮会污染
            gc.collect()
            torch.cuda.empty_cache()
    
    print('|d_model|seq_len|fwd_time|bwd_time|mem_before_bwd(MB)|')
    print('|---|---|---|---|---|')
    for result in results:
        if result[2] != 'OOM':
            print(f'|{result[0]}|{result[1]}|{result[2]:.2f}|{result[3]:.2f}|{result[4]:.2f}|')
        else:
            print(f'|{result[0]}|{result[1]}|{result[2]}|{result[3]}|{result[4]}|')
    print('|---|---|---|---|---|')

if __name__ == '__main__':
    main()