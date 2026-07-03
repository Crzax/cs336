import argparse
import itertools

import torch
import triton

from cs336_systems.opt_kernel import FlashAttnFuncTriton

BATCH = 1
DEFAULT_SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
DEFAULT_DIMS = [16, 32, 64, 128]
DEFAULT_DTYPES = ["bf16", "fp32"]

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def pytorch_attention(Q, K, V, is_causal=True):
    d = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-1, -2)) / (d ** 0.5)
    if is_causal:
        nq, nk = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(
            torch.ones(nq, nk, device=scores.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def make_inputs(seq_len, d, dtype, device="cuda"):
    q = torch.randn(BATCH, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(BATCH, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(BATCH, seq_len, d, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(BATCH, seq_len, d, device=device, dtype=dtype)
    return q, k, v, do


def bench_impl(fn_forward, q, k, v, do):

    def fwd():
        return fn_forward(q, k, v, True)

    def e2e():
        for t in (q, k, v):
            t.grad = None
        out = fn_forward(q, k, v, True)
        out.backward(do)

    try:
        fwd_ms = triton.testing.do_bench(fwd)
    except Exception:
        return None

    # backward-only：先算好 out，计时只跑 backward。用 retain_graph 让多次计时可复用图。
    def bwd():
        for t in (q, k, v):
            t.grad = None
        out.backward(do, retain_graph=True)

    try:
        out = fn_forward(q, k, v, True)
        bwd_ms = triton.testing.do_bench(bwd)
    except Exception as e:
        print(f"    [backward FAILED] {type(e).__name__}: {str(e)[:200]}", flush=True)
        bwd_ms = float("nan")

    try:
        e2e_ms = triton.testing.do_bench(e2e)
    except Exception as e:
        print(f"    [e2e FAILED] {type(e).__name__}: {str(e)[:200]}", flush=True)
        e2e_ms = float("nan")

    return fwd_ms, bwd_ms, e2e_ms


def run(seq_lens, dims, dtypes):
    rows = []
    for dtype_name, d, seq_len in itertools.product(dtypes, dims, seq_lens):
        dtype = _DTYPE_MAP[dtype_name]

        # --- Triton ---
        try:
            q, k, v, do = make_inputs(seq_len, d, dtype)
            tri = bench_impl(FlashAttnFuncTriton.apply, q, k, v, do)
        except torch.cuda.OutOfMemoryError:
            tri = None
        torch.cuda.empty_cache()

        # --- PyTorch 参照 ---
        try:
            q, k, v, do = make_inputs(seq_len, d, dtype)
            ref = bench_impl(pytorch_attention, q, k, v, do)
        except torch.cuda.OutOfMemoryError:
            ref = None
        torch.cuda.empty_cache()

        rows.append((dtype_name, d, seq_len, tri, ref))
        _print_progress(dtype_name, d, seq_len, tri, ref)

    return rows


def _fmt(triple, idx):
    if triple is None:
        return "OOM"
    val = triple[idx]
    if val != val:  # nan
        return "-"
    return f"{val:.3f}"


def _print_progress(dtype_name, d, seq_len, tri, ref):
    print(
        f"[{dtype_name} d={d} seq={seq_len}] "
        f"triton fwd/bwd/e2e = {_fmt(tri,0)}/{_fmt(tri,1)}/{_fmt(tri,2)} ms | "
        f"pytorch = {_fmt(ref,0)}/{_fmt(ref,1)}/{_fmt(ref,2)} ms",
        flush=True,
    )


def print_table(rows):
    header = (
        "| dtype | d | seq_len | Triton fwd (ms) | Triton bwd (ms) | Triton e2e (ms) "
        "| PyTorch fwd (ms) | PyTorch bwd (ms) | PyTorch e2e (ms) |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    print("\n" + header)
    print(sep)
    for dtype_name, d, seq_len, tri, ref in rows:
        print(
            f"| {dtype_name} | {d} | {seq_len} "
            f"| {_fmt(tri,0)} | {_fmt(tri,1)} | {_fmt(tri,2)} "
            f"| {_fmt(ref,0)} | {_fmt(ref,1)} | {_fmt(ref,2)} |"
        )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", type=str, default=None,
                   help="逗号分隔，如 128,256,512")
    p.add_argument("--dims", type=str, default=None, help="逗号分隔，如 16,64")
    p.add_argument("--dtypes", type=str, default=None, help="逗号分隔，bf16,fp32")
    return p.parse_args()


def main():
    args = parse_args()
    seq_lens = ([int(x) for x in args.seq_lens.split(",")]
                if args.seq_lens else DEFAULT_SEQ_LENS)
    dims = ([int(x) for x in args.dims.split(",")]
            if args.dims else DEFAULT_DIMS)
    dtypes = (args.dtypes.split(",") if args.dtypes else DEFAULT_DTYPES)

    assert torch.cuda.is_available(), "需要 GPU"
    rows = run(seq_lens, dims, dtypes)
    print_table(rows)


if __name__ == "__main__":
    main()
