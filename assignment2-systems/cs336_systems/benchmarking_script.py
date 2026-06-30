"""End-to-end benchmarking for forward / forward+backward / full train step."""
import argparse
import gc
import os
import sys
import timeit
import statistics
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def apply_segmented_checkpointing(model: BasicsTransformerLM, segment_size: int) -> None:
    if segment_size is None or segment_size <= 0:
        return

    blocks = list(model.layers)
    n = len(blocks)
    if segment_size > n:
        raise ValueError(f"segment_size={segment_size} > num_layers={n}")

    class _Segment(torch.nn.Module):
        def __init__(self, segment_blocks):
            super().__init__()
            self.blocks = torch.nn.ModuleList(segment_blocks)

        def _run(self, x):
            for blk in self.blocks:
                x = blk(x)
            return x

        def forward(self, x):
            # use_reentrant=False is the modern, hook-based implementation.
            return checkpoint(self._run, x, use_reentrant=False)

    segments = [
        _Segment(blocks[i : i + segment_size])
        for i in range(0, n, segment_size)
    ]
    model.layers = torch.nn.ModuleList(segments)

RUN_TYPES = ("forward", "forward_backward", "full")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument(
        "--run_type",
        type=str,
        choices=RUN_TYPES,
        default="full",
        help="forward | forward_backward | full",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--context_length", type=int, default=512)
    p.add_argument("--d_model", type=int, required=True)
    p.add_argument("--d_ff", type=int, required=True)
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--num_heads", type=int, required=True)
    p.add_argument(
        "--nsys",
        action="store_true",
        help="Gate the timed region with cudaProfilerApi (use with nsys --capture-range=cudaProfilerApi).",
    )
    p.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Run the forward pass under torch.autocast with BF16.",
    )
    p.add_argument(
        "--memory_profile",
        action="store_true",
        help="Record a CUDA memory history snapshot (load it in https://pytorch.org/memory_viz).",
    )
    p.add_argument(
        "--memory_snapshot",
        type=str,
        default=None,
        help="Output path for the memory snapshot pickle (default auto-named under reports/mem).",
    )
    p.add_argument(
        "--checkpoint_segment",
        type=int,
        default=0,
        help="If > 0, wrap every N consecutive transformer blocks in a single "
             "torch.utils.checkpoint() call (no nesting). 0 = no checkpointing.",
    )
    return p.parse_args()


def get_random_batch(args, device):
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    return x, y


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one_step(model, opt, x, y, run_type: str, amp_ctx, fwd_ctx=None):
    fwd_ctx = fwd_ctx if fwd_ctx is not None else nullcontext()
    with nvtx.range("forward"), fwd_ctx, amp_ctx:
        logits = model(x)
        if run_type == "forward":
            # Ensure the forward NVTX range covers the actual GPU work,
            # not just the kernel launches (important when profiling under nsys).
            if logits.is_cuda:
                torch.cuda.synchronize(logits.device)
            return
        loss = cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    with nvtx.range("backward"):
        loss.backward()
        # `loss.backward()` enqueues work onto an autograd worker thread and
        # returns immediately. Without an explicit synchronize, the NVTX range
        # (and any nsys `--capture-range` stop) can end before the GPU kernels
        # have actually run, producing a misleading <1us "backward" block.
        if loss.is_cuda:
            torch.cuda.synchronize(loss.device)
    if run_type == "full":
        with nvtx.range("optimizer"):
            opt.step()
            if loss.is_cuda:
                torch.cuda.synchronize(loss.device)


def run_memory_profile(model, opt, x, y, args, device, amp_ctx, fwd_ctx=None):
    """Record CUDA memory history over a few steps and dump a snapshot pickle."""
    if device.type != "cuda":
        raise RuntimeError("--memory_profile requires a CUDA device.")

    if args.memory_snapshot is not None:
        out_path = args.memory_snapshot
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(here, "..", "reports", "mem")
        os.makedirs(out_dir, exist_ok=True)
        prec = "bf16" if args.mixed_precision else "fp32"
        fname = f"mem_{args.run_type}_ctx{args.context_length}_{prec}.pickle"
        out_path = os.path.normpath(os.path.join(out_dir, fname))

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    try:
        for _ in range(args.steps):
            run_one_step(model, opt, x, y, args.run_type, amp_ctx, fwd_ctx)
        sync(device)
    finally:
        torch.cuda.memory._dump_snapshot(out_path)
        torch.cuda.memory._record_memory_history(enabled=None)

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    precision = "bf16" if args.mixed_precision else "fp32"
    seg = args.checkpoint_segment
    print(
        f"[memory_profile run_type={args.run_type} precision={precision} "
        f"ctx={args.context_length} ckpt_seg={seg}] steps={args.steps} "
        f"peak_allocated={peak:.2f} GiB"
    )
    print(f"  snapshot -> {out_path}")
    print("  Load it at https://pytorch.org/memory_viz (Active memory timeline).")


def main():
    args = get_args()
    device = torch.device(args.device)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    # Optional flat (non-nested) gradient checkpointing on the block stack.
    apply_segmented_checkpointing(model, args.checkpoint_segment)

    # Optimizer is built ONCE outside the loop so its state isn't reset.
    opt = AdamW(model.parameters())
    x, y = get_random_batch(args, device)

    if args.mixed_precision:
        amp_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    else:
        amp_ctx = nullcontext()

    if args.memory_profile and args.run_type == "forward":
        fwd_ctx = torch.no_grad()
    else:
        fwd_ctx = nullcontext()

    # ---- Warm-up ----
    for _ in range(args.warmup):
        run_one_step(model, opt, x, y, args.run_type, amp_ctx, fwd_ctx)
    sync(device)

    # ---- Memory profile (early return) ----
    if args.memory_profile:
        run_memory_profile(model, opt, x, y, args, device, amp_ctx, fwd_ctx)
        return

    # ---- Timed steps ----
    # Under nsys, only profile the timed region (skip warm-up noise).
    if args.nsys:
        torch.cuda.profiler.start()

    per_step = []
    for i in range(args.steps):
        t0 = timeit.default_timer()
        with nvtx.range(f"step_{i}"):
            run_one_step(model, opt, x, y, args.run_type, amp_ctx)
        sync(device)
        per_step.append(timeit.default_timer() - t0)

    if args.nsys:
        torch.cuda.profiler.stop()

    total = sum(per_step)
    mean = statistics.mean(per_step)
    std = statistics.stdev(per_step) if len(per_step) > 1 else 0.0

    precision = "bf16" if args.mixed_precision else "fp32"
    print(
        f"[run_type={args.run_type} precision={precision}] "
        f"steps={args.steps} warmup={args.warmup} "
        f"total={total:.4f}s  mean={mean*1000:.3f}ms  std={std*1000:.3f}ms"
    )


if __name__ == "__main__":
    main()
