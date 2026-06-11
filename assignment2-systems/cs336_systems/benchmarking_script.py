"""End-to-end benchmarking for forward / forward+backward / full train step."""
import argparse
import gc
import sys
import timeit
import statistics

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


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
    return p.parse_args()


def get_random_batch(args, device):
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    return x, y


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_one_step(model, opt, x, y, run_type: str):
    logits = model(x)
    if run_type == "forward":
        return
    loss = cross_entropy(logits, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if run_type == "full":
        opt.step()


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

    # Optimizer is built ONCE outside the loop so its state isn't reset.
    opt = AdamW(model.parameters())
    x, y = get_random_batch(args, device)

    # ---- Warm-up ----
    for _ in range(args.warmup):
        run_one_step(model, opt, x, y, args.run_type)
    sync(device)

    # ---- Timed steps ----
    per_step = []
    for _ in range(args.steps):
        t0 = timeit.default_timer()
        run_one_step(model, opt, x, y, args.run_type)
        sync(device)
        per_step.append(timeit.default_timer() - t0)

    total = sum(per_step)
    mean = statistics.mean(per_step)
    std = statistics.stdev(per_step) if len(per_step) > 1 else 0.0

    print(
        f"[run_type={args.run_type}] "
        f"steps={args.steps} warmup={args.warmup} "
        f"total={total:.4f}s  mean={mean*1000:.3f}ms  std={std*1000:.3f}ms"
    )


if __name__ == "__main__":
    main()
