"""
Training entry point.

Usage example:
    python -m cs336_basics.train \
        --train_data /path/to/train.npy \
        --val_data   /path/to/val.npy   \
        --vocab_size 10000 --context_length 256 \
        --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 \
        --batch_size 32 --total_steps 5000 \
        --lr_max 3e-4 --warmup_steps 100 --cosine_steps 5000 \
        --ckpt_dir runs/exp1 --log_interval 50 --eval_interval 500 \
        --device cuda:0 --wandb
"""
import argparse
import os
import time
import math
import json
from pathlib import Path

import numpy as np
import torch

from cs336_basics.nn.nn_transformer import TransformerLM
from cs336_basics.nn.nn_basic import cross_entropy
from cs336_basics.opt import AdamW, learning_rate_schedule, gradient_clipping
from cs336_basics.data import data_loading, save_checkpoint, load_checkpoint


# ---------- 1. CLI: 控制所有超参（满足要求 1） ----------
def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--train_data", type=str, required=True, help=".npy file of token ids")
    p.add_argument("--val_data",   type=str, required=True)
    p.add_argument("--data_dtype", type=str, default="uint16",
                   help="dtype used to save the token .npy/.bin file (uint16 / int32 ...)")
    # model
    p.add_argument("--vocab_size",     type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--num_heads",      type=int, default=16)
    p.add_argument("--d_ff",           type=int, default=None,
                   help="default = round(8/3 * d_model) to a multiple of 64")
    p.add_argument("--rope_theta",     type=float, default=10000.0)
    # optim
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--total_steps",  type=int,   default=5000)
    p.add_argument("--lr_max",       type=float, default=3e-4)
    p.add_argument("--lr_min",       type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int,   default=100)
    p.add_argument("--cosine_steps", type=int,   default=5000,
                   help="step at which lr reaches lr_min (the 'cosine_steps' in your schedule)")
    p.add_argument("--betas",        type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eps",          type=float, default=1e-8)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    # io / logging
    p.add_argument("--ckpt_dir",       type=str, default="runs/default")
    p.add_argument("--ckpt_interval",  type=int, default=1000)
    p.add_argument("--log_interval",   type=int, default=50)
    p.add_argument("--eval_interval",  type=int, default=500)
    p.add_argument("--eval_batches",   type=int, default=20,
                   help="how many batches to average for validation loss")
    p.add_argument("--resume",         type=str, default=None,
                   help="path to a checkpoint to resume from")
    p.add_argument("--wandb",          action="store_true")
    p.add_argument("--wandb_entity",   type=str, default="sglang-vllm-pku",
                   help="wandb team / user name")
    p.add_argument("--wandb_project",  type=str, default="cs336-a1")
    p.add_argument("--wandb_run_name", type=str, default=None)
    # misc
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--dtype",  type=str, default="float32",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


# ---------- 2. 数据：np.memmap 懒加载（满足要求 2） ----------
def load_dataset(path: str, dtype: str) -> np.ndarray:
    """
    Lazy-load token ids from disk:
    - .npy: use np.load(mmap_mode='r')   (dtype 已经写在 .npy header 里)
    - .bin: use np.memmap(..., dtype=...) 必须显式给 dtype
    Both return an array-like that supports fancy indexing without
    pulling the whole file into RAM.
    """
    p = Path(path)
    if p.suffix == ".npy":
        arr = np.load(p, mmap_mode="r")
    else:
        arr = np.memmap(p, dtype=np.dtype(dtype), mode="r")
    # sanity check：扫一段确认 dtype/vocab_size 没写错
    sample = np.asarray(arr[: min(1_000_000, arr.size)])
    print(f"[data] {path}  len={arr.size:,}  dtype={arr.dtype}  "
          f"min={sample.min()}  max={sample.max()}")
    return arr


# ---------- 3. 评估循环 ----------
@torch.no_grad()
def evaluate(model, val_data, args, n_batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = data_loading(val_data, args.batch_size, args.context_length, args.device)
        logits = model(x)                              # (B, L, V)
        loss = cross_entropy(logits, y)                # 返回标量
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


# ---------- 4. 主训练循环 ----------
def main():
    args = get_args()

    # 4.1 复现性 & 设备
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    # 4.2 数据
    train_data = load_dataset(args.train_data, args.data_dtype)
    val_data   = load_dataset(args.val_data,   args.data_dtype)

    # 4.3 模型
    model = TransformerLM(
        d_model=args.d_model, num_layers=args.num_layers, num_heads=args.num_heads,
        d_ff=args.d_ff, vocab_size=args.vocab_size,
        max_seq_len=args.context_length, theta=args.rope_theta,
        device=device, dtype=dtype,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params")

    # 4.4 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max, betas=tuple(args.betas),
        eps=args.eps, weight_decay=args.weight_decay,
    )

    # 4.5 可选：从 ckpt 恢复
    start_step = 0
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    # 把本次运行的所有超参 dump 到 ckpt_dir/config.json，方便复现
    with open(Path(args.ckpt_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"[resume] from step {start_step}")

    # 4.6 可选：wandb（满足要求 4）
    # 注意：wandb 默认只上传 run.log() 的 scalar metrics；
    # 不会自动上传 checkpoint / 代码（除非显式调 wandb.save/Artifact）
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            save_code=False,         # 不打包代码上传
        )

    # 4.7 训练
    model.train()
    t0 = time.time()
    for step in range(start_step, args.total_steps):
        # (a) 学习率
        lr_now = learning_rate_schedule(
            step, args.lr_max, args.lr_min, args.warmup_steps, args.cosine_steps
        )
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        # (b) 一个 batch
        x, y = data_loading(train_data, args.batch_size, args.context_length, device)

        # (c) forward + backward
        logits = model(x)
        loss = cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # (d) 梯度裁剪（拿到裁剪前的 total L2 norm 用于日志）
        grad_norm = gradient_clipping(model.parameters(), args.grad_clip)

        # (e) 更新
        optimizer.step()

        # (f) 日志（满足要求 4）
        if step % args.log_interval == 0:
            tok_per_step = args.batch_size * args.context_length
            wall_time_s = time.time() - t0
            tps = tok_per_step * (step - start_step + 1) / max(wall_time_s, 1e-9)
            gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
            msg = (f"step {step:>6d}  t {wall_time_s:>7.1f}s  lr {lr_now:.2e}  "
                   f"train_loss {loss.item():.4f}  ppl {math.exp(min(loss.item(), 20)):.2f}  "
                   f"gnorm {gn:.3f}  tok/s {tps:,.0f}")
            print(msg, flush=True)
            if use_wandb:
                wandb.log({
                    "train/loss":      loss.item(),
                    "train/ppl":       math.exp(min(loss.item(), 20)),
                    "train/lr":        lr_now,
                    "train/grad_norm": gn,
                    "train/tok_per_s": tps,
                    "wall_time_s":     wall_time_s,
                }, step=step)

        # (g) 验证
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args, args.eval_batches)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  [eval] step {step}  val_loss {val_loss:.4f}  val_ppl {val_ppl:.2f}", flush=True)
            if use_wandb:
                wandb.log({
                    "val/loss":    val_loss,
                    "val/ppl":     val_ppl,
                    "wall_time_s": time.time() - t0,
                }, step=step)

        # (h) 保存 checkpoint（满足要求 3）
        if step > 0 and step % args.ckpt_interval == 0:
            ckpt_path = Path(args.ckpt_dir) / f"step_{step}.pt"
            save_checkpoint(model, optimizer, step, str(ckpt_path))
            print(f"  [ckpt] saved -> {ckpt_path}", flush=True)

    # 最终再存一次
    final_path = Path(args.ckpt_dir) / "final.pt"
    save_checkpoint(model, optimizer, args.total_steps, str(final_path))
    print(f"[done] total time {(time.time() - t0) / 60:.1f} min, final ckpt -> {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
