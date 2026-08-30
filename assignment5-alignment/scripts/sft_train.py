"""Supervised fine-tuning of Llama 3.1 8B on instruction-tuning data (supplement 4.2).

Full-parameter SFT on a single GPU, entirely in bf16 (params, grads and AdamW
states), with gradient checkpointing, gradient accumulation, cosine LR decay
with linear warmup and gradient clipping. No Hugging Face Trainer.

The packed SFT dataset (cs336_alignment.sft) already provides next-token
labels, so the loss is computed directly from the logits instead of passing
`labels=` to the model (which would shift a second time).

Example (single H20 98GB):

```
uv run python scripts/sft_train.py \
    --model /mnt/cephfs/user_crzaxchen/models/Meta-Llama-3.1-8B \
    --train-data data/sft/train.jsonl \
    --val-data data/sft/test.jsonl \
    --output-dir scripts/results/sft_llama31_8b
```

Memory budget (8B params, bf16): weights 16 GB + grads 16 GB + AdamW m/v 32 GB
= 64 GB, leaving ~30 GB for activations; with gradient checkpointing a
per-device batch of 4 x 512 tokens fits comfortably.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.sft import PackedSFTDataset

try:
    import wandb
except ImportError:  # wandb lives in the `gpu` extra; allow running without it
    wandb = None

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: str) -> float:
    """Token-weighted average cross-entropy over the validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids).logits
        loss_sum = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
        )
        total_loss += loss_sum.item()
        total_tokens += labels.numel()
    model.train()
    return total_loss / total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--train-data", type=Path, default=REPO_ROOT / "data" / "sft" / "train.jsonl")
    parser.add_argument("--val-data", type=Path, default=REPO_ROOT / "data" / "sft" / "test.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "scripts" / "results" / "sft_llama31_8b")
    # Model / data hyperparameters.
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="effective batch = per-device batch * this; 4*8=32 sequences/step")
    # Optimizer hyperparameters.
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="fraction of total optimizer steps for linear warmup")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--adam-betas", type=float, nargs=2, default=(0.9, 0.999))
    # Logging / evaluation / saving.
    parser.add_argument("--log-every", type=int, default=10, help="optimizer steps between train logs")
    parser.add_argument("--eval-every", type=int, default=200, help="optimizer steps between val evaluations")
    parser.add_argument("--max-val-seqs", type=int, default=1024,
                        help="cap on packed validation sequences per evaluation (0 = all)")
    parser.add_argument("--save-every", type=int, default=0,
                        help="save intermediate checkpoint every N optimizer steps (0 = only at end)")
    # Debug knobs.
    parser.add_argument("--max-train-docs", type=int, default=0, help="cap training documents (0 = all)")
    parser.add_argument("--max-steps", type=int, default=0, help="cap optimizer steps (0 = full epoch(s))")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging (console/JSONL still work)")
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda:0"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"

    # ------------------------------------------------------------------ data
    logger.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def load_documents(path: Path, cap: int) -> tuple[list[dict], int]:
        """Return (documents, total document count before any cap)."""
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as f:
            docs = [json.loads(line) for line in f if line.strip()]
        total = len(docs)
        if cap:
            docs = docs[:cap]
        logger.info("Loaded %d documents from %s", len(docs), path)
        return docs, total

    train_docs, n_train_total = load_documents(args.train_data, args.max_train_docs)
    val_docs, _ = load_documents(args.val_data, 0)

    # PackedSFTDataset reads documents from a path; use the original file
    # directly, and only materialize a truncated copy when capping for debug.
    def materialize(docs: list[dict], original: Path, name: str, original_count: int) -> Path:
        if len(docs) == original_count:
            return original
        truncated = args.output_dir / "_data" / name
        truncated.parent.mkdir(parents=True, exist_ok=True)
        with truncated.open("w", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        return truncated

    train_path = materialize(train_docs, args.train_data, "train.jsonl", n_train_total)
    val_path = materialize(val_docs, args.val_data, "val.jsonl", len(val_docs))

    logger.info("Tokenizing and packing training data (seq_length=%d)", args.seq_length)
    train_dataset = PackedSFTDataset(tokenizer, train_path, args.seq_length, shuffle=True)
    logger.info("Packed train: %d sequences (%d tokens)", len(train_dataset),
                len(train_dataset) * args.seq_length + 1)
    logger.info("Tokenizing and packing validation data")
    val_dataset = PackedSFTDataset(tokenizer, val_path, args.seq_length, shuffle=False)
    if args.max_val_seqs and len(val_dataset) > args.max_val_seqs:
        val_dataset = Subset(val_dataset, range(args.max_val_seqs))

    train_loader = DataLoader(
        train_dataset, batch_size=args.per_device_batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_batch_size, shuffle=False)
    logger.info("%d microbatches/epoch, %d val batches per eval",
                len(train_loader), len(val_loader))

    # ----------------------------------------------------------------- model
    logger.info("Loading model from %s (bf16, %s)", args.model, args.attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.config.use_cache = False
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    # ------------------------------------------------------------- optimizer
    microbatches_per_epoch = len(train_loader)
    steps_per_epoch = math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=tuple(args.adam_betas),
        weight_decay=args.weight_decay,
        fused=True,
    )
    # Cosine decay to 0 with linear warmup (same shape as the usual
    # transformers schedule, implemented locally to stay version-proof).
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    hyperparams = {
        "model": str(args.model),
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "n_train_docs": len(train_docs),
        "n_train_sequences": len(train_dataset),
        "seq_length": args.seq_length,
        "epochs": args.epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_sequences": args.per_device_batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "attn_implementation": args.attn_implementation,
    }
    logger.info("Hyperparameters: %s", json.dumps(hyperparams, indent=2))

    if not args.no_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed (it is in the `gpu` extra); use --no-wandb or `uv sync --extra gpu`")
        wandb.init(
            project="cs336-a5-supplement-sft",
            name=args.output_dir.name,
            config=hyperparams,
        )

    # ------------------------------------------------------------ train loop
    start_time = time.perf_counter()
    optimizer_step = 0
    microbatch_idx = 0            # microbatches since last optimizer.step()
    recent_losses: list[float] = []  # losses of the current accumulation window
    tokens_seen = 0
    best_val_loss = float("inf")
    stop_training = False

    def run_optimizer_step() -> None:
        nonlocal optimizer_step, microbatch_idx, recent_losses, best_val_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        window_loss = sum(recent_losses) / len(recent_losses)
        recent_losses = []
        microbatch_idx = 0

        elapsed = time.perf_counter() - start_time
        tokens_per_sec = tokens_seen / elapsed if elapsed > 0 else 0.0
        remaining_tokens = (total_steps - optimizer_step) * (
            args.per_device_batch_size * args.gradient_accumulation_steps * args.seq_length
        )
        eta_h = remaining_tokens / tokens_per_sec / 3600 if tokens_per_sec > 0 else float("nan")
        mem_gb = torch.cuda.max_memory_allocated() / 2**30

        if optimizer_step % args.log_every == 0:
            msg = (f"step {optimizer_step}/{total_steps} | loss {window_loss:.4f} | "
                   f"lr {scheduler.get_last_lr()[0]:.2e} | grad_norm {grad_norm:.2f} | "
                   f"{tokens_per_sec:,.0f} tok/s | ETA {eta_h:.2f} h | peak mem {mem_gb:.1f} GB")
            logger.info(msg)
            append_jsonl(log_path, {
                "step": optimizer_step, "train_loss": window_loss,
                "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                "tokens_per_sec": tokens_per_sec, "eta_hours": eta_h,
                "peak_mem_gb": mem_gb, "tokens_seen": tokens_seen,
            })
            if wandb.run is not None:
                wandb.log({
                    "train/loss": window_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": float(grad_norm),
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/peak_mem_gb": mem_gb,
                }, step=optimizer_step)

        if args.eval_every and optimizer_step % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            best_val_loss = min(best_val_loss, val_loss)
            logger.info("step %d | val loss %.4f (best %.4f)", optimizer_step, val_loss, best_val_loss)
            append_jsonl(log_path, {"step": optimizer_step, "val_loss": val_loss})
            if wandb.run is not None:
                wandb.log({"val/loss": val_loss, "val/best_loss": best_val_loss}, step=optimizer_step)

        if args.save_every and optimizer_step % args.save_every == 0:
            ckpt_dir = args.output_dir / f"checkpoint-{optimizer_step}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info("Saved checkpoint to %s", ckpt_dir)

    for epoch in range(args.epochs):
        if stop_training:
            break
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids).logits
            # Labels are already the next token (packed dataset), so no extra
            # shift. fp32 cross-entropy for numerical stability over the 128k
            # vocab; divide by accumulation steps to average gradients.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), labels.reshape(-1)
            ) / args.gradient_accumulation_steps
            loss.backward()

            recent_losses.append(loss.item() * args.gradient_accumulation_steps)
            tokens_seen += labels.numel()
            microbatch_idx += 1

            if not math.isfinite(recent_losses[-1]):
                raise RuntimeError(
                    f"Non-finite loss at step {optimizer_step + 1}; halting to avoid wasting GPU time."
                )

            if microbatch_idx == args.gradient_accumulation_steps:
                run_optimizer_step()
                if args.max_steps and optimizer_step >= args.max_steps:
                    stop_training = True
                    break

        # Flush a trailing, incomplete accumulation window at epoch end.
        if microbatch_idx > 0 and not stop_training:
            run_optimizer_step()

    # --------------------------------------------------------------- save
    final_val_loss = evaluate(model, val_loader, device)
    train_time_h = (time.perf_counter() - start_time) / 3600
    logger.info("Final val loss: %.4f | train time: %.2f h | tokens: %d",
                final_val_loss, train_time_h, tokens_seen)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved fine-tuned model and tokenizer to %s", args.output_dir)

    summary = {
        **hyperparams,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "train_time_hours": train_time_h,
        "tokens_seen": tokens_seen,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_jsonl(log_path, {"step": optimizer_step, "val_loss": final_val_loss, "final": True})
    logger.info("Summary written to %s", summary_path)

    if wandb.run is not None:
        wandb.log({
            "final/val_loss": final_val_loss,
            "final/train_time_hours": train_time_h,
            "final/tokens_seen": tokens_seen,
        }, step=optimizer_step)
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
