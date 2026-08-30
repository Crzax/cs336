"""DPO training of the instruction-tuned Llama model on Anthropic HH (supplement 6.4).

Two-GPU setup per the handout: the policy pi_theta lives on cuda:0 and the
frozen reference pi_ref on cuda:1, both initialized from the SFT checkpoint.
Examples are NOT batched: every HH preference pair is scored on its own
through both models (cs336_alignment.dpo.compute_per_instance_dpo_components),
and gradient accumulation reaches an effective batch size of 64. RMSprop
(single state buffer per parameter) replaces AdamW to save GPU memory, as in
the original DPO work.

Validation: --val-size (default 200) held-out HH pairs. The tracked metric is
the "classification accuracy" of the implicit reward model, i.e. the fraction
of pairs whose chosen completion has the higher log-probability under the
current policy (the ref-subtracted implicit-reward accuracy is logged too).
The checkpoint with the best validation accuracy is saved to
<output-dir>/best; the final model is saved in <output-dir> itself.

Example (2 x H20 98GB):

    uv run python scripts/dpo_train.py \
        --model scripts/results/sft_llama31_8b \
        --hh-dir data/hh \
        --output-dir scripts/results/dpo_llama31_8b \
        2>&1 | tee logs/dpo_train.log

Memory budget (8B params, bf16): policy weights 16 GB + grads 16 GB + RMSprop
square_avg 16 GB = 48 GB on GPU 0 and reference weights 16 GB on GPU 1; with
per-instance sequences capped at --max-length (1024) tokens, activations are
only a few GB, so both GPUs stay well below 98 GB without gradient
checkpointing.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cs336_alignment.dpo import (
    _response_log_prob,
    _tokenize_prompt_and_response,
    compute_per_instance_dpo_components,
    load_hh_preferences,
)

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
def score_examples(
    model: torch.nn.Module,
    tokenizer,
    examples: list[dict],
    device: str,
) -> list[tuple[float, float]]:
    """(log pi(chosen|x), log pi(rejected|x)) for every example, no gradients."""
    was_training = model.training
    model.eval()
    results = []
    for ex in examples:
        prompt_ids, chosen_ids = _tokenize_prompt_and_response(
            ex["prompt"], ex["chosen"], tokenizer
        )
        _, rejected_ids = _tokenize_prompt_and_response(
            ex["prompt"], ex["rejected"], tokenizer
        )
        chosen_lp = _response_log_prob(model, prompt_ids, chosen_ids, device).item()
        rejected_lp = _response_log_prob(model, prompt_ids, rejected_ids, device).item()
        results.append((chosen_lp, rejected_lp))
    if was_training:
        model.train()
    return results


def validate(
    val_log_probs: list[tuple[float, float]],
    ref_log_probs: list[tuple[float, float]],
    beta: float,
) -> dict[str, float]:
    """Validation metrics from policy and reference log-probabilities.

    acc          : chosen has higher log-probability under the policy
                   (the handout's "classification accuracy").
    acc_implicit : the DPO implicit reward (log pi - log pi_ref) ranks the
                   chosen completion higher.
    loss/margins : mean DPO loss / beta-scaled reward margin over the set.
    """
    n = len(val_log_probs)
    acc = acc_implicit = 0
    losses: list[float] = []
    margins: list[float] = []
    for (pc, pr), (rc, rr) in zip(val_log_probs, ref_log_probs):
        acc += pc > pr
        acc_implicit += (pc - rc) > (pr - rr)
        margin = beta * ((pc - rc) - (pr - rr))
        margins.append(margin)
        # Numerically stable -log sigmoid(margin).
        losses.append(math.log1p(math.exp(-margin)) if margin >= 0 else -margin + math.log1p(math.exp(margin)))
    return {
        "acc": acc / n,
        "acc_implicit": acc_implicit / n,
        "loss": sum(losses) / n,
        "margin": sum(margins) / n,
        "reward_chosen": sum(pc - rc for (pc, _), (rc, _) in zip(val_log_probs, ref_log_probs)) / n,
        "reward_rejected": sum(pr - rr for (_, pr), (_, rr) in zip(val_log_probs, ref_log_probs)) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=REPO_ROOT / "scripts" / "results" / "sft_llama31_8b",
        help="instruction-tuned checkpoint used to initialize BOTH pi_theta and pi_ref",
    )
    parser.add_argument("--hh-dir", type=Path, default=REPO_ROOT / "data" / "hh")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "scripts" / "results" / "dpo_llama31_8b")
    # Model / data hyperparameters.
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--ref-device", default="cuda:1")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="drop pairs whose prompt+response exceeds this many tokens (either side)")
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=64,
                        help="examples per optimizer step (microbatch = 1 example)")
    # Optimizer hyperparameters.
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--rmsprop-alpha", type=float, default=0.99)
    parser.add_argument("--rmsprop-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Logging / evaluation / saving.
    parser.add_argument("--log-every", type=int, default=10, help="optimizer steps between train logs")
    parser.add_argument("--eval-every", type=int, default=50, help="optimizer steps between val evaluations (0 = only at end)")
    parser.add_argument("--no-save-best", action="store_true",
                        help="do not write the best-validation-accuracy checkpoint to <output-dir>/best")
    # Debug knobs.
    parser.add_argument("--max-train-examples", type=int, default=0, help="cap training examples (0 = all)")
    parser.add_argument("--max-steps", type=int, default=0, help="cap optimizer steps (0 = full epoch(s))")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging (console/JSONL still work)")
    parser.add_argument("--attn-implementation", default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="enable gradient checkpointing on the policy (off by default: "
                             "per-instance batches of <=1024 tokens need only a few GB)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    summary_path = args.output_dir / "summary.json"

    # ------------------------------------------------------------------ data
    logger.info("Loading HH preferences from %s", args.hh_dir)
    examples = load_hh_preferences(args.hh_dir)
    logger.info("Loaded %d single-turn HH preference pairs", len(examples))

    logger.info("Loading tokenizer from %s", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Drop pairs that do not fit --max-length tokens (either side), so that
    # per-instance forward/backward passes stay bounded in time and memory.
    kept: list[dict] = []
    n_dropped = 0
    for ex in examples:
        prompt_ids, chosen_ids = _tokenize_prompt_and_response(
            ex["prompt"], ex["chosen"], tokenizer
        )
        _, rejected_ids = _tokenize_prompt_and_response(
            ex["prompt"], ex["rejected"], tokenizer
        )
        if max(len(prompt_ids) + len(chosen_ids), len(prompt_ids) + len(rejected_ids)) <= args.max_length:
            kept.append(ex)
        else:
            n_dropped += 1
    logger.info("Length filter (max_length=%d): kept %d, dropped %d (%.1f%%)",
                args.max_length, len(kept), n_dropped, 100.0 * n_dropped / max(1, len(examples)))

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    val_examples = kept[: args.val_size]
    train_examples = kept[args.val_size:]
    if args.max_train_examples:
        train_examples = train_examples[: args.max_train_examples]
    logger.info("Split: %d train / %d validation pairs", len(train_examples), len(val_examples))

    # ----------------------------------------------------------------- model
    logger.info("Loading policy pi_theta on %s (bf16, %s)", args.policy_device, args.attn_implementation)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.policy_device)
    policy.config.use_cache = False
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    policy.train()

    logger.info("Loading frozen reference pi_ref on %s", args.ref_device)
    ref = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(args.ref_device)
    ref.config.use_cache = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------- optimizer
    microbatches_per_epoch = len(train_examples)
    steps_per_epoch = math.ceil(microbatches_per_epoch / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    optimizer = torch.optim.RMSprop(
        policy.parameters(),
        lr=args.learning_rate,
        alpha=args.rmsprop_alpha,
        eps=args.rmsprop_eps,
        weight_decay=args.weight_decay,
    )

    hyperparams = {
        "model": str(args.model),
        "hh_dir": str(args.hh_dir),
        "n_pairs_loaded": len(examples),
        "n_pairs_dropped_by_length": n_dropped,
        "max_length": args.max_length,
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_examples": args.gradient_accumulation_steps,
        "beta": args.beta,
        "optimizer": "RMSprop",
        "learning_rate": args.learning_rate,
        "rmsprop_alpha": args.rmsprop_alpha,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_schedule": "constant",
        "policy_device": args.policy_device,
        "ref_device": args.ref_device,
        "total_steps": total_steps,
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    logger.info("Hyperparameters: %s", json.dumps(hyperparams, indent=2))

    if not args.no_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed (it is in the `gpu` extra); use --no-wandb or `uv sync --extra gpu`")
        wandb.init(
            project="cs336-a5-supplement-dpo",
            name=args.output_dir.name,
            config=hyperparams,
        )

    # Reference log-probs on the validation set never change: compute them once.
    logger.info("Precomputing reference log-probs for %d validation pairs", len(val_examples))
    ref_val_log_probs = score_examples(ref, tokenizer, val_examples, args.ref_device)

    # ------------------------------------------------------------ train loop
    start_time = time.perf_counter()
    optimizer_step = 0
    microbatch_idx = 0            # examples since last optimizer.step()
    window: list[dict] = []       # per-example stats of the current accumulation window
    examples_seen = 0
    best_val_acc = -1.0
    best_step = None
    stop_training = False

    def evaluate_and_maybe_save() -> dict[str, float]:
        """Validation pass; saves <output-dir>/best when accuracy improves."""
        nonlocal best_val_acc, best_step
        val_lp = score_examples(policy, tokenizer, val_examples, args.policy_device)
        metrics = validate(val_lp, ref_val_log_probs, args.beta)
        improved = metrics["acc"] > best_val_acc
        if improved:
            best_val_acc = metrics["acc"]
            best_step = optimizer_step
            if not args.no_save_best:
                best_dir = args.output_dir / "best"
                policy.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                logger.info("Saved new best checkpoint (val acc %.4f) to %s", best_val_acc, best_dir)
        logger.info(
            "step %d | val acc %.4f (best %.4f @ step %s) | val acc_implicit %.4f | val loss %.4f | val margin %.4f",
            optimizer_step, metrics["acc"], best_val_acc, best_step,
            metrics["acc_implicit"], metrics["loss"], metrics["margin"],
        )
        append_jsonl(log_path, {
            "step": optimizer_step,
            "val_acc": metrics["acc"],
            "val_acc_implicit": metrics["acc_implicit"],
            "val_loss": metrics["loss"],
            "val_margin": metrics["margin"],
            "val_reward_chosen": metrics["reward_chosen"],
            "val_reward_rejected": metrics["reward_rejected"],
            "best_val_acc": best_val_acc,
            "saved_best": improved and not args.no_save_best,
        })
        if wandb.run is not None:
            wandb.log({
                "val/acc": metrics["acc"],
                "val/acc_implicit": metrics["acc_implicit"],
                "val/loss": metrics["loss"],
                "val/margin": metrics["margin"],
                "val/reward_chosen": metrics["reward_chosen"],
                "val/reward_rejected": metrics["reward_rejected"],
                "val/best_acc": best_val_acc,
            }, step=optimizer_step)
        return metrics

    # Baseline evaluation of the (unchanged) policy before the first update.
    if args.eval_every:
        evaluate_and_maybe_save()

    def run_optimizer_step() -> None:
        nonlocal optimizer_step, microbatch_idx, window
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

        def mean(key: str) -> float:
            return sum(w[key] for w in window) / len(window)

        window_stats = {key: mean(key) for key in window[0]} if window else {}
        microbatch_idx = 0
        window = []

        elapsed = time.perf_counter() - start_time
        examples_per_sec = examples_seen / elapsed if elapsed > 0 else 0.0
        remaining_examples = (total_steps - optimizer_step) * args.gradient_accumulation_steps
        eta_h = remaining_examples / examples_per_sec / 3600 if examples_per_sec > 0 else float("nan")
        mem_gb = torch.cuda.max_memory_allocated(torch.device(args.policy_device)) / 2**30

        if optimizer_step % args.log_every == 0 or optimizer_step == total_steps:
            msg = (f"step {optimizer_step}/{total_steps} | loss {window_stats.get('loss', float('nan')):.4f} | "
                   f"margin {window_stats.get('margin', float('nan')):.4f} | "
                   f"train acc {window_stats.get('acc', float('nan')):.3f} | "
                   f"lr {optimizer.param_groups[0]['lr']:.2e} | grad_norm {grad_norm:.2f} | "
                   f"{examples_per_sec:.2f} ex/s | ETA {eta_h:.2f} h | peak mem {mem_gb:.1f} GB")
            logger.info(msg)
            append_jsonl(log_path, {
                "step": optimizer_step,
                "train_loss": window_stats.get("loss"),
                "train_margin": window_stats.get("margin"),
                "train_acc": window_stats.get("acc"),
                "train_reward_chosen": window_stats.get("reward_chosen"),
                "train_reward_rejected": window_stats.get("reward_rejected"),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
                "examples_per_sec": examples_per_sec,
                "eta_hours": eta_h,
                "peak_mem_gb": mem_gb,
                "examples_seen": examples_seen,
            })
            if wandb.run is not None:
                wandb.log({
                    "train/loss": window_stats.get("loss"),
                    "train/margin": window_stats.get("margin"),
                    "train/acc": window_stats.get("acc"),
                    "train/reward_chosen": window_stats.get("reward_chosen"),
                    "train/reward_rejected": window_stats.get("reward_rejected"),
                    "train/grad_norm": float(grad_norm),
                    "train/examples_per_sec": examples_per_sec,
                    "train/peak_mem_gb": mem_gb,
                }, step=optimizer_step)

        if args.eval_every and optimizer_step % args.eval_every == 0:
            evaluate_and_maybe_save()

    for _epoch in range(args.epochs):
        if stop_training:
            break
        for ex in train_examples:
            comps = compute_per_instance_dpo_components(
                policy, ref, tokenizer, args.beta,
                ex["prompt"], ex["chosen"], ex["rejected"],
            )
            loss = comps["loss"] / args.gradient_accumulation_steps
            loss.backward()

            pc = comps["policy_chosen"].item()
            pr = comps["policy_rejected"].item()
            rc = comps["ref_chosen"].item()
            rr = comps["ref_rejected"].item()
            window.append({
                "loss": loss.item() * args.gradient_accumulation_steps,
                "margin": comps["margins"].item(),
                "acc": float(pc > pr),
                "reward_chosen": pc - rc,
                "reward_rejected": pr - rr,
            })
            examples_seen += 1
            microbatch_idx += 1

            if not math.isfinite(window[-1]["loss"]):
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
    final_metrics = evaluate_and_maybe_save()
    train_time_h = (time.perf_counter() - start_time) / 3600
    logger.info("Final val acc %.4f (best %.4f @ step %s) | train time %.2f h | examples %d",
                final_metrics["acc"], best_val_acc, best_step, train_time_h, examples_seen)

    policy.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved final model and tokenizer to %s", args.output_dir)

    summary = {
        **hyperparams,
        "final_val_acc": final_metrics["acc"],
        "final_val_acc_implicit": final_metrics["acc_implicit"],
        "best_val_acc": best_val_acc,
        "best_step": best_step,
        "train_time_hours": train_time_h,
        "examples_seen": examples_seen,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_jsonl(log_path, {"step": optimizer_step, "final": True, "val_acc": final_metrics["acc"]})
    logger.info("Summary written to %s", summary_path)

    if wandb.run is not None:
        wandb.log({
            "final/val_acc": final_metrics["acc"],
            "final/best_val_acc": best_val_acc,
            "final/train_time_hours": train_time_h,
        }, step=optimizer_step)
        wandb.summary.update(summary)
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
