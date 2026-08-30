"""Plot DPO training curves from a run's train_log.jsonl (supplement 6.4).

Left panel : validation accuracy vs optimizer step, both the policy-logprob
             accuracy (the handout's "classification accuracy" metric used for
             best-checkpoint selection) and the implicit-reward accuracy.
Right panel: train DPO loss and reward margin (beta-scaled) vs step.

Example:

    uv run --extra plots python scripts/plot_dpo_curve.py \
        --run scripts/results/dpo_llama31_8b \
        --output docs/figures/dpo_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_log(run_dir: Path) -> list[dict]:
    records = []
    with (run_dir / "train_log.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=REPO_ROOT / "scripts" / "results" / "dpo_llama31_8b")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs" / "figures" / "dpo_curves.png")
    args = parser.parse_args()

    records = load_log(args.run)
    val = [r for r in records if "val_acc" in r and not r.get("final")]
    train = [r for r in records if "train_loss" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    steps = [r["step"] for r in val]
    ax1.plot(steps, [r["val_acc"] for r in val], "o-", label="val acc (policy log-prob)")
    if any("val_acc_implicit" in r for r in val):
        ax1.plot(steps, [r["val_acc_implicit"] for r in val], "s--", label="val acc (implicit reward)")
    best = max(val, key=lambda r: r["val_acc"]) if val else None
    if best:
        ax1.axvline(best["step"], color="gray", linestyle=":", alpha=0.7)
        ax1.annotate(
            f"best {best['val_acc']:.3f} @ step {best['step']}",
            xy=(best["step"], best["val_acc"]),
            xytext=(6, -12), textcoords="offset points", fontsize=9,
        )
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("validation accuracy")
    ax1.set_title("DPO validation accuracy")
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)
    ax1.legend()

    tsteps = [r["step"] for r in train]
    ax2.plot(tsteps, [r["train_loss"] for r in train], label="train DPO loss")
    ax2.plot(tsteps, [r["train_margin"] for r in train], label="train margin (β-scaled)")
    ax2.set_xlabel("optimizer step")
    ax2.set_ylabel("value")
    ax2.set_title("DPO training loss / margin")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle(args.run.name, y=1.02)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")
    if best:
        print(f"best val acc {best['val_acc']:.4f} at step {best['step']}")


if __name__ == "__main__":
    main()
