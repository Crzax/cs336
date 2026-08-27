from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


VARIANTS = {
    "standard_grpo": ("Standard GRPO", ""),
    "grpo_constant": ("GRPO_constant", "grpo_constant"),
    "dr_grpo": ("Dr_GRPO", "dr_grpo"),
    "rft": ("RFT", "rft"),
    "maxrl": ("MaxRL", "maxrl"),
}

METRICS = {
    "loss": ("Loss", "loss"),
    "grad_norm": ("Gradient norm", "grad_norm"),
    "train_entropy": ("Train token entropy", "train_entropy"),
    "train_reward": ("Train reward", "train_reward"),
    "train_format_reward": ("Train format reward", "train_format_reward"),
    "val_reward": ("Validation reward", "val_reward"),
    "val_format_reward": ("Validation format reward", "val_format_reward"),
    "val_accuracy": ("Validation accuracy", "val_accuracy"),
    "val_avg_response_length": ("Validation average response length", "val_avg_response_length"),
}


def read_metrics(results_dir: Path, n_val_examples: int) -> dict[int, dict[int, dict[str, float]]]:
    runs: dict[int, dict[int, dict[str, float]]] = {}
    for path in sorted(results_dir.glob("seed_*/metrics.jsonl")):
        seed = int(path.parent.name.split("_")[-1])
        steps: dict[int, dict[str, float]] = {}
        with path.open(encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                step = int(record["step"])
                values: dict[str, float] = {}
                for key in METRICS:
                    if key in record:
                        value = float(record[key])
                        if key in {"val_reward", "val_format_reward"}:
                            value /= n_val_examples
                        values[key] = value
                steps[step] = values
        runs[seed] = steps
    return runs


def aggregate(runs: dict[int, dict[int, dict[str, float]]], key: str) -> dict[str, list[float]]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for steps in runs.values():
        for step, values in steps.items():
            if key in values:
                by_step[step].append(values[key])
    result = {"step": [], "mean": [], "std": [], "min": [], "max": []}
    for step in sorted(by_step):
        values = by_step[step]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result["step"].append(step)
        result["mean"].append(mean)
        result["std"].append(variance**0.5)
        result["min"].append(min(values))
        result["max"].append(max(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standard GRPO and on-policy variants")
    parser.add_argument("--results-root", type=Path, default=Path("scripts/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/grpo_variants"))
    parser.add_argument("--n-val-examples", type=int, default=1024)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=tuple(VARIANTS))
    args = parser.parse_args()

    all_runs = {}
    for variant in args.variants:
        label, relative_dir = VARIANTS[variant]
        results_dir = args.results_root / relative_dir if relative_dir else args.results_root
        runs = read_metrics(results_dir, args.n_val_examples)
        if set(runs) != {1, 2, 3, 4}:
            raise ValueError(f"{label}: expected seeds 1, 2, 3, 4; found {sorted(runs)}")
        all_runs[variant] = runs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, (title, filename) in METRICS.items():
        variant_values = {variant: aggregate(runs, key) for variant, runs in all_runs.items()}
        variant_values = {variant: values for variant, values in variant_values.items() if values["step"]}
        if not variant_values:
            continue

        plt.figure(figsize=(8, 5))
        metric_summary = {}
        for variant, values in variant_values.items():
            label, _ = VARIANTS[variant]
            steps = values["step"]
            means = values["mean"]
            stds = values["std"]
            plt.plot(steps, means, linewidth=2, label=label)
            plt.fill_between(steps, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.12)
            metric_summary[variant] = values
        plt.xlabel("Training step")
        plt.ylabel(title)
        plt.title(f"GRPO GSM8K variants: {title}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{filename}.png", dpi=180)
        plt.close()
        summary[key] = metric_summary

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
