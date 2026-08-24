from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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
    result = {"step": [], "mean": [], "min": [], "max": [], "std": []}
    for step in sorted(by_step):
        values = by_step[step]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        result["step"].append(step)
        result["mean"].append(mean)
        result["min"].append(min(values))
        result["max"].append(max(values))
        result["std"].append(variance**0.5)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("scripts/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/grpo"))
    parser.add_argument("--n-val-examples", type=int, default=1024)
    args = parser.parse_args()

    runs = read_metrics(args.results_dir, args.n_val_examples)
    if set(runs) != {1, 2, 3, 4}:
        raise ValueError(f"Expected seeds 1, 2, 3, 4; found {sorted(runs)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, list[float]]] = {}
    for key, (title, filename) in METRICS.items():
        values = aggregate(runs, key)
        if not values["step"]:
            continue
        summary[key] = values
        steps = values["step"]
        mean = values["mean"]
        lower = values["min"]
        upper = values["max"]
        plt.figure(figsize=(7, 4.5))
        plt.plot(steps, mean, linewidth=2, label="mean")
        plt.fill_between(steps, lower, upper, alpha=0.2, label="min–max")
        plt.xlabel("Training step")
        plt.ylabel(title)
        plt.title(f"GRPO GSM8K: {title}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / f"{filename}.png", dpi=180)
        plt.close()

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
