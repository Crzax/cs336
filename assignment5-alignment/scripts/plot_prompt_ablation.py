from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

N_VAL_EXAMPLES = 1024
PROMPT_LABELS = {
    "r1_zero": "r1_zero",
    "question_only": "question_only",
    "r1_zero_three_shot": "r1_zero_three_shot",
}


def read_run(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def run_path(results_dir: Path, prompt: str, seed: int, learning_rate: float) -> Path:
    if prompt == "r1_zero":
        return results_dir / f"seed_{seed}" / "metrics.jsonl"
    return results_dir / f"prompt_{prompt}" / f"lr_{learning_rate:g}" / f"seed_{seed}" / "metrics.jsonl"


def load_runs(results_dir: Path, prompts: list[str], seeds: list[int], learning_rate: float):
    runs: dict[str, dict[int, list[dict[str, float]]]] = {}
    for prompt in prompts:
        runs[prompt] = {}
        for seed in seeds:
            path = run_path(results_dir, prompt, seed, learning_rate)
            if not path.exists():
                raise FileNotFoundError(f"Missing result file: {path}")
            runs[prompt][seed] = read_run(path)
    return runs


def final_record(rows: list[dict[str, float]]) -> dict[str, float]:
    validation_rows = [row for row in rows if "val_reward" in row]
    if not validation_rows:
        raise ValueError("Run has no validation records")
    return validation_rows[-1]


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def plot_final_metrics(runs, output_dir: Path) -> dict:
    metrics = {
        "val_reward": ("Final validation reward", True),
        "val_format_reward": ("Final format reward", True),
        "val_accuracy": ("Final answer accuracy", False),
        "val_avg_response_length": ("Final average response length", False),
    }
    summary = {}
    labels = list(runs)
    positions = list(range(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (key, (title, normalize)) in zip(axes.flat, metrics.items()):
        means, stds = [], []
        for prompt in labels:
            values = [final_record(runs[prompt][seed])[key] for seed in runs[prompt]]
            if normalize:
                values = [value / N_VAL_EXAMPLES for value in values]
            mean, std = mean_std(values)
            means.append(mean)
            stds.append(std)
            summary.setdefault(prompt, {})[key] = {"mean": mean, "std": std, "values": values}
        axis.bar(positions, means, yerr=stds, capsize=4, color="#4c78a8")
        axis.set_title(title)
        axis.set_xticks(positions, [PROMPT_LABELS[p] for p in labels], rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("GRPO prompt ablation: final validation metrics")
    fig.tight_layout()
    fig.savefig(output_dir / "final_metrics.png", dpi=180)
    plt.close(fig)
    return summary


def aggregate_curve(rows_by_seed, key: str, normalize: bool):
    by_step: dict[int, list[float]] = {}
    for rows in rows_by_seed.values():
        for row in rows:
            if key not in row:
                continue
            value = float(row[key])
            if normalize:
                value /= N_VAL_EXAMPLES
            by_step.setdefault(int(row["step"]), []).append(value)
    steps = sorted(by_step)
    means = [sum(by_step[step]) / len(by_step[step]) for step in steps]
    lower = [min(by_step[step]) for step in steps]
    upper = [max(by_step[step]) for step in steps]
    return steps, means, lower, upper


def plot_training_curves(runs, output_dir: Path):
    metrics = {
        "val_accuracy": ("Validation accuracy", True),
        "val_format_reward": ("Validation format reward", True),
        "train_reward": ("Train reward", False),
        "train_entropy": ("Train token entropy", False),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (key, (title, normalize)) in zip(axes.flat, metrics.items()):
        for prompt, rows_by_seed in runs.items():
            steps, means, lower, upper = aggregate_curve(rows_by_seed, key, normalize)
            if not steps:
                continue
            axis.plot(steps, means, linewidth=2, label=PROMPT_LABELS[prompt])
            axis.fill_between(steps, lower, upper, alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("GRPO prompt ablation: training curves")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("scripts/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures/prompt_ablation"))
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--prompts", nargs="+", default=("r1_zero", "question_only", "r1_zero_three_shot"), choices=tuple(PROMPT_LABELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=(2, 3))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results_dir, args.prompts, args.seeds, args.learning_rate)
    summary = plot_final_metrics(runs, args.output_dir)
    plot_training_curves(runs, args.output_dir)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
