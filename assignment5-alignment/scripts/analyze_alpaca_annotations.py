"""Error analysis for AlpacaEval annotations (safety/RLHF supplement, §3.3(d)).

Samples examples where the baseline response was dispreferred versus GPT-4
Turbo by the Llama 3.3 70B annotator, prints them side by side, and sanity
checks win/loss counts against the reported winrate.

Usage:
```
uv run python scripts/analyze_alpaca_annotations.py \
    --annotations scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json \
    --model-outputs scripts/results/alpaca_eval_baseline.json \
    --reference-outputs data/alpaca_eval/alpaca_eval_gpt4_turbo.json
```
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("scripts/alpaca_eval_vllm_llama3_3_70b_fn/annotations_seed0_configs.json"),
    )
    parser.add_argument("--model-outputs", type=Path, default=Path("scripts/results/alpaca_eval_baseline.json"))
    parser.add_argument(
        "--reference-outputs", type=Path, default=Path("data/alpaca_eval/alpaca_eval_gpt4_turbo.json")
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=500, help="Truncation for printed outputs.")
    args = parser.parse_args()

    with open(args.annotations, encoding="utf-8") as f:
        annotations = json.load(f)
    with open(args.model_outputs, encoding="utf-8") as f:
        model_outputs = json.load(f)
    with open(args.reference_outputs, encoding="utf-8") as f:
        reference_outputs = json.load(f)

    # instruction -> output maps; used to identify which side of each annotated
    # pair belongs to the baseline, regardless of the ordering convention.
    ours = {r["instruction"]: r["output"] for r in model_outputs}
    refs = {r["instruction"]: r["output"] for r in reference_outputs}

    results = []  # (annotation, baseline_output, reference_output, won: bool|None)
    skipped = 0
    unparsed = 0
    for ann in annotations:
        instruction = ann["instruction"]
        if instruction not in ours or instruction not in refs:
            skipped += 1
            continue
        ours_out, ref_out = ours[instruction], refs[instruction]
        # Which position does the baseline occupy in this annotation?
        # (alpaca_eval puts the evaluated model at output_1, reference at
        # output_2, but verify by string match.)
        baseline_is_1 = ann.get("output_1") == ours_out
        baseline_is_2 = ann.get("output_2") == ours_out
        if not (baseline_is_1 or baseline_is_2):
            baseline_is_1, baseline_is_2 = True, False
        # preference convention (alpaca_eval main.py): 1 = output_1 wins,
        # 1.5 = tie, 2 = output_2 wins, -1 = judge output failed to parse.
        preference = ann.get("preference")
        if preference is None or preference == -1:
            unparsed += 1
            continue
        if preference == 1.5:
            won = None  # tie
        elif baseline_is_1:
            won = preference == 1
        else:
            won = preference == 2
        results.append((ann, ours_out, ref_out, won))

    n = len(results)
    counts = Counter("tie" if won is None else ("win" if won else "loss") for _, _, _, won in results)
    print(f"=== AlpacaEval annotation analysis ({n} parsed, {unparsed} unparsed, {skipped} unmatched) ===")
    if unparsed:
        print(f"WARNING: {unparsed} annotations failed to parse (preference=-1) and were excluded.")
    print(f"baseline wins : {counts['win']} ({counts['win'] / n:.2%})")
    print(f"baseline loss : {counts['loss']} ({counts['loss'] / n:.2%})")
    print(f"ties          : {counts['tie']} ({counts['tie'] / n:.2%})")
    # Same definition as alpaca_eval's get_winrate: ties count as half a win.
    print(f"winrate ((wins + ties/2) / n) = {(counts['win'] + 0.5 * counts['tie']) / n:.4f}")

    losses = [(ann, ours_out, ref_out) for ann, ours_out, ref_out, won in results if won is False]
    rng = random.Random(args.seed)
    sample = rng.sample(losses, min(args.num_samples, len(losses)))
    print(f"\n--- {len(sample)} random dispreferred examples (of {len(losses)}) ---")
    for i, (ann, ours_out, ref_out) in enumerate(sample, 1):
        print(f"\n[{i}] instruction: {ann['instruction'][:200]!r}")
        print(f"    baseline ({len(ours_out)} chars): {ours_out[:args.max_chars]!r}")
        print(f"    gpt4-turbo ({len(ref_out)} chars): {ref_out[:args.max_chars]!r}")
        if ann.get("annotator"):
            print(f"    annotator: {str(ann['annotator'])[:200]!r}")


if __name__ == "__main__":
    main()
