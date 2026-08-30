"""Error analysis for SimpleSafetyTests judgments (safety supplement, §3.4(d)).

Reads the JSONL produced by scripts/evaluate_safety.py, counts safe vs unsafe
overall and per harm area, and samples examples judged unsafe.

Usage:
```
uv run python scripts/analyze_sst_annotations.py \
    --judged-path scripts/results/sst_baseline_judged.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judged-path", type=Path, default=Path("scripts/results/sst_baseline_judged.jsonl")
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=400)
    args = parser.parse_args()

    with open(args.judged_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    n = len(records)
    safe = sum(1 for r in records if r["metrics"]["safe"] == 1.0)
    unsafe = [r for r in records if r["metrics"]["safe"] == 0.0]
    print(f"=== SimpleSafetyTests judged: {args.judged_path} ===")
    print(f"safe   : {safe}/{n} ({safe / n:.2%})")
    print(f"unsafe : {len(unsafe)}/{n} ({len(unsafe) / n:.2%})")

    by_area = defaultdict(lambda: [0, 0])  # harm_area -> [safe, total]
    for r in records:
        by_area[r["harm_area"]][0] += int(r["metrics"]["safe"])
        by_area[r["harm_area"]][1] += 1
    print("\nper harm area:")
    for area, (s, t) in sorted(by_area.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {area:45s} safe {s}/{t} ({s / t:.0%})")

    # Judge verdict distribution (raw annotator response text)
    verdicts = Counter(str(r["model_response"]).strip().lower() for r in records)
    print(f"\njudge raw verdicts: {dict(verdicts)}")

    rng = random.Random(args.seed)
    sample = rng.sample(unsafe, min(args.num_samples, len(unsafe)))
    print(f"\n--- {len(sample)} random unsafe examples (of {len(unsafe)}) ---")
    for i, r in enumerate(sample, 1):
        print(f"\n[{i}] [{r['harm_area']} / {r['category']}]")
        print(f"    prompt: {r['prompts_final'][:150]!r}")
        print(f"    output ({len(r['output'])} chars): {r['output'][:args.max_chars]!r}")
        print(f"    judge said: {str(r['model_response'])[:80]!r}")


if __name__ == "__main__":
    main()
