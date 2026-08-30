"""Zero-shot SimpleSafetyTests baseline for Llama 3.1 8B (safety supplement, §3.4).

Loads the SimpleSafetyTests prompts, generates zero-shot responses with vLLM,
and serializes them as JSONL compatible with scripts/evaluate_safety.py (which
judges safety with Llama 3.3 70B Instruct -- see part (c)).

Each serialized entry contains "prompts_final" and "output" (required by the
evaluator) plus id/harm_area/category for error analysis.

Example (single H20):

```
uv run python scripts/eval_sst_baseline.py \
    --model /mnt/cephfs/user_crzaxchen/models/Meta-Llama-3.1-8B \
    --output-path scripts/results/sst_baseline.jsonl
```
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts_safety"

STOP_STRINGS = ["# Query:", "\n```"]


def load_sst(path: Path) -> list[dict]:
    """Read the SimpleSafetyTests CSV."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "id": row["id"],
            "harm_area": row["harm_area"],
            "category": row["category"],
            "prompts_final": row["prompts_final"],
        }
        for row in rows
    ]


def build_prompts(examples: list[dict], prompt_style: str = "zero_shot") -> list[str]:
    task_template = (PROMPT_DIR / "simple_safety_tests_zero_shot.prompt").read_text()
    system_template = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
    alpaca_template = (PROMPT_DIR / "alpaca_sft.prompt").read_text()
    prompts = []
    for e in examples:
        task_prompt = task_template.format(instruction=e["prompts_final"]).strip()
        if prompt_style == "sft":
            # Match the instruction-tuning format exactly: the prompt is the
            # Alpaca template up to (and including) the "### Response:\n"
            # header, so generation starts where training responses began.
            prompts.append(
                alpaca_template.partition("{response}")[0].format(instruction=task_prompt)
            )
        else:
            prompts.append(system_template.format(instruction=task_prompt))
    return prompts


def generate_offline(model_path: str, prompts: list[str], max_tokens: int, num_gpus: int):
    """One-shot batched inference with the vLLM `LLM` API (no server)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop=STOP_STRINGS,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    if all(str(o.request_id).isdigit() for o in outputs):
        outputs = sorted(outputs, key=lambda o: int(o.request_id))
    generations = [
        {
            "text": o.outputs[0].text,
            "finish_reason": o.outputs[0].finish_reason,
            "num_tokens": len(o.outputs[0].token_ids),
        }
        for o in outputs
    ]
    return generations, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--prompt-style",
        default="zero_shot",
        choices=["zero_shot", "sft"],
        help="'sft' formats each instruction with the Alpaca template used for "
             "instruction tuning, instead of the zero-shot system prompt.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "scripts" / "results" / "sst_baseline.jsonl",
    )
    args = parser.parse_args()

    examples = load_sst(REPO_ROOT / "data" / "simple_safety_tests" / "simple_safety_tests.csv")
    prompts = build_prompts(examples, args.prompt_style)
    logger.info("Loaded %d SimpleSafetyTests examples", len(examples))

    generations, elapsed = generate_offline(args.model, prompts, args.max_tokens, args.num_gpus)

    records = []
    for example, gen in zip(examples, generations):
        records.append(
            {
                **example,
                "output": gen["text"],
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(records)
    lens = [g["num_tokens"] for g in generations]
    finish = Counter(g["finish_reason"] for g in generations)
    print(f"=== SimpleSafetyTests zero-shot: {args.model} (n={n}) ===")
    print(f"generation wall time     : {elapsed:.1f}s")
    print(f"throughput               : {n / elapsed:.2f} examples/s, {sum(lens) / elapsed:.1f} tok/s")
    print(f"generated tokens         : mean={statistics.mean(lens):.1f} median={statistics.median(lens)} max={max(lens)}")
    print(f"finish reasons           : {dict(finish)}")
    print(f"harm areas               : {dict(Counter(r['harm_area'] for r in records))}")
    print(f"\nWrote JSONL to {args.output_path}")
    print("Next: judge safety with scripts/evaluate_safety.py (see docs/supplement_3.md 3.4(c)).")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
