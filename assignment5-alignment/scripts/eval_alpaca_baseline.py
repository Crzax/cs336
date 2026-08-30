"""Zero-shot AlpacaEval baseline for Llama 3.1 8B (safety/RLHF supplement, §3.3).

Loads the AlpacaEval instructions, generates zero-shot responses with vLLM, and
serializes them as a JSON array compatible with the `alpaca_eval` evaluator:
each entry has "instruction", "output", "generator", and "dataset".

Note: winrate judging (Llama 3.3 70B Instruct) is a separate step done via the
`alpaca_eval` CLI -- see docs/supplement_3.3_alpaca.md part (c).

Example (single H20):

```
uv run python scripts/eval_alpaca_baseline.py \
    --model /mnt/cephfs/user_crzaxchen/models/Meta-Llama-3.1-8B \
    --output-path scripts/results/alpaca_eval_baseline.json
```
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import time
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts_safety"

GENERATOR_NAME = "llama-3.1-8b-base"

STOP_STRINGS = ["# Query:", "\n```"]


def load_alpaca_eval(path: Path) -> list[dict]:
    """Read the AlpacaEval reference set; instructions + dataset ids come from it."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_prompts(examples: list[dict], prompt_style: str = "zero_shot") -> list[str]:
    task_template = (PROMPT_DIR / "alpaca_eval_zero_shot.prompt").read_text()
    system_template = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
    alpaca_template = (PROMPT_DIR / "alpaca_sft.prompt").read_text()
    prompts = []
    for e in examples:
        task_prompt = task_template.format(instruction=e["instruction"]).strip()
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
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap #examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generator-name", default=GENERATOR_NAME)
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
        default=REPO_ROOT / "scripts" / "results" / "alpaca_eval_baseline.json",
    )
    args = parser.parse_args()

    examples = load_alpaca_eval(REPO_ROOT / "data" / "alpaca_eval" / "alpaca_eval_gpt4_turbo.json")
    if args.limit is not None:
        random.Random(args.seed).shuffle(examples)
        examples = examples[: args.limit]
    prompts = build_prompts(examples, args.prompt_style)
    logger.info("Loaded %d AlpacaEval examples", len(examples))

    generations, elapsed = generate_offline(args.model, prompts, args.max_tokens, args.num_gpus)

    records = []
    for example, gen in zip(examples, generations):
        records.append(
            {
                "instruction": example["instruction"],
                "output": gen["text"],
                "generator": args.generator_name,
                "dataset": example["dataset"],
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n = len(records)
    lens = [g["num_tokens"] for g in generations]
    finish = Counter(g["finish_reason"] for g in generations)
    print(f"=== AlpacaEval zero-shot: {args.model} (n={n}) ===")
    print(f"generation wall time     : {elapsed:.1f}s")
    print(f"throughput               : {n / elapsed:.2f} examples/s, {sum(lens) / elapsed:.1f} tok/s")
    print(f"generated tokens         : mean={statistics.mean(lens):.1f} median={statistics.median(lens)} max={max(lens)}")
    print(f"finish reasons           : {dict(finish)}")
    print(f"dataset sources          : {dict(Counter(r['dataset'] for r in records))}")
    print(f"\nWrote JSON array to {args.output_path}")
    print(f"Next: judge winrate with the alpaca_eval CLI (see docs/supplement_3.3_alpaca.md (c)).")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
