"""Zero-shot GSM8K baseline for Llama 3.1 8B (safety/RLHF supplement, §3.2).

Loads the GSM8K test split, formats the zero-shot prompts, generates greedily
with vLLM, parses the final number from each generation, computes accuracy and
throughput, and serializes every example + generation + score to JSONL.

Note: this uses the *safety supplement* prompt pair (gsm8k_zero_shot.prompt
inside zero_shot_system_prompt.prompt) and the final-number parser, not the
r1_zero/boxed setup from the main RL assignment.

Two generation backends are available:
  --backend offline (default): `vllm.LLM` batched inference, no server.
  --backend server           : `cs336_alignment.vllm_utils.VLLMServer` HTTP API.

Example (single H20):

```
uv run python scripts/eval_gsm8k_baseline.py \
    --model /mnt/cephfs/user_crzaxchen/models/Meta-Llama-3.1-8B \
    --output-path scripts/results/gsm8k_baseline.jsonl
```
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from pathlib import Path

from cs336_alignment.metrics import gsm8k_is_correct, parse_gsm8k_response

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
GSM8K_DIR = REPO_ROOT / "data" / "gsm8k"
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts_safety"

STOP_STRINGS = ["# Query:", "\n```"]


def load_gsm8k(split: str) -> list[dict]:
    """Read GSM8K JSONL into examples with the gold answer split out."""
    examples = []
    with open(GSM8K_DIR / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            examples.append(
                {
                    "question": item["question"],
                    # Full chain-of-thought reference solution.
                    "reference_solution": item["answer"],
                    # Final answer after the "####" delimiter.
                    "answer": item["answer"].split("####")[-1].strip(),
                }
            )
    return examples


def build_prompts(examples: list[dict], prompt_style: str = "zero_shot") -> list[str]:
    task_template = (PROMPT_DIR / "gsm8k_zero_shot.prompt").read_text()
    system_template = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
    alpaca_template = (PROMPT_DIR / "alpaca_sft.prompt").read_text()
    prompts = []
    for e in examples:
        task_prompt = task_template.format(question=e["question"]).strip()
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


def generate_server(model_path: str, prompts: list[str], max_tokens: int, gpu: int, port: int):
    """Inference through the `VLLMServer` helper used by the RL scripts."""
    from cs336_alignment.vllm_utils import VLLMServer

    server = VLLMServer(model_id=model_path, gpu=gpu, seed=0, port=port)
    server.start()
    try:
        start = time.perf_counter()
        completions = server.generate_completions(
            prompts=prompts,
            sampling_params={
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "n": 1,
                "seed": 0,
                "top_p": 1.0,
                "stop": STOP_STRINGS,
                "include_stop_str_in_output": False,
            },
            batch_size=2048,
        )
        elapsed = time.perf_counter() - start
    finally:
        server.stop()
    generations = [
        {"text": c.text, "finish_reason": c.finish_reason, "num_tokens": len(c.token_ids)}
        for c in completions
    ]
    return generations, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap #examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="offline", choices=["offline", "server"])
    parser.add_argument(
        "--prompt-style",
        default="zero_shot",
        choices=["zero_shot", "sft"],
        help="'sft' wraps the GSM8K task prompt in the Alpaca template used for "
             "instruction tuning, instead of the zero-shot system prompt.",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for the 'server' backend.")
    parser.add_argument("--port", type=int, default=8001, help="Port for the 'server' backend.")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip generation and re-run scoring/error analysis on --output-path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT / "scripts" / "results" / "gsm8k_baseline.jsonl",
    )
    args = parser.parse_args()

    if args.analyze_only:
        with open(args.output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        elapsed = float("nan")
    else:
        examples = load_gsm8k(args.split)
        if args.limit is not None:
            random.Random(args.seed).shuffle(examples)
            examples = examples[: args.limit]
        prompts = build_prompts(examples, args.prompt_style)
        logger.info("Loaded %d GSM8K %s examples", len(examples), args.split)

        if args.backend == "offline":
            generations, elapsed = generate_offline(
                args.model, prompts, args.max_tokens, args.num_gpus
            )
        else:
            generations, elapsed = generate_server(
                args.model, prompts, args.max_tokens, args.gpu, args.port
            )

        records = []
        for example, prompt, gen in zip(examples, prompts, generations):
            prediction = parse_gsm8k_response(gen["text"])
            records.append(
                {
                    **example,
                    "prompt": prompt,
                    "generation": gen["text"],
                    "finish_reason": gen["finish_reason"],
                    "num_generated_tokens": gen["num_tokens"],
                    "prediction": prediction,
                    # Unparseable generations count as incorrect.
                    "correct": float(gsm8k_is_correct(prediction, example["answer"])),
                }
            )

        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(records)
    num_failed = sum(1 for r in records if r["prediction"] is None)
    num_correct = sum(int(r["correct"]) for r in records)
    total_tokens = sum(r["num_generated_tokens"] for r in records)

    print(f"=== GSM8K zero-shot baseline: {args.model} ({args.split}, n={n}) ===")
    print(f"accuracy                 : {num_correct / n:.4f} ({num_correct}/{n})")
    parsed = n - num_failed
    if parsed:
        parsed_correct = sum(int(r["correct"]) for r in records if r["prediction"] is not None)
        print(f"accuracy | parsed        : {parsed_correct / parsed:.4f} ({parsed_correct}/{parsed})")
    print(f"failed parses            : {num_failed} ({num_failed / n:.2%})")
    if not args.analyze_only:
        print(f"generation wall time     : {elapsed:.1f}s")
        print(f"throughput               : {n / elapsed:.2f} examples/s, {total_tokens / elapsed:.1f} tok/s")
    print(f"mean generated tokens    : {total_tokens / n:.1f}")
    print(f"finish reasons           : {dict(Counter(r['finish_reason'] for r in records))}")

    # Did the model keep answering the same question, or drift into new ones?
    truncated = [r for r in records if r["finish_reason"] == "length"]
    print(f"hit max_tokens           : {len(truncated)} ({len(truncated) / n:.2%})")
    if truncated:
        trunc_correct = sum(int(r["correct"]) for r in truncated)
        print(f"  of which correct       : {trunc_correct} ({trunc_correct / len(truncated):.2%})")

    # Does the prediction show up anywhere in the reference solution? A cheap
    # proxy for "right intermediate work, wrong final number".
    wrong = [r for r in records if not r["correct"]]
    if wrong:
        in_ref = sum(
            1
            for r in wrong
            if r["prediction"] is not None
            and re.search(rf"(?<![\d.]){re.escape(r['prediction'])}(?![\d.])", r["reference_solution"])
        )
        print(f"wrong preds appearing as an intermediate value in gold CoT: {in_ref}/{len(wrong)}")

    rng = random.Random(args.seed)
    failures = [r for r in records if r["prediction"] is None]
    if failures:
        print(f"\n--- up to 5 unparseable generations (of {len(failures)}) ---")
        for record in rng.sample(failures, min(5, len(failures))):
            print(f"  Q: {record['question'][:120]!r}")
            print(f"    gen: {record['generation'][:300]!r}")

    incorrect = [r for r in records if not r["correct"] and r["prediction"] is not None]
    if incorrect:
        print(f"\n--- 10 random incorrect (but parseable) generations (of {len(incorrect)}) ---")
        for record in rng.sample(incorrect, min(10, len(incorrect))):
            print(f"  Q: {record['question'][:200]!r}")
            print(f"    gold={record['answer']} pred={record['prediction']}")
            print(f"    gen: {record['generation'][:400]!r}")
            print(f"    ref: {record['reference_solution'][:200]!r}")

    print(f"\n{n} records at {args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
