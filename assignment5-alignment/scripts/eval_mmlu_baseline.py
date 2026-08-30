"""Zero-shot MMLU baseline for Llama 3.1 8B (safety/RLHF supplement, §3.1).

Loads the MMLU test split, formats the zero-shot prompts, generates greedily
with vLLM, parses predictions, computes accuracy and throughput, and serializes
every example + generation + score to JSONL.

Two generation backends are available:
  --backend offline (default): `vllm.LLM` batched inference, no server.
  --backend server           : `cs336_alignment.vllm_utils.VLLMServer` HTTP API,
                               matching the RL scripts in this repo.

Example (single H20):

```
uv run python scripts/eval_mmlu_baseline.py \
    --model /mnt/cephfs/user_crzaxchen/models/Meta-Llama-3.1-8B \
    --output-path scripts/results/mmlu_baseline.jsonl
```
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from cs336_alignment.metrics import parse_mmlu_response

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
MMLU_DIR = REPO_ROOT / "data" / "mmlu"
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts_safety"


def load_mmlu(split_dir: Path) -> list[dict]:
    """Read all `<subject>_<split>.csv` files into MMLU example dicts."""
    examples = []
    for csv_path in sorted(split_dir.glob("*.csv")):
        # e.g. "high_school_geography_test.csv" -> "high school geography"
        subject = csv_path.stem.rsplit("_", 1)[0].replace("_", " ")
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) != 6:
                    logger.warning("Skipping malformed row in %s: %r", csv_path, row)
                    continue
                question, *options, answer = row
                examples.append(
                    {
                        "subject": subject,
                        "question": question,
                        "options": options,
                        "answer": answer.strip(),
                    }
                )
    return examples


def build_prompts(examples: list[dict], prompt_style: str = "zero_shot") -> list[str]:
    task_template = (PROMPT_DIR / "mmlu_zero_shot.prompt").read_text()
    system_template = (PROMPT_DIR / "zero_shot_system_prompt.prompt").read_text()
    alpaca_template = (PROMPT_DIR / "alpaca_sft.prompt").read_text()
    prompts = []
    for example in examples:
        task_prompt = task_template.format(
            subject=example["subject"],
            question=example["question"],
            options=example["options"],
        )
        if prompt_style == "sft":
            # Match the instruction-tuning format exactly: the prompt is the
            # Alpaca template up to (and including) the "### Response:\n"
            # header, so generation starts where training responses began.
            prompt = alpaca_template.partition("{response}")[0].format(
                instruction=task_prompt.strip()
            )
        else:
            prompt = system_template.format(instruction=task_prompt.strip())
        prompts.append(prompt)
    return prompts


STOP_STRINGS = ["# Query:", "\n```"]


def generate_offline(model_path: str, prompts: list[str], max_tokens: int, num_gpus: int):
    """One-shot batched inference with the vLLM `LLM` API (no server)."""
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        # The system prompt makes the model close its markdown block and start a
        # new turn; stopping there keeps generations short without truncating
        # the answer sentence.
        stop=STOP_STRINGS,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.perf_counter() - start
    # vLLM returns outputs in input order, but sort by numeric request_id when
    # available as a defensive check.
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
        start: float = time.perf_counter()
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
        {
            "text": c.text,
            "finish_reason": c.finish_reason,
            "num_tokens": len(c.token_ids),
        }
        for c in completions
    ]
    return generations, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--split", default="test", choices=["dev", "val", "test"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap #examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        default="zero_shot",
        choices=["zero_shot", "sft"],
        help="'sft' wraps the MMLU task prompt in the Alpaca template used for "
             "instruction tuning, instead of the zero-shot system prompt.",
    )
    parser.add_argument(
        "--backend",
        default="offline",
        choices=["offline", "server"],
        help="'offline' uses vllm.LLM directly; 'server' reuses cs336_alignment.vllm_utils.VLLMServer.",
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
        default=REPO_ROOT / "scripts" / "results" / "mmlu_baseline.jsonl",
    )
    args = parser.parse_args()

    if args.analyze_only:
        with open(args.output_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        elapsed = float("nan")
    else:
        examples = load_mmlu(MMLU_DIR / args.split)
        if args.limit is not None:
            random.Random(args.seed).shuffle(examples)
            examples = examples[: args.limit]
        prompts = build_prompts(examples, args.prompt_style)
        logger.info("Loaded %d MMLU %s examples", len(examples), args.split)

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
            prediction = parse_mmlu_response(example, gen["text"])
            records.append(
                {
                    **example,
                    "prompt": prompt,
                    "generation": gen["text"],
                    "finish_reason": gen["finish_reason"],
                    "num_generated_tokens": gen["num_tokens"],
                    "prediction": prediction,
                    # Unparseable generations count as incorrect.
                    "correct": float(prediction is not None and prediction == example["answer"]),
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

    print(f"=== MMLU zero-shot baseline: {args.model} ({args.split}, n={n}) ===")
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
    print(f"predicted letter counts  : {dict(Counter(r['prediction'] for r in records))}")

    per_subject = defaultdict(list)
    for record in records:
        per_subject[record["subject"]].append(record["correct"])
    ranked = sorted(per_subject.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    print("\n--- worst 5 subjects ---")
    for subject, scores in ranked[:5]:
        print(f"  {subject:40s} {sum(scores) / len(scores):.3f} (n={len(scores)})")
    print("--- best 5 subjects ---")
    for subject, scores in ranked[-5:]:
        print(f"  {subject:40s} {sum(scores) / len(scores):.3f} (n={len(scores)})")

    rng = random.Random(args.seed)
    failures = [r for r in records if r["prediction"] is None]
    if failures:
        print(f"\n--- up to 5 unparseable generations (of {len(failures)}) ---")
        for record in rng.sample(failures, min(5, len(failures))):
            print(f"  [{record['subject']}] Q: {record['question'][:100]!r}")
            print(f"    gen: {record['generation'][:300]!r}")

    incorrect = [r for r in records if not r["correct"] and r["prediction"] is not None]
    if incorrect:
        print(f"\n--- 10 random incorrect (but parseable) generations (of {len(incorrect)}) ---")
        for record in rng.sample(incorrect, min(10, len(incorrect))):
            print(f"  [{record['subject']}] Q: {record['question'][:160]!r}")
            print(f"    options: {record['options']}")
            print(f"    gold={record['answer']} pred={record['prediction']}")
            print(f"    gen: {record['generation'][:200]!r}")

    print(f"\n{n} records at {args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
