"""Evaluate an instruction-tuned model (SFT or DPO) on the supplement benchmarks.

Unlike the zero-shot baseline scripts (eval_*_baseline.py, which wrap tasks in
zero_shot_system_prompt.prompt for the base model), every benchmark here is
prompted in the Alpaca SFT format the model was trained on
(cs336_alignment/prompts_safety/alpaca_sft.prompt): the rendered task text is
the {instruction} and the model generates the {response}. This matches the
"*_sft" problems in supplement section 5 and is reused for the DPO
evaluations in 6.4 (the DPO model is trained on the same template).

Benchmarks:
  alpaca_eval : AlpacaEval 2.0 instructions -> JSON array for the
                `alpaca_eval` CLI (winrate judged separately with
                Llama 3.3 70B Instruct, as in the baseline chapter).
  sst         : SimpleSafetyTests prompts -> JSONL for
                scripts/evaluate_safety.py (safe proportion judged there).
  gsm8k       : GSM8K test split -> JSONL + accuracy
                (parser: cs336_alignment.metrics.parse_gsm8k_response).
  mmlu        : MMLU test split -> JSONL + accuracy
                (parser: cs336_alignment.metrics.parse_mmlu_response).

Task texts are formatted with the same zero-shot task templates as the
baselines (gsm8k_zero_shot.prompt / mmlu_zero_shot.prompt) before being
wrapped in the Alpaca template.

Example (single H20):

    uv run python scripts/eval_instruct.py \
        --model scripts/results/dpo_llama31_8b/best \
        --benchmark alpaca_eval \
        --generator-name llama-3.1-8b-sft-dpo \
        --output-path scripts/results/alpaca_eval_dpo.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import statistics
import time
from collections import Counter
from pathlib import Path

from cs336_alignment.metrics import (
    gsm8k_is_correct,
    parse_gsm8k_response,
    parse_mmlu_response,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts_safety"

DEFAULT_MAX_TOKENS = {"alpaca_eval": 1024, "sst": 1024, "gsm8k": 512, "mmlu": 512}


def alpaca_prompt_prefix() -> str:
    """The Alpaca SFT template up to (and including) the '### Response:' header.

    Identical to the prompt half used by cs336_alignment.dpo (and the packed
    SFT dataset), so evaluation prompts match training-time tokenization.
    """
    template = (PROMPT_DIR / "alpaca_sft.prompt").read_text()
    return template.partition("{response}")[0]


def build_prompts(benchmark: str, examples: list[dict]) -> list[str]:
    prefix_template = alpaca_prompt_prefix()
    task_template = None
    if benchmark in ("gsm8k", "mmlu"):
        task_template = (PROMPT_DIR / f"{benchmark}_zero_shot.prompt").read_text()
    prompts = []
    for e in examples:
        if benchmark == "gsm8k":
            instruction = task_template.format(question=e["question"]).strip()
        elif benchmark == "mmlu":
            instruction = task_template.format(
                subject=e["subject"], question=e["question"], options=e["options"]
            ).strip()
        else:  # alpaca_eval / sst: the instruction is used as-is
            instruction = e["instruction"] if benchmark == "alpaca_eval" else e["prompts_final"]
        prompts.append(prefix_template.format(instruction=instruction))
    return prompts


def load_examples(benchmark: str) -> list[dict]:
    if benchmark == "alpaca_eval":
        path = REPO_ROOT / "data" / "alpaca_eval" / "alpaca_eval_gpt4_turbo.json"
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if benchmark == "sst":
        path = REPO_ROOT / "data" / "simple_safety_tests" / "simple_safety_tests.csv"
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
    if benchmark == "gsm8k":
        path = REPO_ROOT / "data" / "gsm8k" / "test.jsonl"
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                examples.append(
                    {
                        "question": item["question"],
                        "reference_solution": item["answer"],
                        "answer": item["answer"].split("####")[-1].strip(),
                    }
                )
        return examples
    if benchmark == "mmlu":
        split_dir = REPO_ROOT / "data" / "mmlu" / "test"
        examples = []
        for csv_path in sorted(split_dir.glob("*.csv")):
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
    raise ValueError(f"unknown benchmark {benchmark!r}")


def generate_offline(model_path: str, prompts: list[str], max_tokens: int, num_gpus: int):
    """One-shot batched inference with the vLLM `LLM` API (no server).

    No stop strings: unlike the base model, the instruction-tuned models emit
    the EOS token they were trained on after each response.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=num_gpus, dtype="bfloat16")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="SFT or DPO checkpoint directory")
    parser.add_argument("--benchmark", required=True, choices=["alpaca_eval", "sst", "gsm8k", "mmlu"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="generation cap (default: per-benchmark, matching the baseline scripts)")
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap #examples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generator-name", default="llama-3.1-8b-instruct")
    parser.add_argument(
        "--output-path", type=Path, default=None,
        help="default: scripts/results/<benchmark>_instruct.json(l)",
    )
    args = parser.parse_args()

    if args.max_tokens is None:
        args.max_tokens = DEFAULT_MAX_TOKENS[args.benchmark]
    if args.output_path is None:
        suffix = "json" if args.benchmark == "alpaca_eval" else "jsonl"
        args.output_path = REPO_ROOT / "scripts" / "results" / f"{args.benchmark}_instruct.{suffix}"

    examples = load_examples(args.benchmark)
    if args.limit is not None:
        random.Random(args.seed).shuffle(examples)
        examples = examples[: args.limit]
    prompts = build_prompts(args.benchmark, examples)
    logger.info("Loaded %d %s examples", len(examples), args.benchmark)

    generations, elapsed = generate_offline(args.model, prompts, args.max_tokens, args.num_gpus)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(examples)
    lens = [g["num_tokens"] for g in generations]
    finish = Counter(g["finish_reason"] for g in generations)

    print(f"=== {args.benchmark} (alpaca prompt): {args.model} (n={n}) ===")
    print(f"generation wall time     : {elapsed:.1f}s")
    print(f"throughput               : {n / elapsed:.2f} examples/s, {sum(lens) / elapsed:.1f} tok/s")
    print(f"generated tokens         : mean={statistics.mean(lens):.1f} median={statistics.median(lens)} max={max(lens)}")
    print(f"finish reasons           : {dict(finish)}")

    if args.benchmark == "alpaca_eval":
        records = [
            {
                "instruction": example["instruction"],
                "output": gen["text"],
                "generator": args.generator_name,
                "dataset": example["dataset"],
            }
            for example, gen in zip(examples, generations)
        ]
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"dataset sources          : {dict(Counter(r['dataset'] for r in records))}")
        print(f"\nWrote JSON array to {args.output_path}")
        print("Next: judge winrate with the alpaca_eval CLI (see docs/supplement_3.md).")
        return

    if args.benchmark == "sst":
        records = [{**example, "output": gen["text"]} for example, gen in zip(examples, generations)]
        with open(args.output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"harm areas               : {dict(Counter(r['harm_area'] for r in records))}")
        print(f"\nWrote JSONL to {args.output_path}")
        print("Next: judge safety with scripts/evaluate_safety.py (see docs/supplement_3.md).")
        return

    # gsm8k / mmlu: parse and score.
    records = []
    for example, prompt, gen in zip(examples, prompts, generations):
        if args.benchmark == "gsm8k":
            prediction = parse_gsm8k_response(gen["text"])
            correct = float(gsm8k_is_correct(prediction, example["answer"]))
        else:
            prediction = parse_mmlu_response(example, gen["text"])
            correct = float(prediction is not None and prediction == example["answer"])
        records.append(
            {
                **example,
                "prompt": prompt,
                "generation": gen["text"],
                "finish_reason": gen["finish_reason"],
                "num_generated_tokens": gen["num_tokens"],
                "prediction": prediction,
                "correct": correct,
            }
        )
    with open(args.output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    num_correct = sum(int(r["correct"]) for r in records)
    num_failed = sum(1 for r in records if r["prediction"] is None)
    print(f"accuracy                 : {num_correct / n:.4f} ({num_correct}/{n})")
    if n - num_failed:
        parsed_correct = sum(int(r["correct"]) for r in records if r["prediction"] is not None)
        print(f"accuracy | parsed        : {parsed_correct / (n - num_failed):.4f} ({parsed_correct}/{n - num_failed})")
    print(f"failed parses            : {num_failed} ({num_failed / n:.2%})")
    print(f"\n{n} records at {args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(module)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
