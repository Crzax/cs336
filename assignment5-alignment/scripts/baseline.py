import json
import pathlib
from collections import Counter

from cs336_alignment.vllm_utils import VLLMServer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn


def safe_grade(fn, response, gt):
    try:
        return fn(response, gt)
    except Exception as e:
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0,
            "error": repr(e),
        }


def evaluate(server, prompts, reward_fn, sampling_params, questions, short_answers):
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=None,
    )
    records = []
    for question, prompt, gt, completion in zip(questions, prompts, short_answers, completions):
        grade = safe_grade(reward_fn, completion.text, gt)
        records.append(
            {
                "question": question,
                "prompt": prompt,
                "response": completion.text,
                "ground_truth": gt,
                "format_reward": grade["format_reward"],
                "answer_reward": grade["answer_reward"],
                "finish_reason": completion.finish_reason,
                "num_tokens": len(completion.token_ids),
                "error": grade.get("error"),
            }
        )
    return records


def summarize(records, name):
    counter = Counter((int(r["format_reward"]), int(r["answer_reward"])) for r in records)
    both = counter[(1, 1)]       # format=1, answer=1
    format_only = counter[(1, 0)]  # format=1, answer=0
    neither = counter[(0, 0)]      # format=0, answer=0
    n = len(records)
    print(f"--- {name} ({n} examples) ---")
    print(f"  (1) format=1 answer=1: {both}")
    print(f"  (2) format=1 answer=0: {format_only}")
    print(f"  (3) format=0 answer=0: {neither}")
    print(f"  accuracy={both / n:.4f}  format_rate={(both + format_only) / n:.4f}")
    return records


def main():
    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=0, seed=0, port=8001)
    server.start()

    path = pathlib.Path(__file__).parent.parent / "data/gsm8k/test.jsonl"
    prompt_dir = pathlib.Path(__file__).parent.parent / "cs336_alignment/prompts"

    with open(path, "r") as f:
        data = [json.loads(line) for line in f]

    questions = [item["question"] for item in data]
    short_answers = [item["answer"].split("####")[-1].strip() for item in data]

    prompt_specs = [
        (
            "question_only",
            prompt_dir / "question_only.prompt",
            question_only_reward_fn,
            {"temperature": 1.0, "max_tokens": 512, "n": 1, "seed": 0},
        ),
        (
            "r1_zero",
            prompt_dir / "r1_zero.prompt",
            r1_zero_reward_fn,
            {
                "temperature": 1.0,
                "max_tokens": 512,
                "n": 1,
                "seed": 0,
                "stop": ["</answer>"],
                "include_stop_str_in_output": True,
            },
        ),
        (
            "r1_zero_three_shot",
            prompt_dir / "r1_zero_three_shot_gsm8k.prompt",
            r1_zero_reward_fn,
            {
                "temperature": 1.0,
                "max_tokens": 512,
                "n": 1,
                "seed": 0,
                "stop": ["</answer>"],
                "include_stop_str_in_output": True,
            },
        ),
    ]

    all_records = {}
    for name, prompt_file, reward_fn, sampling_params in prompt_specs:
        template = prompt_file.read_text()
        prompts = [template.format(question=q) for q in questions]
        records = evaluate(
            server,
            prompts,
            reward_fn,
            sampling_params,
            questions,
            short_answers,
        )
        all_records[name] = records

        out_path = pathlib.Path(__file__).parent / f"{name}_results.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for name, records in all_records.items():
        summarize(records, name)

    server.stop()


if __name__ == "__main__":
    main()
