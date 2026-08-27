import argparse
import json
import pathlib
from typing import Literal

import torch
import wandb

from cs336_alignment.checkpoint import (
    get_model_and_tokenizer,
    get_response_log_probs,
    grpo_train_step,
    tokenize_prompt_and_output,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer


MODEL_ID = "allenai/OLMo-2-0425-1B"
n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
inference_batch_size = 256
train_batch_size = 8
rollout_group_size = 8
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0

METHODS: dict[str, Literal["none", "noclip", "grpo", "gspo", "cispo"]] = {
    "offpolicy_naive": "none",
    "offpolicy_noclip": "noclip",
    "offpolicy_clip": "grpo",
    "offpolicy_gspo": "gspo",
    "offpolicy_cispo": "cispo",
}
CLIPRANGES: dict[str, float | None] = {
    "offpolicy_naive": None,
    "offpolicy_noclip": None,
    "offpolicy_clip": 0.2,
    "offpolicy_gspo": 3e-4,
    "offpolicy_cispo": 0.2,
}


def append_jsonl(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(server, policy, val_prompts, val_short_answers, step, record) -> None:
    server.sync_policy_weights(policy)
    completions = server.generate_completions(
        prompts=val_prompts,
        sampling_params={
            "temperature": sampling_temperature,
            "max_tokens": sampling_max_tokens,
            "n": 1,
            "seed": 5201314,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
    )
    responses = [completion.text for completion in completions]
    val_avg_response_length = sum(len(completion.token_ids) for completion in completions) / n_val_examples
    val_reward = 0
    val_format_reward = 0
    val_answer_reward = 0
    for response, ground_truth in zip(responses, val_short_answers):
        rewards = r1_zero_reward_fn(response, ground_truth)
        val_reward += int(rewards["reward"])
        val_format_reward += int(rewards["format_reward"])
        val_answer_reward += int(rewards["answer_reward"])

    record.update(
        {
            "val_reward": val_reward,
            "val_format_reward": val_format_reward,
            "val_answer_reward": val_answer_reward,
            "val_accuracy": val_answer_reward / n_val_examples,
            "val_avg_response_length": val_avg_response_length,
        }
    )
    print(
        f"step={step} val_accuracy={record['val_accuracy']:.4f} "
        f"val_format_reward={val_format_reward / n_val_examples:.4f} "
        f"val_avg_response_length={val_avg_response_length:.2f}"
    )


def save_rollout(
    phase: str,
    step: int,
    server,
    policy,
    prompts: list[str],
    questions: list[str],
    answers: list[str],
    seed: int,
    run_dir: pathlib.Path,
) -> list[dict]:
    server.sync_policy_weights(policy)
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": sampling_temperature,
            "max_tokens": sampling_max_tokens,
            "n": 1,
            "seed": 5201314,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
    )
    records = []
    for question, completion, ground_truth in zip(questions, completions, answers):
        rewards = r1_zero_reward_fn(completion.text, ground_truth)
        record = {
            "phase": phase,
            "step": step,
            "seed": seed,
            "question": question,
            "response": completion.text,
            "ground_truth": ground_truth,
            "reward": int(rewards["reward"]),
            "format_reward": int(rewards["format_reward"]),
            "answer_reward": int(rewards["answer_reward"]),
            "response_length": len(completion.token_ids),
        }
        records.append(record)
        append_jsonl(run_dir / "rollout.jsonl", record)
    return records


def init_wandb(seed: int, method: str) -> None:
    wandb.init(
        project="cs336-grpo-off-policy-gsm8k",
        group="grpo-off-policy",
        name=f"{method}-seed-{seed}",
        config={
            "method": method,
            "seed": seed,
            "model": MODEL_ID,
            "n_train_examples": n_train_examples,
            "n_val_examples": n_val_examples,
            "num_rollout_steps": num_rollout_steps,
            "learning_rate": learning_rate,
            "inference_batch_size": inference_batch_size,
            "train_batch_size": train_batch_size,
            "train_updates_per_rollout": inference_batch_size // train_batch_size,
            "group_size": rollout_group_size,
            "temperature": sampling_temperature,
            "max_tokens": sampling_max_tokens,
            "max_grad_norm": max_grad_norm,
            "cliprange": CLIPRANGES[method],
        },
    )


def log_rollouts(records: list[dict], step: int) -> None:
    table = wandb.Table(
        columns=[
            "step",
            "phase",
            "question",
            "response",
            "ground_truth",
            "reward",
            "format_reward",
            "answer_reward",
            "response_length",
        ]
    )
    for record in records:
        table.add_data(
            record["step"],
            record["phase"],
            record["question"],
            record["response"],
            record["ground_truth"],
            record["reward"],
            record["format_reward"],
            record["answer_reward"],
            record["response_length"],
        )
    wandb.log({"rollouts": table}, step=step)


def compute_old_log_probs(
    policy,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: str,
) -> torch.Tensor:
    tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        log_probs = get_response_log_probs(
            policy,
            tokenized["input_ids"].to(device),
            tokenized["labels"].to(device),
            return_token_entropy=False,
        )["log_probs"]
    if was_training:
        policy.train()
    return log_probs.detach().float().cpu()


def train_rollout_batch(
    policy,
    tokenizer,
    optimizer,
    method: str,
    train_prompts: list[str],
    train_responses: list[str],
    train_ground_truths: list[str],
    old_log_probs_by_chunk: list[torch.Tensor] | None,
    device: str,
) -> dict[str, float]:
    implementation_method = METHODS[method]
    method_cliprange = CLIPRANGES[method]
    losses = []
    grad_norms = []
    entropies = []
    rewards = []
    format_rewards = []
    clip_count = 0.0
    clip_total = 0.0
    pruned_sequences = 0

    for chunk_index, start in enumerate(range(0, inference_batch_size, train_batch_size)):
        end = start + train_batch_size
        old_log_probs = None
        if old_log_probs_by_chunk is not None:
            old_log_probs = old_log_probs_by_chunk[chunk_index]
        loss, metadata = grpo_train_step(
            model=policy,
            tokenizer=tokenizer,
            optimizer=optimizer,
            gradient_accumulation_steps=1,
            max_grad_norm=max_grad_norm,
            reward_fn=r1_zero_reward_fn,
            repeated_prompts=train_prompts[start:end],
            rollout_responses=train_responses[start:end],
            repeated_ground_truths=train_ground_truths[start:end],
            group_size=rollout_group_size,
            baseline="mean",
            advantage_eps=1e-6,
            advantage_normalizer="std",
            importance_reweighting_method=implementation_method,
            old_log_probs=old_log_probs,
            cliprange=method_cliprange,
            loss_normalization="sequence",
            normalization_constant=None,
        )
        losses.append(float(loss.detach().cpu()))
        grad_norms.append(float(metadata["grad_norm"]))
        entropies.append(float(metadata["mean_token_entropy"]))
        rewards.append(float(metadata["mean_reward"]))
        format_rewards.append(float(metadata["mean_format_reward"]))
        clip_count += float(metadata["clip_count"])
        clip_total += float(metadata["clip_total"])
        pruned_sequences += int(metadata["num_pruned_sequences"])

    return {
        "loss": sum(losses) / len(losses),
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "train_entropy": sum(entropies) / len(entropies),
        "train_reward": sum(rewards) / len(rewards),
        "train_format_reward": sum(format_rewards) / len(format_rewards),
        "clip_fraction": clip_count / clip_total if clip_total else 0.0,
        "num_pruned_sequences": pruned_sequences,
    }


def main(
    seed: int,
    method: str,
    train_device: str,
    vllm_device: int,
    port: int,
    nccl_master_port: int,
) -> None:
    if inference_batch_size % train_batch_size != 0:
        raise ValueError("inference_batch_size must be divisible by train_batch_size")
    if train_batch_size != rollout_group_size:
        raise ValueError("train_batch_size must equal group_size so each update keeps a full reward group")

    run_dir = pathlib.Path(__file__).parent / "results" / "off_policy" / method / f"seed_{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to append to existing run: {metrics_path}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_wandb(seed, method)
    policy, tokenizer = get_model_and_tokenizer(MODEL_ID, device=train_device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    server = VLLMServer(
        model_id=MODEL_ID,
        gpu=vllm_device,
        seed=seed,
        port=port,
    )
    server.start()
    try:
        # The script runs several seeds in one process. Use a distinct NCCL
        # bootstrap port per seed so the previous weight-sync group cannot collide.
        seed_nccl_master_port = nccl_master_port + seed - 1
        server.init_weight_sync(train_device, master_port=seed_nccl_master_port)
        data_dir = pathlib.Path(__file__).parent.parent / "data" / "gsm8k"
        with (data_dir / "train.jsonl").open() as f:
            data = [json.loads(line) for line in f][:n_train_examples]
        with (data_dir / "test.jsonl").open() as f:
            val_data = [json.loads(line) for line in f][:n_val_examples]

        questions = [item["question"] for item in data]
        short_answers = [item["answer"].split("####")[-1].strip() for item in data]
        val_questions = [item["question"] for item in val_data]
        val_short_answers = [item["answer"].split("####")[-1].strip() for item in val_data]
        template = (
            pathlib.Path(__file__).parent.parent
            / "cs336_alignment"
            / "prompts"
            / "r1_zero.prompt"
        ).read_text()
        prompts = [template.format(question=question) for question in questions]
        val_prompts = [template.format(question=question) for question in val_questions]

        run_dir.mkdir(parents=True, exist_ok=True)
        initial_records = save_rollout(
            "before", 0, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir
        )
        log_rollouts(initial_records, 0)

        num_prompts_per_batch = inference_batch_size // rollout_group_size
        for step in range(1, num_rollout_steps + 1):
            server.sync_policy_weights(policy)
            prompt_indices = torch.randint(0, len(prompts), (num_prompts_per_batch,))
            repeated_prompts = [prompts[i] for i in prompt_indices]
            repeated_ground_truths = [short_answers[i] for i in prompt_indices]
            train_prompts = [prompt for prompt in repeated_prompts for _ in range(rollout_group_size)]
            train_ground_truths = [
                answer for answer in repeated_ground_truths for _ in range(rollout_group_size)
            ]
            completions = server.generate_completions(
                prompts=repeated_prompts,
                sampling_params={
                    "temperature": sampling_temperature,
                    "max_tokens": sampling_max_tokens,
                    "n": rollout_group_size,
                    "seed": seed + step,
                    "stop": ["</answer>"],
                    "include_stop_str_in_output": True,
                },
            )
            rollout_responses = [completion.text for completion in completions]

            old_log_probs_by_chunk = None
            if METHODS[method] != "none":
                old_log_probs_by_chunk = []
                for start in range(0, inference_batch_size, train_batch_size):
                    old_log_probs_by_chunk.append(
                        compute_old_log_probs(
                            policy,
                            tokenizer,
                            train_prompts[start : start + train_batch_size],
                            rollout_responses[start : start + train_batch_size],
                            train_device,
                        )
                    )

            train_metrics = train_rollout_batch(
                policy,
                tokenizer,
                optimizer,
                method,
                train_prompts,
                rollout_responses,
                train_ground_truths,
                old_log_probs_by_chunk,
                train_device,
            )
            record = {
                "step": step,
                "seed": seed,
                "method": method,
                "train_updates": inference_batch_size // train_batch_size,
                **train_metrics,
            }
            if step % 4 == 0:
                evaluate(server, policy, val_prompts, val_short_answers, step, record)
            append_jsonl(metrics_path, record)
            wandb.log(
                {
                    "train/loss": record["loss"],
                    "train/grad_norm": record["grad_norm"],
                    "train/entropy": record["train_entropy"],
                    "train/reward": record["train_reward"],
                    "train/format_reward": record["train_format_reward"],
                    "train/clip_fraction": record["clip_fraction"],
                    "train/num_pruned_sequences": record["num_pruned_sequences"],
                    **(
                        {
                            "val/accuracy": record["val_accuracy"],
                            "val/format_reward": record["val_format_reward"] / n_val_examples,
                            "val/avg_response_length": record["val_avg_response_length"],
                        }
                        if "val_accuracy" in record
                        else {}
                    ),
                },
                step=step,
            )
            print(
                f"method={method} seed={seed} step={step} "
                f"reward={record['train_reward']:.4f} "
                f"clip_fraction={record['clip_fraction']:.4f}"
            )

        final_records = save_rollout(
            "after",
            num_rollout_steps + 1,
            server,
            policy,
            val_prompts[:4],
            val_questions[:4],
            val_short_answers[:4],
            seed,
            run_dir,
        )
        log_rollouts(final_records, num_rollout_steps + 1)
    finally:
        server.stop()
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run off-policy GRPO comparisons on GSM8K")
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--train-device", default="cuda:0")
    parser.add_argument("--vllm-device", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--nccl-master-port", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seeds:
        main(
            seed,
            args.method,
            args.train_device,
            args.vllm_device,
            args.port,
            args.nccl_master_port,
        )
