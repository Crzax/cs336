import argparse
import json
import pathlib
from typing import Literal, TypedDict

import torch
import wandb

from cs336_alignment.checkpoint import get_model_and_tokenizer, grpo_train_step
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.vllm_utils import VLLMServer

n_train_examples = 6400
n_val_examples = 1024
num_rollout_steps = 200
learning_rate = 1e-5
rollout_batch_size = train_batch_size = 256
group_size = 8
gradient_accumulation_steps = 32
sampling_temperature = 1.0
sampling_max_tokens = 512
max_grad_norm = 1.0

class VariantConfig(TypedDict):
    display_name: str
    baseline: Literal["mean", "none"]
    advantage_normalizer: Literal["std", "none", "mean"]
    loss_normalization: Literal["constant"]
    normalization_constant: int


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "grpo_constant": {
        "display_name": "GRPO_constant",
        "baseline": "mean",
        "advantage_normalizer": "std",
        "loss_normalization": "constant",
        "normalization_constant": 32,
    },
    "dr_grpo": {
        "display_name": "Dr_GRPO",
        "baseline": "mean",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": 32,
    },
    "rft": {
        "display_name": "RFT",
        "baseline": "none",
        "advantage_normalizer": "none",
        "loss_normalization": "constant",
        "normalization_constant": 32,
    },
    "maxrl": {
        "display_name": "MaxRL",
        "baseline": "mean",
        "advantage_normalizer": "mean",
        "loss_normalization": "constant",
        "normalization_constant": 32,
    },
}


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate(server, policy, val_prompts, val_short_answers, step, record):
    server.sync_policy_weights(policy)
    val_completions = server.generate_completions(
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
    val_responses = [completion.text for completion in val_completions]
    val_avg_resp_len = sum(len(completion.token_ids) for completion in val_completions) / n_val_examples
    val_reward, val_format_reward, val_answer_reward = 0, 0, 0
    for response, gt in zip(val_responses, val_short_answers):
        val_rewards = r1_zero_reward_fn(response, gt)
        val_reward += int(val_rewards["reward"])
        val_format_reward += int(val_rewards["format_reward"])
        val_answer_reward += int(val_rewards["answer_reward"])
    val_accuracy = val_answer_reward / n_val_examples
    print(
        f"Step {step}, val reward: {val_reward}, val format reward: {val_format_reward}, "
        f"val answer reward: {val_answer_reward}, val avg resp len: {val_avg_resp_len}"
    )
    record["val_reward"] = val_reward
    record["val_format_reward"] = val_format_reward
    record["val_answer_reward"] = val_answer_reward
    record["val_accuracy"] = val_accuracy
    record["val_avg_response_length"] = val_avg_resp_len


def save_rollout(phase, step, server, policy, prompts, questions, answers, seed, run_dir):
    server.sync_policy_weights(policy)
    rollout_path = run_dir / "rollout.jsonl"
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
        append_jsonl(rollout_path, record)
    return records


def init_wandb(seed, variant_config):
    wandb.init(
        project="cs336-grpo-gsm8k",
        group="grpo-on-policy-variants",
        name=f"{variant_config['display_name']}-seed-{seed}",
        config={
            "variant": variant_config["display_name"],
            "seed": seed,
            "model": "allenai/OLMo-2-0425-1B",
            "n_train_examples": n_train_examples,
            "n_val_examples": n_val_examples,
            "num_rollout_steps": num_rollout_steps,
            "learning_rate": learning_rate,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "temperature": sampling_temperature,
            "max_tokens": sampling_max_tokens,
            "max_grad_norm": max_grad_norm,
            "baseline": variant_config["baseline"],
            "advantage_normalizer": variant_config["advantage_normalizer"],
            "loss_normalization": variant_config["loss_normalization"],
            "normalization_constant": variant_config["normalization_constant"],
        },
    )


def wandb_log(sample_rollout_records, step):
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
    for record in sample_rollout_records:
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


def main(seed, variant, train_device="cuda:0", vllm_device=1, port=8001, nccl_master_port=None):
    variant_config = VARIANT_CONFIGS[variant]
    results_root = pathlib.Path(__file__).parent / "results"
    run_dir = results_root / variant / f"seed_{seed}"
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to append to existing run: {metrics_path}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    init_wandb(seed, variant_config)

    policy, tokenizer = get_model_and_tokenizer("allenai/OLMo-2-0425-1B", device=train_device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    num_prompts_per_batch = rollout_batch_size // group_size

    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=vllm_device, seed=seed, port=port)
    server.start()
    try:
        server.init_weight_sync(train_device, master_port=nccl_master_port)

        data_dir = pathlib.Path(__file__).parent.parent / "data/gsm8k"
        with (data_dir / "train.jsonl").open() as f:
            data = [json.loads(line) for line in f][:n_train_examples]
        with (data_dir / "test.jsonl").open() as f:
            val_data = [json.loads(line) for line in f][:n_val_examples]

        questions = [item["question"] for item in data]
        val_questions = [item["question"] for item in val_data]
        short_answers = [item["answer"].split("####")[-1].strip() for item in data]
        val_short_answers = [item["answer"].split("####")[-1].strip() for item in val_data]

        prompt_path = pathlib.Path(__file__).parent.parent / "cs336_alignment/prompts/r1_zero.prompt"
        template = prompt_path.read_text()
        prompts = [template.format(question=question) for question in questions]
        val_prompts = [template.format(question=question) for question in val_questions]

        sample_rollout_records = save_rollout(
            "before", 0, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir
        )
        wandb_log(sample_rollout_records, 0)

        for step in range(1, num_rollout_steps + 1):
            server.sync_policy_weights(policy)
            prompt_indices = torch.randint(0, len(prompts), (num_prompts_per_batch,))
            repeated_prompts = [prompts[i] for i in prompt_indices]
            repeated_ground_truths = [short_answers[i] for i in prompt_indices]
            train_prompts = [prompt for prompt in repeated_prompts for _ in range(group_size)]
            train_ground_truths = [answer for answer in repeated_ground_truths for _ in range(group_size)]

            completions = server.generate_completions(
                prompts=repeated_prompts,
                sampling_params={
                    "temperature": sampling_temperature,
                    "max_tokens": sampling_max_tokens,
                    "n": group_size,
                    "seed": seed + step,
                    "stop": ["</answer>"],
                    "include_stop_str_in_output": True,
                },
            )
            rollout_responses = [completion.text for completion in completions]

            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=train_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=train_ground_truths,
                group_size=group_size,
                baseline=variant_config["baseline"],
                advantage_eps=1e-6,
                advantage_normalizer=variant_config["advantage_normalizer"],
                importance_reweighting_method="none",
                old_log_probs=None,
                cliprange=None,
                loss_normalization=variant_config["loss_normalization"],
                normalization_constant=variant_config["normalization_constant"],
            )

            record = {
                "step": step,
                "seed": seed,
                "variant": variant_config["display_name"],
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(metadata["grad_norm"]),
                "train_entropy": float(metadata["mean_token_entropy"]),
                "train_reward": float(metadata["mean_reward"]),
                "train_format_reward": float(metadata["mean_format_reward"]),
                "num_pruned_sequences": int(metadata["num_pruned_sequences"]),
            }
            wandb.log(
                {
                    "train/loss": record["loss"],
                    "train/grad_norm": record["grad_norm"],
                    "train/entropy": record["train_entropy"],
                    "train/reward": record["train_reward"],
                    "train/format_reward": record["train_format_reward"],
                    "train/num_pruned_sequences": record["num_pruned_sequences"],
                },
                step=step,
            )

            if step % 4 == 0:
                evaluate(server, policy, val_prompts, val_short_answers, step, record)
                wandb.log(
                    {
                        "val/reward": float(record["val_reward"]) / n_val_examples,
                        "val/format_reward": float(record["val_format_reward"]) / n_val_examples,
                        "val/answer_reward": float(record["val_answer_reward"]) / n_val_examples,
                        "val/accuracy": record["val_accuracy"],
                        "val/avg_response_length": record["val_avg_response_length"],
                    },
                    step=step,
                )
            if step % 50 == 0:
                sample_rollout_records = save_rollout(
                    "running", step, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir
                )
                if step % 100 == 0:
                    wandb_log(sample_rollout_records, step)

            append_jsonl(metrics_path, record)

        sample_rollout_records = save_rollout(
            "after", num_rollout_steps + 1, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir
        )
        wandb_log(sample_rollout_records, num_rollout_steps + 1)
    finally:
        server.stop()
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="Run on-policy GRPO variants on GSM8K")
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANT_CONFIGS), default=tuple(VARIANT_CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4))
    parser.add_argument("--train-device", type=str, default="cuda:0",
                        help="CUDA device for the training model, e.g. 'cuda:0'.")
    parser.add_argument("--vllm-device", type=int, default=1,
                        help="Physical GPU index for the vLLM server.")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port for the vLLM server.")
    parser.add_argument("--nccl-master-port", type=int, default=None,
                        help="Fixed NCCL master port for weight sync. Required to run "
                             "multiple training processes in parallel on the same host; "
                             "each process must use a distinct port.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for variant in args.variants:
        for seed in args.seeds:
            main(seed, variant, args.train_device, args.vllm_device, args.port, args.nccl_master_port)
