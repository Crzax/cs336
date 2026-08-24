import argparse
import wandb
import torch
from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.vllm_utils import VLLMServer
import json
import pathlib
from cs336_alignment.drgrpo_grader import question_only_reward_fn, r1_zero_reward_fn
from cs336_alignment.checkpoint import grpo_train_step

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

PROMPT_CONFIGS = {
    "question_only": ("question_only.prompt", question_only_reward_fn),
    "r1_zero": ("r1_zero.prompt", r1_zero_reward_fn),
    "r1_zero_three_shot": ("r1_zero_three_shot_gsm8k.prompt", r1_zero_reward_fn),
}


def make_sampling_params(prompt_name, n, seed):
    params = {
        "temperature": sampling_temperature,
        "max_tokens": sampling_max_tokens,
        "n": n,
        "seed": seed,
    }
    if prompt_name != "question_only":
        params.update({
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        })
    return params


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def evaluate(server, policy, val_prompts, val_short_answers, step, record, prompt_name, reward_fn):
    server.sync_policy_weights(policy)
    val_completions = server.generate_completions(
        prompts=val_prompts,
        sampling_params=make_sampling_params(prompt_name, 1, 5201314),
    )
    val_responses = [completion.text for completion in val_completions]
    val_avg_resp_len = sum([len(completion.token_ids) for completion in val_completions]) / n_val_examples
    val_reward, val_format_reward, val_answer_reward = 0, 0, 0
    for response, gt in zip(val_responses, val_short_answers):
        val_rewards = reward_fn(response, gt)
        val_reward += int(val_rewards["reward"])
        val_format_reward += int(val_rewards["format_reward"])
        val_answer_reward += int(val_rewards["answer_reward"])
    val_accuracy = val_answer_reward / n_val_examples
    print(f"Step {step}, val reward: {val_reward}, val format reward: {val_format_reward}, \
            val answer reward: {val_answer_reward},  \
            val avg resp len: {val_avg_resp_len}")
    record["val_reward"] = val_reward
    record["val_format_reward"] = val_format_reward
    record["val_answer_reward"] = val_answer_reward
    record["val_accuracy"] = val_accuracy
    record["val_avg_response_length"] = val_avg_resp_len

def save_rollout(phase, step, server, policy, prompts, ques, anss, seed, run_dir, prompt_name, reward_fn):
    server.sync_policy_weights(policy)
    rollout_path = run_dir / "rollout.jsonl"
    comps = server.generate_completions(
        prompts=prompts,
        sampling_params=make_sampling_params(prompt_name, 1, 5201314),
    )
    resps = [completion.text for completion in comps]
    records = []
    for que, resp, gt,comp in zip(ques, resps, anss, comps):
        rewards = reward_fn(resp, gt)
        reward = int(rewards["reward"])
        format_reward = int(rewards["format_reward"])
        answer_reward = int(rewards["answer_reward"])
        response_length = len(comp.token_ids)
        rollout_record = {
            "phase": phase,
            "step": step,
            "seed": seed,
            "question": que,
            "response": resp,
            "ground_truth": gt,
            "reward": reward,
            "format_reward": format_reward,
            "answer_reward": answer_reward,
            "response_length": response_length,
        }
        records.append(rollout_record)
        append_jsonl(rollout_path, rollout_record)

    return records

def init_wandb(prompt_name, learning_rate, seed):
    wandb.init(
        project="cs336-grpo-gsm8k",
        group="grpo-prompt-ablation",
        name=f"prompt_{prompt_name}_lr_{learning_rate}_seed_{seed}",
        config={
            "prompt": prompt_name,
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
        },
    )

def build_rollout_table(sample_rollout_records):
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
    return table

def main(prompt_name, seed, run_learning_rate):
    prompt_file, reward_fn = PROMPT_CONFIGS[prompt_name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    train_device = "cuda:0"
    vllm_device = 1
    run_dir = pathlib.Path(__file__).parent / "results" / f"prompt_{prompt_name}" / f"lr_{run_learning_rate:g}" / f"seed_{seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing run: {run_dir}. "
            "Remove the directory before rerunning this configuration."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    init_wandb(prompt_name, run_learning_rate, seed)

    policy, tokenizer = get_model_and_tokenizer("allenai/OLMo-2-0425-1B", device=train_device)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=run_learning_rate, betas=(0.9, 0.95), weight_decay=0.0
    )
    num_prompts_per_batch = rollout_batch_size // group_size

    server = VLLMServer(model_id="allenai/OLMo-2-0425-1B", gpu=vllm_device, seed=seed, port=8001)
    server.start()
    try:
        server.init_weight_sync(train_device)

        # load data
        test_path = pathlib.Path(__file__).parent.parent / "data/gsm8k/train.jsonl"
        val_path = pathlib.Path(__file__).parent.parent / "data/gsm8k/test.jsonl"
        with open(test_path, "r") as f:
            data = [json.loads(line) for line in f]
            data = data[:n_train_examples]
        with open(val_path, "r") as f:
            val_data = [json.loads(line) for line in f]
            val_data = val_data[:n_val_examples]

        questions = [item["question"] for item in data]
        val_questions = [item["question"] for item in val_data]
        short_answers = [item["answer"].split("####")[-1].strip() for item in data]
        val_short_answers = [item["answer"].split("####")[-1].strip() for item in val_data]
        
        prompt_dir = pathlib.Path(__file__).parent.parent / "cs336_alignment/prompts"
            
        template = (prompt_dir / prompt_file).read_text()
        prompts = [template.format(question=q) for q in questions]
        val_prompts = [template.format(question=q) for q in val_questions]

        sample_rollout_records = save_rollout(
            "before", 0, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir, prompt_name, reward_fn
        )
        wandb.log({"rollouts": build_rollout_table(sample_rollout_records)}, step=0)

        for step in range(1, num_rollout_steps + 1):
            server.sync_policy_weights(policy)

            # sample prompts
            prompt_indices = torch.randint(0, len(prompts), (num_prompts_per_batch,))
            repeated_prompts = [prompts[i] for i in prompt_indices]
            repeated_ground_truths = [short_answers[i] for i in prompt_indices]
            
            train_prompts = [prompt for prompt in repeated_prompts for _ in range(group_size)]
            train_ground_truths = [ground_truth for ground_truth in repeated_ground_truths for _ in range(group_size)]

            # run policy
            completions = server.generate_completions(
                prompts=repeated_prompts,
                sampling_params=make_sampling_params(prompt_name, group_size, seed + step),
            )
            rollout_responses = [completion.text for completion in completions]
            
            # run grpo train step
            loss, metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                reward_fn=reward_fn,
                repeated_prompts=train_prompts,
                rollout_responses=rollout_responses,
                repeated_ground_truths=train_ground_truths,
                group_size=group_size,
                baseline="mean",
                advantage_eps=1e-6,
                advantage_normalizer="std",
                importance_reweighting_method="none",
                old_log_probs=None,
                cliprange=None,
                loss_normalization="sequence",
                normalization_constant=None,
            )
            print(f"Step {step}, loss: {loss.item()}")
            for k, v in metadata.items():
                print(f"{k}: {v}",end=" ")
            
            record = {
                "step": step,
                "seed": seed,
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(metadata["grad_norm"]),
                "train_entropy": float(metadata["mean_token_entropy"]),
                "train_reward": float(metadata["mean_reward"]),
                "train_format_reward": float(metadata["mean_format_reward"]),
            }
            log_data = {
                "train/loss": record["loss"],
                "train/grad_norm": record["grad_norm"],
                "train/entropy": record["train_entropy"],
                "train/reward": record["train_reward"],
                "train/format_reward": record["train_format_reward"],
            }

            if step % 4 == 0:
                evaluate(server, policy, val_prompts, val_short_answers, step, record, prompt_name, reward_fn)
                log_data.update(
                    {
                        "val/reward": record["val_reward"] / n_val_examples,
                        "val/format_reward": record["val_format_reward"] / n_val_examples,
                        "val/answer_reward": record["val_answer_reward"] / n_val_examples,
                        "val/accuracy": record["val_accuracy"],
                        "val/avg_response_length": record["val_avg_response_length"],
                    }
                )

            if step % 50 == 0:
                sample_rollout_records = save_rollout(
                    "running", step, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir, prompt_name, reward_fn
                )
                if step % 100 == 0:
                    log_data["rollouts"] = build_rollout_table(sample_rollout_records)

            wandb.log(log_data, step=step)

            append_jsonl(metrics_path, record)
        sample_rollout_records = save_rollout(
            "after", num_rollout_steps + 1, server, policy, val_prompts[:4], val_questions[:4], val_short_answers[:4], seed, run_dir, prompt_name, reward_fn
        )
        wandb.log({"rollouts": build_rollout_table(sample_rollout_records)}, step=num_rollout_steps + 1)
    finally:
        server.stop()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        choices=tuple(PROMPT_CONFIGS),
        nargs="+",
        default=("question_only", "r1_zero_three_shot"),
        help="Prompt variants to train.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=learning_rate,
        help="Learning rate for every prompt variant.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(2, 3),
        help="Random seeds to run for each prompt variant.",
    )
    args = parser.parse_args()
    for prompt_name in args.prompts:
        for seed in args.seeds:
            main(prompt_name, seed, args.learning_rate)
