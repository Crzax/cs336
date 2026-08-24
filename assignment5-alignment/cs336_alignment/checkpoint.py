from cProfile import label
import re
import token
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from collections.abc import Callable
from typing import Literal
from torch.optim import Optimizer

def get_model_and_tokenizer(model_id_or_dir: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_dir,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" if device=='cpu' else "flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_dir)
    return model, tokenizer

def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
    assert len(prompt_strs) == len(output_strs)
    prompt_ids = [tokenizer.encode(s, add_special_tokens=False) for s in prompt_strs]
    output_ids = [tokenizer.encode(s, add_special_tokens=False) for s in output_strs]

    full_ids = [prompt_ids[i] + output_ids[i] for i in range(len(prompt_ids))]
    maxlen = max(len(ids) for ids in full_ids)
    full_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (maxlen - len(ids)) for ids in full_ids])
    full_mask = torch.tensor([[0] * len(prompt_ids[i]) + [1] * len(output_ids[i]) + [0] * (maxlen - len(prompt_ids[i]) - len(output_ids[i])) for i in range(len(prompt_ids))])
    input_ids = full_ids[:, :-1]
    labels = full_ids[:, 1:]
    response_mask = full_mask[:, 1:]
    return {
        "input_ids":input_ids,
        "labels":labels,
        "response_mask":response_mask,
    }

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids, labels=labels).logits
    log_probs_full = logits.log_softmax(dim=-1)
    per_token_log_probs = log_probs_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probs = torch.softmax(logits, dim=-1)
    token_entropy = - torch.sum(probs * log_probs_full, dim=-1)
    return {
        "log_probs": per_token_log_probs,
        "token_entropy": token_entropy if return_token_entropy else None,
    }

def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    rewards = []
    format_rewards = []
    answer_rewards = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        all_reward = reward_fn(response, gt)
        rewards.append(all_reward["reward"])
        format_rewards.append(all_reward["format_reward"])
        answer_rewards.append(all_reward["answer_reward"])
    reward = torch.tensor(rewards)
    format_reward = torch.tensor(format_rewards)
    answer_reward = torch.tensor(answer_rewards)
    return reward, {
        "mean_format_reward": torch.mean(format_reward), 
        "mean_answer_reward": torch.mean(answer_reward), 
        "mean_reward": torch.mean(reward)
    }

def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
):
    raw_rewards = raw_rewards.view(-1, group_size)
    normalized_rewards = raw_rewards
    if baseline == "mean":
        normalized_rewards = raw_rewards - raw_rewards.mean(dim=-1, keepdim=True)
    elif baseline == "none":
        normalized_rewards = raw_rewards
    else:
        raise ValueError(f"Invalid baseline: {baseline}")
    
    if advantage_normalizer == "std":
        normalized_rewards = normalized_rewards / (raw_rewards.std(dim=-1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "mean":
        normalized_rewards = normalized_rewards / (raw_rewards.mean(dim=-1, keepdim=True) + advantage_eps)
    elif advantage_normalizer == "none":
        normalized_rewards = normalized_rewards
    else:
        raise ValueError(f"Invalid advantage_normalizer: {advantage_normalizer}")

    return normalized_rewards.flatten(), {}

def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    advantages = raw_rewards_or_advantages.view(-1, 1)

    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        # J = A * log pi_theta; loss = -J (per token).
        per_token_policy_gradient_loss = -advantages * policy_log_probs
    else:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for importance reweighting")
        if importance_reweighting_method in ("grpo", "gspo") and cliprange is None:
            raise ValueError("cliprange is required for 'grpo' and 'gspo'")

        per_token_log_ratio = policy_log_probs - old_log_probs

        if importance_reweighting_method == "gspo":
            # Sequence-level weight s = exp(mean log-ratio over response tokens), shared across time.
            if response_mask is None:
                raise ValueError("response_mask is required for 'gspo'")
            mask = response_mask.to(per_token_log_ratio.dtype)
            seq_log_ratio = (per_token_log_ratio * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True)
            importance_weight = torch.exp(seq_log_ratio)
        else:
            # Token-level weight w = pi_theta / pi_0.
            importance_weight = torch.exp(per_token_log_ratio)

        if importance_reweighting_method == "noclip":
            per_token_policy_gradient_loss = -advantages * importance_weight
        else:
            # PPO/GRPO- or GSPO-style clipping:
            # J = min(A * w, A * clip(w, 1 - eps, 1 + eps)); loss = -J.
            clipped_weight = importance_weight.clamp(1 - cliprange, 1 + cliprange)
            objective = torch.minimum(advantages * importance_weight, advantages * clipped_weight)
            per_token_policy_gradient_loss = -objective
            # Fraction of positions where the unclipped path is active (i.e., weight got clipped).
            metadata["clip_mask"] = (advantages * importance_weight != objective).to(policy_log_probs.dtype)
            if importance_reweighting_method == "gspo":
                # Broadcast the sequence-level loss back over tokens for aggregation.
                per_token_policy_gradient_loss = per_token_policy_gradient_loss.expand_as(policy_log_probs)
                metadata["clip_mask"] = metadata["clip_mask"].expand_as(policy_log_probs)
    return per_token_policy_gradient_loss, metadata

def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    masked_loss = per_token_policy_gradient_loss * mask.to(
        dtype=per_token_policy_gradient_loss.dtype
    )

    if loss_normalization == "sequence":
        sequence_lengths = mask.sum(dim=-1).to(
            dtype=per_token_policy_gradient_loss.dtype
        )
        return (masked_loss.sum(dim=-1) / sequence_lengths).mean()

    if loss_normalization == "constant":
        if normalization_constant is None:
            raise ValueError(
                "normalization_constant is required for constant normalization"
            )
        return masked_loss.sum() / normalization_constant

    raise NotImplementedError(
        f"Unsupported loss normalization: {loss_normalization}"
    )
    
def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    optimizer: Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    # Reward normalization
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    # Importance reweighting and clipping
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    # Loss normalization
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    B = len(rollout_responses)
    G = gradient_accumulation_steps
    microbatch = B // G
    pro_and_outs= tokenize_prompt_and_output(repeated_prompts, rollout_responses, tokenizer)
    raw_rewards, rewards_details = compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)
    advantages, _ = compute_group_normalized_rewards(raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer)
    optimizer.zero_grad(set_to_none=True)
    loss = 0
    grad_norm = 0
    total_token_entropy = 0
    total_count = 0

    device = next(model.parameters()).device
    input_ids = pro_and_outs["input_ids"].to(device)
    labels = pro_and_outs["labels"].to(device)
    response_mask = pro_and_outs["response_mask"].to(device)
    advantages = advantages.to(device)

    for start in range(0, len(rollout_responses), microbatch):
        sl = slice(start, start + microbatch)
        log_probs, token_entropy = get_response_log_probs(model, input_ids[sl], labels[sl], True).values()
        per_token_loss, _ = compute_policy_gradient_loss(
            advantages[sl], 
                     log_probs, importance_reweighting_method, 
                        old_log_probs[sl] if old_log_probs is not None else None, 
                                      cliprange, response_mask[sl])
        loss_mb = aggregate_loss_across_microbatch(per_token_loss, 
                    response_mask[sl], loss_normalization, normalization_constant)
        loss_mb = loss_mb / G
        loss_mb.backward()
        loss += loss_mb.detach()
        total_token_entropy += (token_entropy * response_mask[sl]).sum().detach()
        total_count += response_mask[sl].sum().detach()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss, {
        "grad_norm": grad_norm,
        "mean_token_entropy": (total_token_entropy / total_count).item(),
        **rewards_details
    }