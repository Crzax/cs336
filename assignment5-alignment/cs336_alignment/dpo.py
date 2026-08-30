"""DPO utilities (supplement 6): HH preference data loading and DPO loss.

The Anthropic HH preference files store each comparison as a JSONL row with
"chosen" and "rejected" conversation strings, where a conversation is
"\\n\\nHuman: ..." / "\\n\\nAssistant: ..." turns concatenated into one string.
The chosen and rejected conversations share the same prompt (and possibly
several turns) before diverging at the first preference-relevant reply.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

HH_FILES = (
    "harmless-base",
    "helpful-base",
    "helpful-online",
    "helpful-rejection-sampled",
)

_ALPACA_SFT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts_safety" / "alpaca_sft.prompt"
)

# Capturing-group split keeps the role markers so turns can be paired with
# their messages.
_TURN_PATTERN = re.compile(r"\n\n(Human|Assistant): ")


def _parse_conversation(text: str) -> tuple[list[str], list[str]]:
    """Split an HH conversation string into (human_messages, assistant_messages)."""
    text = text.strip()
    # The turn pattern requires a "\n\n" prefix, so re-attach one when the
    # conversation starts directly with "Human: " after stripping.
    if text.startswith("Human: "):
        text = "\n\n" + text
    parts = _TURN_PATTERN.split(text)
    # parts == ['', 'Human', ' msg', 'Assistant', ' reply', ...]
    humans: list[str] = []
    assistants: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        role, message = parts[i], parts[i + 1]
        if role == "Human":
            humans.append(message.strip())
        else:
            assistants.append(message.strip())
    return humans, assistants


def _to_single_turn_example(
    chosen_text: str, rejected_text: str, source: str
) -> dict | None:
    """Convert one HH comparison to {prompt, chosen, rejected, source}.

    Returns None for anything that is not a clean single-turn comparison:
    multi-turn conversations (the human sent more than one message), pairs
    whose prompts diverge, or empty assistant replies.
    """
    chosen_humans, chosen_assistants = _parse_conversation(chosen_text)
    rejected_humans, rejected_assistants = _parse_conversation(rejected_text)
    if len(chosen_humans) != 1 or len(rejected_humans) != 1:
        return None
    if len(chosen_assistants) != 1 or len(rejected_assistants) != 1:
        return None
    if chosen_humans[0] != rejected_humans[0]:
        return None
    if not chosen_assistants[0] or not rejected_assistants[0]:
        return None
    return {
        "prompt": chosen_humans[0],
        "chosen": chosen_assistants[0],
        "rejected": rejected_assistants[0],
        "source": source,
    }


def load_hh_preferences(hh_dir: str | Path) -> list[dict]:
    """Load the four Anthropic HH preference files into one list for DPO training.

    Each returned example is a dict with keys:
        prompt:   the single human message (instruction),
        chosen:   the preferred assistant reply,
        rejected: the dispreferred assistant reply,
        source:   which file it came from (e.g. "helpful-base").

    Only single-turn conversations are kept (see `_to_single_turn_example`).
    """
    examples: list[dict] = []
    for name in HH_FILES:
        path = Path(hh_dir) / f"{name}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                example = _to_single_turn_example(
                    record["chosen"], record["rejected"], name
                )
                if example is not None:
                    examples.append(example)
    return examples


def _tokenize_prompt_and_response(
    prompt: str, response: str, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[int], list[int]]:
    """Tokenize one (prompt, response) pair with the Alpaca SFT template.

    The prompt half is the template prefix up to (and including) the
    "### Response:" header; the response half is the bare response plus the
    EOS token. The prompt keeps the tokenizer's default special-token
    behavior (BOS for Llama), matching how PackedSFTDataset tokenizes; the
    response must not add another BOS, hence add_special_tokens=False.
    """
    template = _ALPACA_SFT_PROMPT_PATH.read_text()
    prompt_template = template.partition("{response}")[0]
    prompt_str = prompt_template.format(instruction=prompt)
    prompt_ids = tokenizer(prompt_str).input_ids
    response_ids = tokenizer(response, add_special_tokens=False).input_ids
    return prompt_ids, response_ids + [tokenizer.eos_token_id]


def _response_log_prob(
    model: torch.nn.Module,
    prompt_ids: list[int],
    response_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Sum of conditional log-probs of the response tokens (incl. EOS) given the prompt."""
    input_ids = torch.tensor(
        [prompt_ids + response_ids], dtype=torch.long, device=device
    )
    logits = model(input_ids).logits  # (1, T, V)
    log_probs = logits.log_softmax(dim=-1)
    # Position t-1 predicts token t: score every token except the first.
    targets = input_ids[:, 1:]
    token_log_probs = (
        log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    )  # (1, T-1)
    # The last len(response_ids) entries correspond to the response tokens.
    return token_log_probs[:, -len(response_ids) :].sum()


def compute_per_instance_dpo_components(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> dict[str, torch.Tensor]:
    """Per-instance DPO loss plus its scalar components (for training loops).

    Returns a dict of scalar tensors (all detached except ``loss``, which keeps
    its autograd graph on the policy model's device):

        loss:             -log sigmoid(beta * margin)
        margins:          beta * [(pi_c - ref_c) - (pi_r - ref_r)]
        policy_chosen:    log pi(yw|x)      (sum over response tokens, incl. EOS)
        policy_rejected:  log pi(yl|x)
        ref_chosen:       log pi_ref(yw|x)
        ref_rejected:     log pi_ref(yl|x)

    The policy and reference models may live on different devices; results are
    returned on the policy model's device.
    """
    device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    prompt_ids, chosen_ids = _tokenize_prompt_and_response(
        prompt, response_chosen, tokenizer
    )
    _, rejected_ids = _tokenize_prompt_and_response(
        prompt, response_rejected, tokenizer
    )

    policy_chosen = _response_log_prob(lm, prompt_ids, chosen_ids, device)
    policy_rejected = _response_log_prob(lm, prompt_ids, rejected_ids, device)
    with torch.no_grad():
        ref_chosen = _response_log_prob(
            lm_ref, prompt_ids, chosen_ids, ref_device
        ).to(device)
        ref_rejected = _response_log_prob(
            lm_ref, prompt_ids, rejected_ids, ref_device
        ).to(device)

    margins = beta * (
        (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    )
    return {
        "loss": -F.logsigmoid(margins),
        "margins": margins.detach(),
        "policy_chosen": policy_chosen.detach(),
        "policy_rejected": policy_rejected.detach(),
        "ref_chosen": ref_chosen.detach(),
        "ref_rejected": ref_rejected.detach(),
    }


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Per-instance DPO loss:

        L = -log sigmoid(beta * [(log pi(yw|x) - log pi_ref(yw|x))
                                 - (log pi(yl|x) - log pi_ref(yl|x))])

    The policy and reference models may live on different devices; the loss is
    returned on the policy model's device.
    """
    return compute_per_instance_dpo_components(
        lm, lm_ref, tokenizer, beta, prompt, response_chosen, response_rejected
    )["loss"]
