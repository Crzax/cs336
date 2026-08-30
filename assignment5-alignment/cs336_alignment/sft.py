"""Packed instruction-tuning data loading (supplement 4.2.1).

Converts prompt-response pairs into fixed-length language-modeling sequences:
each document is rendered with the Alpaca SFT template, tokenized (the Llama
tokenizer prepends BOS), and terminated with the EOS token as a document
delimiter. All documents are concatenated into one token stream, then chopped
into consecutive non-overlapping chunks of ``seq_length``; labels are the same
stream shifted left by one, so they stay contiguous across chunk boundaries.
"""

import gzip
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

_ALPACA_SFT_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts_safety" / "alpaca_sft.prompt"
)


class PackedSFTDataset(Dataset):
    """Packed supervised-fine-tuning dataset for instruction tuning.

    Args:
        tokenizer: transformers tokenizer (default ``add_special_tokens=True``
            behavior is used, so Llama tokenizers prepend BOS).
        dataset_path: path to a .jsonl (or .jsonl.gz) file where each line is a
            JSON object with "prompt" and "response" keys.
        seq_length: number of tokens per packed sequence.
        shuffle: if True, shuffle documents before concatenation.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        dataset_path: str | Path,
        seq_length: int,
        shuffle: bool,
    ) -> None:
        with open(_ALPACA_SFT_PROMPT_PATH) as f:
            template = f.read()

        # Support both plain and gzipped JSONL.
        opener = gzip.open if str(dataset_path).endswith(".gz") else open
        with opener(dataset_path, "rt") as f:
            documents = [json.loads(line) for line in f if line.strip()]

        if shuffle:
            random.shuffle(documents)

        # Render documents with the Alpaca SFT template.
        texts = [
            template.format(
                instruction=document["prompt"],
                response=document["response"],
            ).strip()
            for document in documents
        ]

        # Tokenize in batches (the fast tokenizer parallelizes over the batch)
        # and append the EOS delimiter after every document. Batched calls give
        # the same result as per-document calls, including the prepended BOS.
        eos_token_id = tokenizer.eos_token_id
        token_ids: list[int] = []
        batch_size = 1024
        for start in range(0, len(texts), batch_size):
            for ids in tokenizer(texts[start : start + batch_size]).input_ids:
                token_ids.extend(ids)
                token_ids.append(eos_token_id)

        # Next-token prediction over the packed stream: inputs drop the last
        # token, labels drop the first. Trailing tokens that do not fill a
        # complete chunk are dropped.
        tokens = torch.tensor(token_ids, dtype=torch.long)
        num_sequences = (len(tokens) - 1) // seq_length
        self.seq_length = seq_length
        self.input_ids = tokens[:-1][: num_sequences * seq_length].view(
            num_sequences, seq_length
        )
        self.labels = tokens[1:][: num_sequences * seq_length].view(
            num_sequences, seq_length
        )

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Return a DataLoader yielding batches of packed SFT examples.

    Iterating once over the result is a single epoch; the final batch may be
    smaller than ``batch_size`` (``drop_last=False``).
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
