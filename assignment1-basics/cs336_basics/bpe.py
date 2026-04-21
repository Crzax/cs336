import re
import regex
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def train_bpe(input_path, vocab_size, special_tokens=None) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE model on the given input file.
    Returns a tuple of (vocab, merges), where vocab is a dict of byte strings to integers,
    and merges is a list of tuples of byte strings representing the merges.
    """
    vocab = {i : bytes([i]) for i in range(256)} # 词表
    merges = [] # 合并结果

    with open(input_path, "rb") as f:
        # 分割语料
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        word_counts = {} 
        pair_counts = {}
        pair_to_word = {}

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            text = f.read(end - start).decode("utf-8", errors="ignore")
            pattern = "|".join(re.escape(token) for token in special_tokens)
            parts = re.split(pattern, text)
            # 预分词
            # dict[tuple(bytes,...), int]
            
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for part in parts:
                matches = regex.finditer(PAT, part)
                for match in matches:
                    token_str = match.group()
                    token_bytes = token_str.encode("utf-8")
                    key = tuple(bytes([b]) for b in token_bytes)
                    word_counts[key] = word_counts.get(key, 0) + 1
            
        
        for word, count in word_counts.items():
            if len(word) > 1:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])        
                    pair_counts[pair] = pair_counts.get(pair, 0) + count
                    pair_to_word.setdefault(pair, set()).add(word)

        nums_merge = vocab_size - 256 - len(special_tokens)
        for _ in range(nums_merge):
            max_pair= max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
            pair_bytes = max_pair[0] + max_pair[1]
            merges.append(max_pair)
            vocab[len(vocab)] = pair_bytes
            
            # 合并 & update
            for word in list(pair_to_word[max_pair]):
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                            new_word.append(word[i] + word[i+1])
                            i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                for i in range(len(word) - 1):
                    pair = (word[i], word[i+1])
                    pair_counts[pair] -= word_counts[word]
                    pair_to_word[pair].discard(word)   
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]

                new_word = tuple(new_word)
                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + word_counts[word]
                    pair_to_word.setdefault(pair, set()).add(new_word)
                word_counts[new_word] = word_counts.pop(word) + word_counts.get(new_word, 0)
            
            pair_counts.pop(max_pair, None)
            pair_to_word.pop(max_pair, None)
            
    for spt in special_tokens:
        vocab[len(vocab)] = spt.encode("utf-8")
    return (vocab, merges)