from cs336_basics import tokenizer
from cs336_basics.bpe import find_chunk_boundaries
import pathlib
from multiprocessing import Pool
import numpy as np

def process_chunk(args):
    input_path, start, end, tokenizer = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)
        chunk = chunk.decode("utf-8")
        ids = tokenizer.encode(chunk)
        return np.array(ids, dtype=np.uint16)

def encode_(file_path, tokenizer):
    with open(file_path, "rb") as f:
        num_chunks = 2048
        num_processes = 32
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
        args_list = [
            (file_path, start, end, tokenizer)
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        parts = []
        with Pool(processes=num_processes) as pool:
            for result in pool.imap(process_chunk, args_list, chunksize=1):
                parts.append(result)
                
        ids = np.concatenate(parts)
        np.save(file_path.with_suffix(".npy"), ids)
                
if __name__ == "__main__":
    ppath = pathlib.Path(__file__).resolve().parent.parent

    ts_tokenizer = tokenizer.tokenizer.from_files(
        vocab_filepath="vocab_ts.json",
        merges_filepath="merges_ts.txt",
        special_tokens=["<|endoftext|>"]
    )
    ts_valid_path = ppath / "data/TinyStoriesV2-GPT4-valid.txt"    
    ts_train_path = ppath / "data/TinyStoriesV2-GPT4-train.txt"
    
    encode_(ts_valid_path, ts_tokenizer)
    encode_(ts_train_path, ts_tokenizer)

    owt_tokenizer = tokenizer.tokenizer.from_files(
        vocab_filepath="vocab_owt.json",
        merges_filepath="merges_owt.txt",
        special_tokens=["<|endoftext|>"]
    )
    owt_valid_path = ppath / "data/owt_valid.txt"
    owt_train_path = ppath / "data/owt_train.txt"
    encode_(owt_valid_path, owt_tokenizer)
    encode_(owt_train_path, owt_tokenizer)
  