import time
import tracemalloc
import pathlib
import json
from cs336_basics.bpe import train_bpe

def bytes_to_display(b: bytes) -> str:
    try:
        s = b.decode("utf-8")
        if s.isprintable():
            return s
    except:
        pass
    return b.hex()

def train_bpe_tinystories():
    input_path = (pathlib.Path(__file__).resolve().parent.parent) / "data/TinyStoriesV2-GPT4-train.txt"
    start_time = time.time()
    tracemalloc.start()

    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
    )

    cost_time = time.time() - start_time
    print(f"cost time: {cost_time:.2f} s")
    current, peak = tracemalloc.get_traced_memory()
    print(f"current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"peak memory usage: {peak / 1024 / 1024:.2f} MB")
    tracemalloc.stop()

    vocab_json = {bytes_to_display(v): k for k, v in vocab.items()}
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)
    
    with open("merges.txt", "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(f"{bytes_to_display(merge[0])} {bytes_to_display(merge[1])}\n")

    return (vocab, merges)

if __name__ == "__main__":
    train_bpe_tinystories()