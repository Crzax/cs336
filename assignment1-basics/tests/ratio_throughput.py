import random
from cs336_basics import tokenizer
import pathlib, time

def iter_docs(path, sep) :
    doc = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            doc += line
            while sep in doc:
                idx = doc.index(sep)
                yield doc[:idx]
                doc = doc[idx + len(sep):]
        if doc:
            yield doc

def sample_docs(docs_iter, k):
    docs = []
    i = 0
    for doc in docs_iter:
        if i < k:
            docs.append(doc)
        else:
            j = random.randint(0, i)
            if j < k:
                docs[j] = doc
        i += 1
    return docs

def get_ratio(docs, tokenizer):
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))
    return total_bytes / total_tokens

def get_throughput(docs, tokenizer):
    start_time = time.time()
    for doc in docs:
        tokenizer.encode(doc)
        
    cost_time = time.time() - start_time
    total_bytes = 0
    for doc in docs:
        total_bytes += len(doc.encode("utf-8"))
    return total_bytes / cost_time

if __name__ == "__main__":
    random.seed(42)
    ppath = pathlib.Path(__file__).resolve().parent.parent
    owt_path = ppath / "data/owt_valid.txt"
    ts_path = ppath / "data/TinyStoriesV2-GPT4-valid.txt"
    owt_tokenizer = tokenizer.tokenizer.from_files(
        vocab_filepath="vocab_owt.json",
        merges_filepath="merges_owt.txt",
        special_tokens=["<|endoftext|>"]
    )
    ts_tokenizer = tokenizer.tokenizer.from_files(
        vocab_filepath="vocab_ts.json",
        merges_filepath="merges_ts.txt",
        special_tokens=["<|endoftext|>"]
    )
    owt_docs_iter = iter_docs(owt_path, "<|endoftext|>")
    owt_sample_docs = sample_docs(owt_docs_iter, 10)

    ts_docs_iter = iter_docs(ts_path, "<|endoftext|>")
    ts_sample_docs = sample_docs(ts_docs_iter, 10)
    
    print(f'owt ratio is :{get_ratio(owt_sample_docs, owt_tokenizer)}')
    print(f'ts ratio is :{get_ratio(ts_sample_docs, ts_tokenizer)}')

    print(f're owt ratio is :{get_ratio(owt_sample_docs, ts_tokenizer)}')
    print(f're ts ratio is :{get_ratio(ts_sample_docs, owt_tokenizer)}')

    print(f'owt throughput is :{get_throughput(owt_sample_docs, owt_tokenizer)} byte/s')
