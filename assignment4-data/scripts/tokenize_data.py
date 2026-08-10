"""把过滤后的数据 tokenize 并序列化成 np.uint16 二进制文件（流式处理，内存恒定）。

输入格式（filter_wet.py 的输出）：
  - 每篇文档的正文是一行或多行普通文本；
  - 文档与文档之间用单独一行 "<|endoftext|>" 分隔。

处理逻辑（按「文档」而非「行」处理，这点很重要）：
  - 按 "<|endoftext|>" 把文件切成若干篇文档；
  - 每篇文档**整体**做 tokenizer.encode（保留文档内部的换行符 \n）；
  - 每篇文档末尾追加一个 eos_token_id 作为文档边界。

为什么必须整篇 encode 而不是逐行：
  逐行处理时如果用 splitlines() 就会把换行符丢掉，导致产出的 token 流里完全没有
  换行 token（GPT-2 里 "\n" = token 198），而真实网页文本（以及 C4 验证集）含有
  大量换行。这种分布错配会让模型在验证集上表现极差。同时逐行 encode 还会把上下
  行的词粘连在一起（例如 "2024" 和 "December" 变成 "2024December"）。

为什么必须流式处理：
  过滤后的数据可达数十 GB（1000 个 WET 文件约 36 GB 文本 / 约 8.5B token）。
  如果先把所有文档读进 list、再把所有 token id 收集成一个 Python list，
  仅token部分就需要 8.5e9 ×~36 字节 ≈ 300 GB 内存，必然 OOM。
  因此这里改为：一次只处理一个输入文件，边tokenize 边以 np.uint16 追加写盘，
  内存占用与总数据量无关。

输出：
  - 一个扁平的 np.uint16 二进制文件，等价于 ids_array.tofile(output_path)，
    与题目示例代码及 cs336_basics 训练脚本兼容。

用法:
  uv run python scripts/tokenize_data.py \
      --input data/filtered_full \
      --output data/tokenized/data_full.bin
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# filter_wet.py 写入时使用的文档分隔标记
END_OF_TEXT_MARKER = "<|endoftext|>"

# 每个 worker 进程内缓存的 tokenizer（fork 后各进程独立持有）
_tokenizer_cache = None
_model_name = "gpt2"


def _get_tokenizer():
    """按需加载 tokenizer 并在进程内缓存。

    设置 model_max_length 为一个很大的值，避免 encode 超长文本时刷
    "Token indices sequence length is longer than..." 的告警（我们只是编码、
    不喂给模型，长度不受 1024 上下文限制）。
    """
    global _tokenizer_cache
    if _tokenizer_cache is None:
        tok = AutoTokenizer.from_pretrained(_model_name)
        tok.model_max_length = int(1e9)
        _tokenizer_cache = tok
    return _tokenizer_cache


def _tokenize_document(doc: str) -> list[int]:
    """把一篇完整文档编码成 token id 列表，并在末尾追加 eos。

    doc 内部保留原有换行符，因此换行会被正常编码成 token。
    """
    tokenizer = _get_tokenizer()
    return tokenizer.encode(doc) + [tokenizer.eos_token_id]


def split_documents(content: str) -> list[str]:
    """按 <|endoftext|> 切分出各篇文档（保留文档内部换行）。"""
    docs = []
    for doc in content.split(END_OF_TEXT_MARKER):
        doc = doc.strip("\n")
        if doc.strip():
            docs.append(doc)
    return docs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="过滤后的数据（单个文件或目录）")
    parser.add_argument("--output", required=True, help="输出的 tokenized 二进制文件路径")
    parser.add_argument("--model", default="gpt2", help="tokenizer 模型名，默认 gpt2")
    parser.add_argument("--max-workers", type=int, default=None, help="并行 worker 数，默认用满所有核")
    parser.add_argument("--chunksize", type=int, default=16,
                        help="imap 的 chunksize：每个 worker 一次领取多少篇文档，减少进程间通信开销")
    args = parser.parse_args()

    global _model_name
    _model_name = args.model

    input_path = Path(args.input)
    if input_path.is_dir():
        file_paths = sorted(input_path.glob("*.warc.wet.gz"))
    else:
        file_paths = [input_path]
    print(f"待处理 {len(file_paths)} 个过滤输出文件")

    num_cpus = len(os.sched_getaffinity(0))
    max_workers = args.max_workers or num_cpus
    print(f"使用 {max_workers} 个 worker（本机 {num_cpus} 核）")

    # 主进程先加载tokenizer，fork 出的 worker 直接复用（Linux fork 语义）
    tokenizer = _get_tokenizer()
    eos_id = tokenizer.eos_token_id
    newline_id = tokenizer.encode("\n")[0]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    total_docs = 0
    n_newline = 0
    n_eos = 0

    # 流式：逐个输入文件处理，边tokenize 边追加写盘，内存不随总数据量增长
    with open(output_path, "wb") as fout, multiprocessing.Pool(processes=max_workers) as pool:
        for fp in tqdm(file_paths, desc="Tokenizing files"):
            documents = split_documents(fp.read_text(encoding="utf-8"))
            if not documents:
                continue

            file_ids: list[int] = []
            for ids in pool.imap(_tokenize_document, documents, chunksize=args.chunksize):
                file_ids.extend(ids)

            # 本文件的 token 立刻转成 uint16 落盘，随后释放内存
            arr = np.asarray(file_ids, dtype=np.uint16)
            arr.tofile(fout)

            total_tokens += arr.size
            total_docs += len(documents)
            n_newline += int((arr == newline_id).sum())
            n_eos += int((arr == eos_id).sum())
            del file_ids, arr

    print(f"\nTokenized {args.input} into {total_tokens:,} tokens（{total_docs:,} 篇文档）")
    print(f"写入 {output_path}（{output_path.stat().st_size / 1e9:.2f} GB）")

    # 校验：读回文件头尾确认可解析，并汇报换行 / eos 统计
    back = np.fromfile(output_path, dtype=np.uint16, count=300)
    print("\n校验:")
    print(f"  总 token 数            : {total_tokens:,}")
    print(f"  换行 token({newline_id}) 数: {n_newline:,}  (必须远大于 0)")
    print(f"  eos token({eos_id}) 数 : {n_eos:,}  (应等于文档数 {total_docs:,})")
    print(f"  文件大小 / 2 == token数: {output_path.stat().st_size // 2 == total_tokens}")
    print("\n样例 decode（前 300 token）:")
    print(tokenizer.decode(back.tolist()))


if __name__ == "__main__":
    main()
