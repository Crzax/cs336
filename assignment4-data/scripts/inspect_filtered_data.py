"""检查过滤后的数据质量（Assignment 4, Problem: inspect_filtered_data）。

(a) 从过滤后的输出目录里随机抽 5 篇保留文档，评估是否适合语言建模。
(b) 重新跑过滤逻辑，收集被丢弃的文档，随机抽 5 篇，说明被哪条规则删掉、是否合理。
(c) 打印过滤统计，辅助撰写数据管线迭代报告。

用法:
  uv run python scripts/inspect_filtered_data.py \
      --filtered-dir data/filtered \
      --wet-dir data/wet_batch \
      --seed 336
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType

# scripts/ 不是 Python 包（没有 __init__.py），直接运行本脚本时不能用相对/绝对包导入。
# 把 scripts 目录加进 sys.path，当作普通模块导入 filter_wet。
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filter_wet import apply_filters
# 每篇文档最多打印的字符数（文档可能很长，只展示开头即可）
MAX_EXCERPT_CHARS = 100000
# 每篇文档开头最多打印的行数
MAX_EXCERPT_LINES = 60


def iter_filtered_docs(filtered_dir: Path):
    """遍历过滤后的输出目录，按 <|endoftext|> 把一篇篇文档切出来。

    输出目录里每个文件是一批保留文本，篇与篇之间用单独一行的
    "<|endoftext|>" 分隔（见 filter_wet.py 的写入逻辑）。
    """
    for fp in sorted(filtered_dir.glob("*.warc.wet.gz")):
        text = fp.read_text(encoding="utf-8")
        for doc in text.split("<|endoftext|>"):
            doc = doc.strip()
            if doc:
                yield doc


def iter_wet_docs(wet_dir: Path):
    """遍历原始 WET 目录，逐条 record 产出 (url, text)。

    和 filter_wet.py 的读取方式一致：record_types=conversion 只拿正文。
    """
    for fp in sorted(wet_dir.glob("*.warc.wet.gz")):
        with open(fp, "rb") as fh:
            for record in ArchiveIterator(fh, record_types=WarcRecordType.conversion):
                try:
                    url = record.headers["WARC-Target-URI"]
                except KeyError:
                    url = ""
                text = record.reader.read().decode("utf-8", errors="replace")
                yield url, text
  
def compute_metrics(text: str) -> dict[str, object]:
    """计算几个用于质量评估的轻量指标。"""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)
    lines = [l for l in text.splitlines() if l.strip()]
    n_lines = len(lines)
    avg_word_len = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    return {
        "words": n_words,
        "chars": n_chars,
        "lines": n_lines,
        "avg_word_len": round(avg_word_len, 2),
    }


def excerpt(text: str) -> str:
    """返回文档开头的一段（限制字符数和行数）。"""
    lines = text.splitlines()
    out_lines = []
    total = 0
    for ln in lines:
        out_lines.append(ln)
        total += len(ln) + 1
        if len(out_lines) >= MAX_EXCERPT_LINES or total >= MAX_EXCERPT_CHARS:
            break
    body = "\n".join(out_lines)
    if len(body) > MAX_EXCERPT_CHARS:
        body = body[:MAX_EXCERPT_CHARS] + " …"
    return body


def sample_kept(filtered_dir: Path, rng: random.Random) -> None:
    """(a) 从保留数据里随机抽 5 篇并打印。"""
    print("=" * 80)
    print("(a) 保留数据中随机抽样的 5 篇")
    print("=" * 80)

    docs = list(iter_filtered_docs(filtered_dir))
    print(f"\n[info] 共 {len(docs)} 篇保留文档\n")

    for i, doc in enumerate(rng.sample(docs, k=min(5, len(docs))), 1):
        m = compute_metrics(doc)
        print(f"--- 示例 {i} 词数={m['words']} 行数={m['lines']} 平均词长={m['avg_word_len']} ---")
        print(excerpt(doc))
        print(f"\n>>> 评价: （这里写 1-2 句话：内容是否连贯 / 是否单一主题 / 有没有导航噪声，结论是否值得用于 LM）\n")
        print("-" * 80)


def sample_discarded(wet_dir: Path, rng: random.Random) -> None:
    """(b) 重新跑过滤，收集被丢弃的文档并随机抽 5 篇。

    注意：主流程 filter_wet.py 只存保留文本，没存被丢弃的，
    所以这里要重读原始 WET 再跑一遍 apply_filters。
    只采样前 N 篇丢弃样本，避免内存爆炸（丢弃量很大）。
    """
    print("=" * 80)
    print("(b) 被过滤掉的文档（重跑过滤逻辑收集）")
    print("=" * 80)

    discarded: list[tuple[str, list[str], str]] = []  # (url, rules, text)
    breakdown: Counter[str] = Counter()
    n_scanned = 0

    for url, text in iter_wet_docs(wet_dir):
        n_scanned += 1
        keep, rules = apply_filters(url, text)
        if not keep:
            for r in rules:
                breakdown[r] += 1
            # 控制内存：丢弃样本很多，抽样到 N 篇就够展示用了
            if len(discarded) < 200:
                discarded.append((url, rules, text))
        # 扫描到足够多就提前停，节省时间（丢弃量远大于保留量）
        if n_scanned >= 2000:
            break

    print(f"\n[info] 扫描 {n_scanned} 条，丢弃 breakdown: {dict(breakdown)}\n")

    for i, (url, rules, text) in enumerate(rng.sample(discarded, k=min(5, len(discarded))), 1):
        print(f"--- 示例 {i} 命中规则: {', '.join(rules)} ---")
        print(f"url: {url}")
        print(excerpt(text))
        print(f"\n>>> 评价: （这里写 1-2 句话：被 {rules} 删除是否合理，是否有误杀）\n")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-dir", required=True, help="filter_wet.py 的输出目录")
    parser.add_argument("--wet-dir", required=True, help="原始 WET 文件目录")
    parser.add_argument("--seed", type=int, default=336, help="随机种子，保证可复现")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    sample_kept(Path(args.filtered_dir), rng)
    sample_discarded(Path(args.wet_dir), rng)


if __name__ == "__main__":
    main()
