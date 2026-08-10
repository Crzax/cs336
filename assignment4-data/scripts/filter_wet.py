"""过滤 Common Crawl WET 文件，产出高质量语言建模训练数据。


输入: /shared-data/english-wet-data/ 下的 2500 个 *.warc.wet.gz 文件。
      每个 WET 文件是一个 WARC 压缩包，内部含多条 "conversion" 记录，
      每条 conversion 记录 = 一篇网页的纯文本（已去 HTML）。
输出: 每篇保留下来的文本，按顺序写入输出文件（每篇之间用 <|endoftext|> 分隔）。

并行方式: ProcessPoolExecutor 多进程，每个 worker 处理一个 WET 文件。

涉及的第三方库：
  - fastwarc.warc.ArchiveIterator   : 迭代读取 WARC/WET 文件里的每条 record
  - fastwarc.warc.WarcRecordType    : 记录类型枚举（conversion 表示网页纯文本）
  - tldextract.TLDExtract           : 从 URL 提取主域名，用于域名黑名单过滤
  - concurrent.futures.ProcessPoolExecutor / as_completed : 多进程并行
  - tqdm                             : 进度条

用法:
  uv run python scripts/filter_wet.py --input-dir /shared-data/english-wet-data \
      --output-dir data/filtered --max-workers 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from collections import Counter
from pathlib import Path

from fastwarc.warc import ArchiveIterator, WarcRecordType
from tldextract import TLDExtract
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 可配置项
# ---------------------------------------------------------------------------
# 英文置信度阈值（复用文档里的 0.7）
LANG_THRESHOLD = 0.7

# 域名黑名单：把你在 docs/2.md 里观察到的垃圾站域名放这里（色情/赌博/SEO spam 等）。
# 形式是主域名，例如 {"pornsite.com", "gambling.bet"}，注意不要带 "www." 前缀。
BAD_DOMAINS: set[str] = set()

# 全局提取器（tldextract 首次调用会加载内置后缀表，放模块级避免每个 worker 重复加载）
_extractor = TLDExtract()


def extract_domain(url: str) -> str:
    """从 URL 提取主域名，例如 https://www.example.com/page -> "example.com"。

    tldextract 用法：
        info = TLDExtract()("https://www.example.com/a/b")
        info.subdomain == "www", info.domain == "example", info.suffix == "com"
    主域名通常是 f"{info.domain}.{info.suffix}"。
    """
    info = _extractor(url)
    return f"{info.domain}.{info.suffix}"


def apply_filters(url: str, text: str) -> tuple[bool, list[str]]:
    """对单条网页文本应用全部过滤，返回 (是否保留, 被丢弃时命中的规则列表)。

    过滤顺序建议（先快后慢，尽早丢弃省钱）：
        1. 域名黑名单（最快，字符串匹配）
        2. 空文本 / 过短（gopher_quality_filter 内部有长度判断）
        3. 语言过滤（identify_language，fastText，较慢）
        4. 质量过滤（gopher_quality_filter 启发式 / quality_classifier 分类器）
        5. 有害内容过滤（nsfw_detect / toxic_detect，最慢，各跑一个模型）
    每丢弃一层，把命中的规则名 append 到 dropped_rules，用于统计 breakdown。

    注意：PII 遮蔽是"替换"而非"丢弃"，所以不在本函数里做（见 apply_masking）。
    """
    dropped_rules: list[str] = []

    # 1. 域名黑名单
    URL = extract_domain(url) 
    if URL in BAD_DOMAINS:
        dropped_rules.append("bad_domain")
        return False, dropped_rules
    # 2. 语言过滤（复用 cs336_data.language_id.identify_language）
    from cs336_data.language_id import identify_language
    lang, score = identify_language(text)
    if lang != "en" or score < LANG_THRESHOLD:
        dropped_rules.append("not_english")
        return False, dropped_rules
    # 3. 质量过滤（复用 cs336_data.quality.gopher_quality_filter）
    from cs336_data.quality import gopher_quality_filter
    if not gopher_quality_filter(text):
        dropped_rules.append("low_quality")
        return False, dropped_rules
    # 4. 有害内容过滤（复用 cs336_data.harmful_detect.nsfw_detect/toxic_detect）
    from cs336_data.harmful_detect import nsfw_detect, toxic_detect
    if nsfw_detect(text)[0] == "nsfw":
        dropped_rules.append("nsfw")
        return False, dropped_rules
    if toxic_detect(text)[0] == "toxic":
        dropped_rules.append("toxic")
        return False, dropped_rules

    return True, dropped_rules


def apply_masking(text: str) -> str:
    """对保留下来的文本做 PII 遮蔽（替换，不丢弃）。"""
    from cs336_data.pii import mask_emails, mask_phone_numbers, mask_ips
    text, _ = mask_emails(text)
    text, _ = mask_phone_numbers(text)
    text, _ = mask_ips(text)
    return text


def process_single_wet_file(input_path: str, output_path: str) -> tuple[str, Counter]:
    """处理一个 WET 文件：读 -> 逐条过滤 -> 写保留文本。

    返回 (输出文件路径, 本文件的过滤统计 Counter)。
    返回统计而不是用全局变量，是为了多进程下能正确汇总（每个进程有独立内存）。
    """
    stats: Counter[str] = Counter()
    kept_texts: list[str] = []

    with open(input_path, "rb") as fh:
        # record_types=conversion：只在迭代层面返回 conversion 记录（WET 的正文）
        # 这样既过滤了多余记录，代码也更简洁。
        for record in ArchiveIterator(fh, record_types=WarcRecordType.conversion):
            # 从记录头拿 URL（用于域名过滤）；拿不到就空串
            try:
                url = record.headers["WARC-Target-URI"]
            except KeyError:
                url = ""
            # conversion 记录的正文就是纯文本字节，直接 decode
            text = record.reader.read().decode("utf-8", errors="replace")

            stats["total_records"] += 1

            keep, dropped_rules = apply_filters(url, text)
            if not keep:
                for rule in dropped_rules:
                    stats[f"dropped_{rule}"] += 1
                continue

            # 保留的文本做 PII 遮蔽
            text = apply_masking(text)
            kept_texts.append(text)

    stats["kept"] = len(kept_texts)

    # 写入：每篇文本之间用空行或 <|endoftext|> 分隔。之后 tokenize 时会再处理。
    # 这里用 "<|endoftext|>" 行分隔，便于后续按行读回。
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, t in enumerate(kept_texts):
            if i > 0:
                fout.write("<|endoftext|>\n")
            fout.write(t.rstrip("\n"))
            fout.write("\n")

    return output_path, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="存放 2500 个 WET 文件的目录")
    parser.add_argument("--output-dir", required=True, help="过滤结果的输出目录")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="并行 worker 数，默认用满所有 CPU 核")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wet_filepaths = sorted(input_dir.glob("*.warc.wet.gz"))
    print(f"找到 {len(wet_filepaths)} 个 WET 文件")

    # 单进程测试模式：只处理前 N 个，便于快速验证过滤效果
    n_debug = os.environ.get("DEBUG_N_FILES")
    if n_debug:
        wet_filepaths = wet_filepaths[: int(n_debug)]
        print(f"[debug] 只处理前 {len(wet_filepaths)} 个文件")

    num_cpus = len(os.sched_getaffinity(0))
    max_workers = args.max_workers or num_cpus
    print(f"使用 {max_workers} 个 worker（本机 {num_cpus} 核）")

    totals: Counter[str] = Counter()
    start = __import__("time").perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_single_wet_file,
                str(path),
                str(output_dir / path.name),
            )
            for path in wet_filepaths
        ]
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            _, stats = future.result()  # 输出路径这里用不到，忽略
            totals.update(stats)

    elapsed = __import__("time").perf_counter() - start
    print(f"\n总耗时: {elapsed:.1f}s")
    print(f"过滤 breakdown:")
    for key in sorted(totals):
        print(f"  {key}: {totals[key]}")


if __name__ == "__main__":
    main()
