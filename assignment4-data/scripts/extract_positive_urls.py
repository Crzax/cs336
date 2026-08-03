"""从单个 Wikipedia dump shard 里抽取 <ref> 引用中的外部 URL，并随机采样。

用法:
    uv run python scripts/extract_positive_urls.py --num-urls 3000
"""

import argparse
import bz2
import random
import re
import urllib.request
from pathlib import Path

DUMP_DATE = "20260501"
SHARD = f"enwiki-{DUMP_DATE}-pages-articles-multistream1.xml-p1p41242.bz2"
DATA_DIR = Path(__file__).parent.parent / "data/quality_classifier"

# 与 download_data.py 中一致的 URL 正则
URL_RE = re.compile(
    r"\b(?:https?|telnet|gopher|file|wais|ftp):[\w/#~:.?+=&%@!\-.:?\\-]+?"
    r"(?=[.:?\-]*(?:[^\w/#~:.?+=&%@!\-.:?\-]|$))"
)
REF_RE = re.compile(r"&lt;ref&gt(.*?)&lt;/ref&gt;")

# Wikimedia 会拒绝 Python 默认的 User-Agent（返回 403），必须自报身份
USER_AGENT = "cs336-student/1.0 (course assignment; contact: student@example.com)"

# 这些域名要么是聚合/归档站（抓回来多是样板页），要么不是 HTML，直接跳过
SKIP_PATTERNS = (
    "ghostarchive.org", "web.archive.org", "archive.today", "archive.is",
    "doi.org", "worldcat.org", "jstor.org", "wikipedia.org", "wikimedia.org",
    "youtube.com", "twitter.com", "facebook.com", "books.google",
)


def should_skip(url: str) -> bool:
    low = url.lower()
    if low.endswith((".pdf", ".jpg", ".png", ".gif", ".zip", ".mp3", ".mp4")):
        return True
    return any(p in low for p in SKIP_PATTERNS)


def download_dump(url: str, dest: Path) -> None:
    """带 User-Agent 的流式下载（走环境变量里的 http(s)_proxy）。"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as fh:
            while chunk := resp.read(1 << 20):  # 1MB
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r[download] {done / 1e6:.0f}/{total / 1e6:.0f} MB "
                          f"({done / total:.1%})", end="", flush=True)
        print()
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-urls", type=int, default=3000, help="采样多少个 URL")
    parser.add_argument("--seed", type=int, default=336)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dump_path = DATA_DIR / SHARD

    # 1. 下载 shard（约 283MB），已存在则跳过
    if not dump_path.exists():
        print(f"[download] {SHARD} (~283MB) ...", flush=True)
        download_dump(
            f"https://dumps.wikimedia.org/enwiki/{DUMP_DATE}/{SHARD}", dump_path
        )
    print(f"[download] ready: {dump_path}", flush=True)

    # 2. 流式解压 + 抽 URL（不把 XML 落盘）
    urls: set[str] = set()
    with bz2.open(dump_path, "rt", errors="ignore") as f:
        for i, line in enumerate(f):
            if "&lt;ref&gt" not in line:
                continue
            for ref in REF_RE.findall(line):
                for url in URL_RE.findall(ref):
                    if not should_skip(url):
                        urls.add(url)
            if i % 2_000_000 == 0 and i:
                print(f"[scan] line={i:,} urls={len(urls):,}", flush=True)

    print(f"[scan] done, unique urls: {len(urls):,}", flush=True)

    # 3. 随机采样
    random.seed(args.seed)
    pool = sorted(urls)  # 先排序保证可复现
    sampled = random.sample(pool, min(args.num_urls, len(pool)))

    out = DATA_DIR / "positive_urls.txt"
    out.write_text("\n".join(sampled) + "\n")
    print(f"[write] {len(sampled)} urls -> {out}", flush=True)
    print(f"[hint] 可删除 dump 释放空间: rm {dump_path}")


if __name__ == "__main__":
    main()
