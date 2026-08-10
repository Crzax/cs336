"""从 Common Crawl 直接下载 N 个 WET 文件（不依赖 modal）。

官方 scripts/download_data.py 依赖 modal + cs336_data/wet_files.py 里未实现的
is_english，本地跑不了。本脚本只做一件事：把Common Crawl 某次爬取的 WET 文件
随机抽 N 个下载到本地目录，供 scripts/filter_wet.py 处理。

抽样方式与 cs336_data/wet_files.py 保持一致（同一 crawl、同一随机种子），
便于复现：
  crawl_id     = "CC-MAIN-2026-17"
  shuffle_seed = 336

每个 WET 文件约 65 MB（gz 压缩），约含 8-9M GPT-2 token。
参考：训练 16384 步 × 524288 tokens/step = 8.59B tokens，
      因此约需 1000 个文件才能做到 1 个 epoch。

特性：
  - 多线程并发下载（下载是IO 密集，用线程即可）
  - 断点续传友好：目标文件已存在且大小与服务器一致则跳过
  - 下载到 .tmp 再原子改名，避免中断后留下半个损坏文件

用法:
  uv run python scripts/download_wet.py --output-dir data/wet_batch --num-files 1000
  uv run python scripts/download_wet.py --output-dir data/wet_batch --num-files 1000 --workers 32
"""

from __future__ import annotations

import argparse
import gzip
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://data.commoncrawl.org/"
# 与 cs336_data/wet_files.py 中EnglishWetFiles 的默认值保持一致
DEFAULT_CRAWL_ID = "CC-MAIN-2026-17"
DEFAULT_SHUFFLE_SEED = 336


def fetch_wet_paths(crawl_id: str) -> list[str]:
    """下载并解析wet.paths.gz，返回该次爬取全部 WET 文件的相对路径列表。"""
    paths_url = f"{BASE_URL}crawl-data/{crawl_id}/wet.paths.gz"
    print(f"获取 WET 路径清单: {paths_url}")
    with urllib.request.urlopen(paths_url) as resp:
        raw = resp.read()
    paths = gzip.decompress(raw).decode().splitlines()
    paths = [p.strip() for p in paths if p.strip()]
    print(f"该次爬取共 {len(paths):,} 个 WET 文件")
    return paths


def download_one(rel_path: str, output_dir: Path) -> tuple[str, str]:
    """下载单个 WET 文件，返回 (文件名, 状态)。状态为 "skipped" / "downloaded" / "failed: ..."。"""
    url = BASE_URL + rel_path
    filename = rel_path.split("/")[-1]
    destination = output_dir / filename

    try:
        # 已存在且大小与服务器一致 -> 跳过（断点续传）
        if destination.exists():
            request = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(request) as resp:
                remote_size = int(resp.headers.get("Content-Length", -1))
            if remote_size > 0 and destination.stat().st_size == remote_size:
                return filename, "skipped"

        # 先写 .tmp，成功后原子改名，避免中断留下损坏文件
        tmp_path = destination.with_suffix(destination.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(destination)
        return filename, "downloaded"
    except Exception as exc:  # noqa: BLE001 - 单个文件失败不应中断整批
        return filename, f"failed: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, help="WET 文件保存目录")
    parser.add_argument("--num-files", type=int, default=1000,
                        help="下载多少个 WET 文件（1000 个约 65GB，约够 1 个 epoch）")
    parser.add_argument("--crawl-id", default=DEFAULT_CRAWL_ID, help="Common Crawl 爬取批次 ID")
    parser.add_argument("--seed", type=int, default=DEFAULT_SHUFFLE_SEED,
                        help="随机抽样种子（与 wet_files.py 一致，默认 336）")
    parser.add_argument("--workers", type=int, default=16, help="并发下载线程数")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = fetch_wet_paths(args.crawl_id)

    # 固定种子随机抽样，保证可复现
    n = min(args.num_files, len(all_paths))
    rng = random.Random(args.seed)
    selected = rng.sample(all_paths, n)
    print(f"随机抽取 {n} 个文件（seed={args.seed}），并发 {args.workers} 线程下载到 {output_dir}")

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, p, output_dir): p for p in selected}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading WET"):
            filename, status = future.result()
            if status == "downloaded":
                stats["downloaded"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            else:
                stats["failed"] += 1
                failures.append(f"{filename}: {status}")

    print("\n下载完成:")
    print(f"  新下载 : {stats['downloaded']}")
    print(f"  已存在 : {stats['skipped']}")
    print(f"  失败   : {stats['failed']}")
    if failures:
        print("\n失败列表（可重跑本脚本自动续传）:")
        for line in failures[:20]:
            print(f"  {line}")
        if len(failures) > 20:
            print(f"  ... 其余 {len(failures) - 20} 个")

    total_bytes = sum(f.stat().st_size for f in output_dir.glob("*.warc.wet.gz"))
    n_files = len(list(output_dir.glob("*.warc.wet.gz")))
    print(f"\n目录现有 {n_files} 个 WET 文件，共 {total_bytes / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
