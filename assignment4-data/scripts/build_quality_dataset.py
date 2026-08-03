"""构建 fastText 质量分类器的训练集。

正样本: data/quality_classifier/warc/*.warc.gz  -> __label__wiki
负样本: data/CC/example.warc.gz                 -> __label__cc

用法:
    uv run python scripts/build_quality_dataset.py --max-per-class 3000
"""

import argparse
import random
from pathlib import Path

from fastwarc.warc import ArchiveIterator

from cs336_data.extract import extract_text_from_html_bytes
from cs336_data.language_id import identify_language
from cs336_data.quality import gopher_quality_filter

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data/quality_classifier"
CC_WARC = ROOT / "data/CC/example.warc.gz"

MAX_CHARS = 5000        # 单样本截断长度，防止超长文档主导训练
LANG_THRESHOLD = 0.6    # 英文置信度下限

# wget 即使收到 403/404/503 也会把错误页写进 WARC，这些页面必须剔除，
# 否则正样本里会混入 "403 ERROR The request could not be satisfied" 之类的垃圾。
ERROR_PAGE_MARKERS = (
    "403 error", "404 not found", "request could not be satisfied",
    "request blocked", "access denied", "página no encontrada",
    "are you a robot", "enable javascript", "captcha",
    "service unavailable", "site can't be reached", "error 500",
)


def looks_like_error_page(text: str) -> bool:
    head = text[:400].lower()
    return any(m in head for m in ERROR_PAGE_MARKERS)


def iter_warc_texts(path: Path, *, only_http_ok: bool = False):
    """遍历一个 WARC 文件，yield 抽取出的正文文本。

    only_http_ok: 只保留 HTTP 200 的响应（用于我们自己 wget 抓的正样本）。
    """
    with open(path, "rb") as fh:
        for record in ArchiveIterator(fh):
            if record.headers.get("WARC-Type") != "response":
                continue
            if only_http_ok:
                status = getattr(record, "http_headers", None)
                code = getattr(status, "status_code", None) if status else None
                if code is not None and code != 200:
                    continue
            try:
                text = extract_text_from_html_bytes(record.reader.read())
            except Exception:
                continue
            if text and not looks_like_error_page(text):
                yield text


def clean(text: str, *, use_gopher: bool, use_harmful: bool) -> str | None:
    """清洗单条样本，不合格返回 None。"""
    text = text.strip()
    if not text:
        return None

    # 1. 只保留高置信度英文
    lang, score = identify_language(text)
    if lang != "en" or score < LANG_THRESHOLD:
        return None

    # 2. Gopher 规则过滤（能干掉导航样板页、超短页）
    if use_gopher and not gopher_quality_filter(text):
        return None

    # 3. TODO: 是否过滤 harmful 内容。开启会慢不少（每条要跑两个 fastText 模型），
    #    正样本建议开，负样本可以不开（我们本来就要 CC 里的低质内容当负样本）。
    if use_harmful:
        from cs336_data.harmful_detect import nsfw_detect, toxic_detect

        if nsfw_detect(text)[0] == "nsfw":
            return None
        if toxic_detect(text)[0] == "toxic":
            return None

    # 4. fastText 要求单行；截断超长文本
    return " ".join(text.split())[:MAX_CHARS]


def collect(paths: list[Path], limit: int, *, use_gopher: bool, use_harmful: bool,
            tag: str, only_http_ok: bool = False) -> list[str]:
    out: list[str] = []
    for path in paths:
        for raw in iter_warc_texts(path, only_http_ok=only_http_ok):
            cleaned = clean(raw, use_gopher=use_gopher, use_harmful=use_harmful)
            if cleaned:
                out.append(cleaned)
                if len(out) % 500 == 0:
                    print(f"[{tag}] kept {len(out)}", flush=True)
            if len(out) >= limit:
                return out
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=3000)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--harmful-filter", action="store_true",
                        help="正样本额外跑 nsfw/toxic 过滤（较慢）")
    args = parser.parse_args()

    pos_warcs = sorted((DATA_DIR / "warc").glob("*.warc.gz"))
    if not pos_warcs:
        raise SystemExit("没有正样本 WARC，请先跑 fetch_positive_warc.sh")

    print(f"[pos] {len(pos_warcs)} warc files")
    positives = collect(pos_warcs, args.max_per_class, use_gopher=True,
                        use_harmful=args.harmful_filter, tag="pos",
                        only_http_ok=True)
    print(f"[pos] total = {len(positives)}")

    # 关键: 负样本数量与正样本对齐，避免类别不平衡
    negatives = collect([CC_WARC], len(positives), use_gopher=False,
                        use_harmful=False, tag="neg")
    print(f"[neg] total = {len(negatives)}")

    n = min(len(positives), len(negatives))
    rows = [f"__label__wiki {t}" for t in positives[:n]]
    rows += [f"__label__cc {t}" for t in negatives[:n]]

    random.seed(args.seed)
    random.shuffle(rows)

    n_valid = int(len(rows) * args.valid_frac)
    (DATA_DIR / "valid.txt").write_text("\n".join(rows[:n_valid]) + "\n")
    (DATA_DIR / "train.txt").write_text("\n".join(rows[n_valid:]) + "\n")
    print(f"[write] train={len(rows) - n_valid} valid={n_valid} "
          f"(每类 {n} 条) -> {DATA_DIR}")


if __name__ == "__main__":
    main()
