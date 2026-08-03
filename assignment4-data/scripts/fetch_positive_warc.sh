#!/usr/bin/env bash
# 并发抓取 positive_urls.txt 里的网页，输出多个 WARC 文件。
# 瓶颈是网络 I/O 而非 CPU，并发数可以远超核数。
# 用法: bash scripts/fetch_positive_warc.sh [并发数]   默认 128
set -u

JOBS="${1:-128}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/quality_classifier"
URLS="$DIR/positive_urls.txt"
OUT="$DIR/warc"

if [ ! -f "$URLS" ]; then
    echo "缺少 $URLS，请先运行 extract_positive_urls.py" >&2
    exit 1
fi

rm -rf "$OUT" && mkdir -p "$OUT"
cd "$OUT"

total=$(wc -l < "$URLS")
[ "$JOBS" -gt "$total" ] && JOBS="$total"

# 轮转切分(按行号取模)，而不是按连续块切。
# 原因: URL 列表里同域名的往往相邻，连续切会让某一份集中遇到慢站/死站而拖尾。
awk -v n="$JOBS" '{print > ("chunk_" (NR % n))}' "$URLS"

echo "[fetch] $total urls / $JOBS 并发, 开始 $(date +%H:%M:%S)"
start=$(date +%s)

for chunk in chunk_*; do
    (
        wget --timeout=5 --tries=1 --no-check-certificate \
             --user-agent="Mozilla/5.0 (compatible; cs336-crawler)" \
             --max-redirect=3 --dns-timeout=3 --connect-timeout=3 \
             -i "$chunk" \
             --warc-file="part_${chunk}" \
             -O /dev/null 2>"log_${chunk}.txt"
    ) &
done

# 每 15 秒报一次剩余进度
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    sleep 15
    running=$(jobs -rp | wc -l)
    got=$(ls -1 part_chunk_*.warc.gz 2>/dev/null | wc -l)
    echo "  [progress] 运行中=$running 已产出warc=$got 已用 $(( $(date +%s) - start ))s"
done
wait

rm -f chunk_*
echo "[fetch] 完成，耗时 $(( $(date +%s) - start ))s"
echo "[fetch] WARC 数量: $(ls -1 "$OUT"/*.warc.gz 2>/dev/null | wc -l)"
echo "[fetch] 总大小: $(du -sh "$OUT" | cut -f1)"
# 统计抓取成功率
ok=$(grep -h "HTTP request sent" log_chunk_*.txt 2>/dev/null | grep -c "200 OK" || true)
echo "[fetch] HTTP 200 响应数: $ok / $total"
