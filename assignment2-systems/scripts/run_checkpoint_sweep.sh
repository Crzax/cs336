#!/usr/bin/env bash
# Sweep flat (non-nested) gradient checkpointing segment sizes on the XL model
# to validate the sqrt(N) optimum predicted in the writeup.
#
# XL has N_LAYERS=32, so sqrt(N) ≈ 5.66. We try segment sizes that evenly
# divide 32: 4 (8 segments), 8 (4 segments), 16 (2 segments), plus a
# baseline run with no checkpointing (seg=0). Each run dumps a CUDA memory
# snapshot pickle + a log line containing peak_allocated GiB.
#
# Usage:
#   bash scripts/run_checkpoint_sweep.sh                       # fp32, GPU 0, default sizes
#   GPU=2 bash scripts/run_checkpoint_sweep.sh
#   MIXED=1 bash scripts/run_checkpoint_sweep.sh               # BF16 autocast
#   SEGMENTS="0 4 8 16" bash scripts/run_checkpoint_sweep.sh   # custom sweep
#   BATCH=2 CTX=2048 bash scripts/run_checkpoint_sweep.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_REL="cs336_systems/benchmarking_script.py"
OUT_DIR="${PROJECT_DIR}/reports/mem"
SUMMARY="${OUT_DIR}/checkpoint_sweep_summary.tsv"
mkdir -p "$OUT_DIR"

RUNNER=(${RUNNER:-uv run python})

# XL config from Table 1.
D_MODEL=2560
D_FF=10240
N_LAYERS=32
N_HEADS=32

GPU="${GPU:-0}"
WARMUP="${WARMUP:-3}"
STEPS="${STEPS:-3}"
BATCH="${BATCH:-4}"
CTX="${CTX:-512}"
RUN_TYPE="${RUN_TYPE:-full}"          # full = fwd+bwd+optimizer
SEGMENTS=(${SEGMENTS:-0 4 8 16})      # 0 = baseline (no checkpoint)

AMP=""; PREC="fp32"
[ "${MIXED:-0}" = "1" ] && { AMP="--mixed_precision"; PREC="bf16"; }

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_DIR"
echo "Project dir : $PROJECT_DIR"
echo "Output dir  : $OUT_DIR"
echo "GPU         : $GPU   precision: $PREC   batch: $BATCH   ctx: $CTX"
echo "Run type    : $RUN_TYPE"
echo "Segments    : ${SEGMENTS[*]}   (0 = baseline / no checkpointing)"
echo "Alloc conf  : $PYTORCH_CUDA_ALLOC_CONF"
echo

# Header for the summary TSV
printf "segment_size\tnum_segments\tpeak_GiB\tsnapshot\tstatus\n" > "$SUMMARY"

for SEG in "${SEGMENTS[@]}"; do
  if [ "$SEG" -eq 0 ]; then
    TAG="xl_${RUN_TYPE}_ctx${CTX}_${PREC}_baseline"
    NSEG="-"
  else
    TAG="xl_${RUN_TYPE}_ctx${CTX}_${PREC}_seg${SEG}"
    NSEG=$(( N_LAYERS / SEG ))
  fi
  SNAP="${OUT_DIR}/mem_${TAG}.pickle"
  LOG="${OUT_DIR}/${TAG}.log"

  echo ">>> [GPU $GPU] segment_size=$SEG (num_segments=$NSEG) -> $SNAP"
  CUDA_VISIBLE_DEVICES="$GPU" "${RUNNER[@]}" "$SCRIPT_REL" \
    --run_type "$RUN_TYPE" \
    --d_model "$D_MODEL" --d_ff "$D_FF" \
    --num_layers "$N_LAYERS" --num_heads "$N_HEADS" \
    --context_length "$CTX" --batch_size "$BATCH" \
    --warmup "$WARMUP" --steps "$STEPS" \
    --memory_profile --memory_snapshot "$SNAP" \
    --checkpoint_segment "$SEG" \
    $AMP 2>&1 | tee "$LOG"
  STATUS=${PIPESTATUS[0]}

  if [ "$STATUS" -eq 0 ]; then
    # Parse "peak_allocated=XX.XX GiB" from the log
    PEAK=$(grep -oE 'peak_allocated=[0-9.]+' "$LOG" | tail -n1 | cut -d= -f2)
    PEAK=${PEAK:-NA}
    printf "%s\t%s\t%s\t%s\tOK\n" "$SEG" "$NSEG" "$PEAK" "$SNAP" >> "$SUMMARY"
    echo "    [OK]   peak=${PEAK} GiB"
  else
    printf "%s\t%s\tNA\t%s\tFAIL\n" "$SEG" "$NSEG" "$SNAP" >> "$SUMMARY"
    echo "    [FAIL] see $LOG (likely OOM)"
  fi
  echo
done

echo "Done. Summary:"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$SUMMARY"
else
  cat "$SUMMARY"
fi
echo
echo "Snapshots in $OUT_DIR (load each .pickle at https://pytorch.org/memory_viz)."
