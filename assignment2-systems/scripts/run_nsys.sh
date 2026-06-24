#!/usr/bin/env bash
# Profile forward / backward / optimizer with nsys across 2 model sizes
# (Table 1) x 3 power-of-two context lengths (>128) = 6 combos.
#
# Designed for the H20 box (8 x ~95GB). Each profile uses ONE GPU.
#
# Usage:
#   bash scripts/run_nsys.sh                 # sequential, all on GPU 0
#   PARALLEL=1 bash scripts/run_nsys.sh      # spread 6 jobs across the 8 GPUs
#   GPU=3 bash scripts/run_nsys.sh           # sequential, pin to GPU 3
#   CTX="256 1024 4096" bash scripts/run_nsys.sh   # override context lengths
#
# Outputs: scripts/nsys_out/nsys_<size>_ctx<L>.nsys-rep (+ .sqlite)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_REL="cs336_systems/benchmarking_script.py"
OUT_DIR="${SCRIPT_DIR}/nsys_out"
mkdir -p "$OUT_DIR"

# Override with: RUNNER="python" ...   (if you don't use uv)
RUNNER=(${RUNNER:-uv run python})

WARMUP="${WARMUP:-5}"
STEPS="${STEPS:-10}"
RUN_TYPE="${RUN_TYPE:-full}"          # full = forward+backward+optimizer
PARALLEL="${PARALLEL:-0}"
GPU="${GPU:-0}"

# --- the 6 combos --------------------------------------------------------
# size  d_model  d_ff   num_layers  num_heads
MODELS=(
  "small   768   3072   12  12"
  "medium  1024  4096   24  16"
)
# 3 power-of-two context lengths (>128). The largest should be the longest
# you can fit in memory on the H20 -- see "probe the max" note at bottom.
CONTEXT_LENGTHS=(${CTX:-256 512 2048})

# nsys flags shared by every run.
NSYS_FLAGS=(
  --trace=cuda,nvtx,osrt,cudnn,cublas
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --force-overwrite=true
)

cd "$PROJECT_DIR"
echo "Project dir : $PROJECT_DIR"
echo "Output dir  : $OUT_DIR"
echo "Runner      : ${RUNNER[*]}"
echo "Mode        : $([ "$PARALLEL" = 1 ] && echo parallel || echo "sequential (GPU $GPU)")"
echo "Context len : ${CONTEXT_LENGTHS[*]}"
echo

# Build the flat list of jobs.
JOBS=()
for m in "${MODELS[@]}"; do
  for L in "${CONTEXT_LENGTHS[@]}"; do
    JOBS+=("$m|$L")
  done
done

run_one() {
  # $1 = "size d_model d_ff layers heads", $2 = context_length, $3 = gpu id
  read -r SIZE D_MODEL D_FF N_LAYERS N_HEADS <<< "$1"
  local L="$2" GPU_ID="$3"
  local TAG="${SIZE}_ctx${L}"
  echo ">>> [GPU $GPU_ID] size=$SIZE ctx=$L (d_model=$D_MODEL layers=$N_LAYERS heads=$N_HEADS)"

  CUDA_VISIBLE_DEVICES="$GPU_ID" nsys profile \
    -o "${OUT_DIR}/nsys_${TAG}" \
    "${NSYS_FLAGS[@]}" \
    "${RUNNER[@]}" "$SCRIPT_REL" \
      --run_type "$RUN_TYPE" \
      --d_model "$D_MODEL" --d_ff "$D_FF" \
      --num_layers "$N_LAYERS" --num_heads "$N_HEADS" \
      --context_length "$L" \
      --warmup "$WARMUP" --steps "$STEPS" \
      --nsys \
    > "${OUT_DIR}/nsys_${TAG}.log" 2>&1 \
    && echo "    [OK]   ${TAG}" \
    || echo "    [FAIL] ${TAG} (likely OOM, see ${OUT_DIR}/nsys_${TAG}.log)"
}

if [ "$PARALLEL" = 1 ]; then
  # Spread jobs across all visible GPUs, one job per GPU, in waves.
  NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
  echo "Detected $NGPU GPUs; running in parallel."
  i=0
  for job in "${JOBS[@]}"; do
    GPU_ID=$(( i % NGPU ))
    run_one "${job%|*}" "${job#*|}" "$GPU_ID" &
    i=$(( i + 1 ))
    # Throttle to NGPU concurrent jobs.
    if (( i % NGPU == 0 )); then wait; fi
  done
  wait
else
  for job in "${JOBS[@]}"; do
    run_one "${job%|*}" "${job#*|}" "$GPU"
  done
fi

echo
echo "Done. Reports in $OUT_DIR"
echo "Read forward time with:"
echo "  nsys stats --report nvtx_sum ${OUT_DIR}/nsys_small_ctx512.nsys-rep"

# --- probe the max context length (run once, by hand) --------------------
# To find the largest power-of-two ctx that fits on the H20, try:
#   CTX=2048 bash scripts/run_nsys.sh
#   CTX=4096 bash scripts/run_nsys.sh
#   CTX=8192 bash scripts/run_nsys.sh
# Watch for [FAIL]/OOM in the logs; use the last size that succeeded as your
# "largest" context length, then set CTX="256 512 <max>".
