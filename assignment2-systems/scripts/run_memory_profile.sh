#!/usr/bin/env bash
# Memory-profile the xl model (Table 1) at context lengths 128 and 2048,
# for inference-only (forward) and a full training step (fwd+bwd+optimizer).
#
# Produces 4 snapshot pickles under reports/mem/, to be loaded into
# https://pytorch.org/memory_viz ("Active memory timeline").
#
# Usage:
#   bash scripts/run_memory_profile.sh                 # fp32, GPU 0
#   GPU=2 bash scripts/run_memory_profile.sh           # pin to GPU 2
#   MIXED=1 bash scripts/run_memory_profile.sh         # add BF16 autocast
#   CTX="128 2048" RUN_TYPES="forward full" bash scripts/run_memory_profile.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_REL="cs336_systems/benchmarking_script.py"
OUT_DIR="${PROJECT_DIR}/reports/mem"
mkdir -p "$OUT_DIR"

RUNNER=(${RUNNER:-uv run python})

# xl config from Table 1.
D_MODEL=2560
D_FF=10240
N_LAYERS=32
N_HEADS=32

GPU="${GPU:-0}"
WARMUP="${WARMUP:-3}"          # a few warm-up steps so the timeline isn't polluted by lazy init
STEPS="${STEPS:-3}"           # number of recorded steps in the snapshot
BATCH="${BATCH:-4}"           # lower (e.g. 1) to fit full ctx=2048 on a 95 GiB card
CONTEXT_LENGTHS=(${CTX:-128 2048})
RUN_TYPES=(${RUN_TYPES:-forward full})

# "" = fp32, "--mixed_precision" = BF16 autocast
AMP=""; PREC="fp32"
[ "${MIXED:-0}" = "1" ] && { AMP="--mixed_precision"; PREC="bf16"; }

# Reduce allocator fragmentation (helps near the memory ceiling).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$PROJECT_DIR"
echo "Project dir : $PROJECT_DIR"
echo "Output dir  : $OUT_DIR"
echo "Runner      : ${RUNNER[*]}"
echo "GPU         : $GPU   precision: $PREC   batch: $BATCH"
echo "Context len : ${CONTEXT_LENGTHS[*]}"
echo "Run types   : ${RUN_TYPES[*]}"
echo "Alloc conf  : $PYTORCH_CUDA_ALLOC_CONF"
echo

for RT in "${RUN_TYPES[@]}"; do
  for L in "${CONTEXT_LENGTHS[@]}"; do
    TAG="xl_${RT}_ctx${L}_${PREC}"
    SNAP="${OUT_DIR}/mem_${RT}_ctx${L}_${PREC}.pickle"
    echo ">>> [GPU $GPU] run_type=$RT ctx=$L prec=$PREC -> $SNAP"
    CUDA_VISIBLE_DEVICES="$GPU" "${RUNNER[@]}" "$SCRIPT_REL" \
      --run_type "$RT" \
      --d_model "$D_MODEL" --d_ff "$D_FF" \
      --num_layers "$N_LAYERS" --num_heads "$N_HEADS" \
      --context_length "$L" --batch_size "$BATCH" \
      --warmup "$WARMUP" --steps "$STEPS" \
      --memory_profile --memory_snapshot "$SNAP" \
      $AMP 2>&1 | tee "${OUT_DIR}/${TAG}.log" \
      && echo "    [OK]   $TAG" \
      || echo "    [FAIL] $TAG (likely OOM, see ${OUT_DIR}/${TAG}.log)"
    echo
  done
done

echo "Done. Snapshots in $OUT_DIR"
echo "Open https://pytorch.org/memory_viz and drag-drop each .pickle,"
echo "then screenshot the 'Active memory timeline' for the deliverable."
