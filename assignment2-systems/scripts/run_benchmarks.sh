#!/usr/bin/env bash
# Benchmark forward / forward+backward / full step across the model sizes
# defined in Section 2.1.2.
# Usage: bash run_benchmarks.sh [output.log]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_REL="cs336_systems/benchmarking_script.py"
OUT="${1:-${SCRIPT_DIR}/benchmark_results.log}"

# Run via `uv run` from the project root so deps in pyproject.toml resolve.
# Override with: RUNNER="python" bash run_benchmarks.sh   (if you don't use uv)
RUNNER=(${RUNNER:-uv run python})

WARMUP=2
STEPS=10

# size  d_model  d_ff   num_layers  num_heads
CONFIGS=(
  "small   768   3072   12  12"
  "medium  1024  4096   24  16"
  "large   1280  5120   36  20"
  "xl      2560  10240  32  32"
  "10B     4608  12288  50  36"
)

RUN_TYPES=(forward forward_backward full)

: > "$OUT"
echo "Writing results to $OUT"
echo "Project dir: $PROJECT_DIR"
echo "Runner: ${RUNNER[*]}"

cd "$PROJECT_DIR"

for cfg in "${CONFIGS[@]}"; do
  read -r SIZE D_MODEL D_FF N_LAYERS N_HEADS <<< "$cfg"
  for RT in "${RUN_TYPES[@]}"; do
    echo "==================================================================" | tee -a "$OUT"
    echo ">>> size=${SIZE}  run_type=${RT}" | tee -a "$OUT"
    echo "    d_model=${D_MODEL} d_ff=${D_FF} num_layers=${N_LAYERS} num_heads=${N_HEADS}" \
      | tee -a "$OUT"

    # Don't kill the whole sweep if OOM hits on xl/10B.
    if ! "${RUNNER[@]}" "$SCRIPT_REL" \
        --warmup "$WARMUP" \
        --steps "$STEPS" \
        --run_type "$RT" \
        --d_model "$D_MODEL" \
        --d_ff "$D_FF" \
        --num_layers "$N_LAYERS" \
        --num_heads "$N_HEADS" 2>&1 | tee -a "$OUT"; then
      echo "    [FAILED]  size=${SIZE} run_type=${RT} (non-zero exit, see traceback above)" | tee -a "$OUT"
    fi
  done
done

echo "Done. See $OUT"
