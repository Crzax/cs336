#!/usr/bin/env bash
# Compare vanilla vs torch.compile'd Transformer (Problem torch_compile (b)).
# For each model size and run_type, runs the benchmark twice: once without
# --compile and once with --compile, so the pairs sit next to each other.
# Usage: bash run_compile_compare.sh [output.log]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_REL="cs336_systems/benchmarking_script.py"
OUT="${1:-${SCRIPT_DIR}/compile_compare_results.log}"

# Run via `uv run` from the project root so deps in pyproject.toml resolve.
# Override with: RUNNER="python" bash run_compile_compare.sh
RUNNER=(${RUNNER:-uv run python})

# torch.compile needs a couple of warmup steps to absorb the (slow) first-step
# compilation, so keep WARMUP >= 3.
WARMUP="${WARMUP:-5}"
STEPS="${STEPS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"

# size  d_model  d_ff   num_layers  num_heads
CONFIGS=(
  "small   768   3072   12  12"
  "medium  1024  4096   24  16"
  "large   1280  5120   36  20"
  "xl      2560  10240  32  32"
)

# (b) asks specifically about the forward pass and the full step
# (forward + backward + optimizer). forward_backward is included for context.
RUN_TYPES=(forward full)

# "" = vanilla, "--compile" = torch.compile
COMPILE_FLAGS=("" "--compile")

: > "$OUT"
echo "Writing results to $OUT"
echo "Project dir: $PROJECT_DIR"
echo "Runner: ${RUNNER[*]}"
echo "warmup=${WARMUP} steps=${STEPS} batch=${BATCH_SIZE} ctx=${CONTEXT_LENGTH}"

cd "$PROJECT_DIR"

for cfg in "${CONFIGS[@]}"; do
  read -r SIZE D_MODEL D_FF N_LAYERS N_HEADS <<< "$cfg"
  for RT in "${RUN_TYPES[@]}"; do
    for CF in "${COMPILE_FLAGS[@]}"; do
      LABEL="vanilla"; [ -n "$CF" ] && LABEL="compiled"
      echo "==================================================================" | tee -a "$OUT"
      echo ">>> size=${SIZE}  run_type=${RT}  mode=${LABEL}" | tee -a "$OUT"
      echo "    d_model=${D_MODEL} d_ff=${D_FF} num_layers=${N_LAYERS} num_heads=${N_HEADS}" \
        | tee -a "$OUT"

      # Don't kill the whole sweep if one config OOMs / fails to compile.
      if ! "${RUNNER[@]}" "$SCRIPT_REL" \
          --warmup "$WARMUP" \
          --steps "$STEPS" \
          --run_type "$RT" \
          --batch_size "$BATCH_SIZE" \
          --context_length "$CONTEXT_LENGTH" \
          --d_model "$D_MODEL" \
          --d_ff "$D_FF" \
          --num_layers "$N_LAYERS" \
          --num_heads "$N_HEADS" \
          $CF 2>&1 | tee -a "$OUT"; then
        echo "    [FAILED]  size=${SIZE} run_type=${RT} mode=${LABEL} (non-zero exit)" | tee -a "$OUT"
      fi
    done
  done
done

echo "Done. See $OUT"
