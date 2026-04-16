#!/usr/bin/env bash
set -euo pipefail

# Run SD v1.5 full fine-tuning via official diffusers trainer.
# Usage:
#   bash scripts/server/stage2/run_sd15_stage2_official.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [extra args...]
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   STAGE2_NUM_PROCESSES=2
#   DIFFUSERS_REPO_ROOT=./diffusers

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/stage2/run_sd15_stage2_official.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [extra args...]"
  exit 1
fi

DATASET_ROOT="$1"
RENDER_INPUT="$2"
shift 2
BATCH_SIZE="${1:-8}"
if [[ $# -gt 0 ]]; then shift; fi
EPOCHS="${1:-9}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

BACKBONE_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"

# Derive dataset label
BASE_NAME="$(basename "$DATASET_ROOT")"
PARENT_NAME="$(basename "$(dirname "$DATASET_ROOT")")"
case "$BASE_NAME" in
  train|val|valid|validation|test|testing)
    DATASET_LABEL="${PARENT_NAME}_${BASE_NAME}" ;;
  *)
    DATASET_LABEL="$BASE_NAME" ;;
esac

BACKBONE_SLUG="$(echo "$BACKBONE_NAME" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}/${TIMESTAMP}"

CMD=(
  cspd-stage2 train
  --dataset-root "$DATASET_ROOT"
  --render-input "$RENDER_INPUT"
  --output-dir "$OUTPUT_DIR"
  --backbone-name "$BACKBONE_NAME"
  --training-parameterization full
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --resolution 512
  --sdxl-mixed-precision fp16
  --sdxl-report-to none
)

STAGE2_NUM_PROCESSES="${STAGE2_NUM_PROCESSES:-2}"
CMD+=(--sdxl-num-processes "$STAGE2_NUM_PROCESSES")

if [[ -n "${STAGE2_ACCELERATE_EXTRA_ARGS:-}" ]]; then
  for value in ${STAGE2_ACCELERATE_EXTRA_ARGS}; do
    CMD+=(--sdxl-accelerate-extra-arg "$value")
  done
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] dataset_root:    $DATASET_ROOT"
echo "[INFO] render_input:    $RENDER_INPUT"
echo "[INFO] output_dir:      $OUTPUT_DIR"
echo "[INFO] backbone:        $BACKBONE_NAME"
echo "[INFO] batch_size:      $BATCH_SIZE"
echo "[INFO] epochs:          $EPOCHS"
echo "[INFO] launch:          ${CMD[*]}"

"${CMD[@]}"

echo "[INFO] SD v1.5 Stage 2 training complete."
echo "[INFO] output: $OUTPUT_DIR"
