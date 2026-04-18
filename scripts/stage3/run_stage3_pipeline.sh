#!/usr/bin/env bash
set -euo pipefail

# Run the full Stage 3 pipeline: encode images/captions → cluster → extract visual/semantic modes.
#
# Usage:
#   bash scripts/stage3/run_stage3_pipeline.sh <dataset_root> <stage1_render_records_jsonl> <ipc>
#
# Examples:
#   # ImageNette, IPC=10
#   bash scripts/stage3/run_stage3_pipeline.sh \
#     /media/4T_HDD/cai/datasets/ImageNette/train \
#     runs/stage1/render/ImageNette_train/qwen_local/2026-04-12_XXXXXX/records.jsonl \
#     10
#
#   # ImageNet-1k 5-shot, IPC=1
#   bash scripts/stage3/run_stage3_pipeline.sh \
#     /media/4T_HDD/cai/datasets/ImageNet1k_5shot/train \
#     runs/stage1/render/ImageNet1k_5shot_train/qwen_local/2026-04-12_XXXXXX/records.jsonl \
#     1
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   STAGE3_BATCH_SIZE=8              # encoding batch size
#   STAGE3_DEVICE=cuda               # torch device
#   STAGE3_DTYPE=float16             # weight dtype
#   STAGE3_SEED=42                   # clustering seed

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/stage3/run_stage3_pipeline.sh <dataset_root> <stage1_render_records_jsonl> <ipc>"
  exit 1
fi

DATASET_ROOT="$1"
RENDER_INPUT="$2"
IPC="$3"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
BATCH_SIZE="${STAGE3_BATCH_SIZE:-8}"
DEVICE="${STAGE3_DEVICE:-cuda}"
DTYPE="${STAGE3_DTYPE:-float16}"
SEED="${STAGE3_SEED:-42}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi
if [[ ! -f "$RENDER_INPUT" ]]; then
  echo "[ERROR] Stage 1 render records not found: $RENDER_INPUT"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

# Derive dataset label
derive_dataset_label() {
  local dataset_root="$1"
  local base_name parent_name
  base_name="$(basename "$dataset_root")"
  parent_name="$(basename "$(dirname "$dataset_root")")"
  case "$base_name" in
    train|val|valid|validation|test|testing)
      if [[ -n "$parent_name" && "$parent_name" != "." && "$parent_name" != "/" ]]; then
        printf '%s_%s\n' "$parent_name" "$base_name"; return
      fi ;;
  esac
  printf '%s\n' "$base_name"
}

DATASET_LABEL="$(derive_dataset_label "$DATASET_ROOT")"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/stage3/${DATASET_LABEL}/ipc${IPC}/${TIMESTAMP}"

echo "============================================================"
echo "[Stage 3] Visual/Semantic Mode Discovery"
echo "  dataset_root:  $DATASET_ROOT"
echo "  render_input:  $RENDER_INPUT"
echo "  output_dir:    $OUTPUT_DIR"
echo "  ipc:           $IPC"
echo "  batch_size:    $BATCH_SIZE"
echo "  device:        $DEVICE"
echo "  dtype:         $DTYPE"
echo "  seed:          $SEED"
echo "============================================================"

cspd-stage3 run \
  --dataset-root "$DATASET_ROOT" \
  --render-input "$RENDER_INPUT" \
  --output-dir "$OUTPUT_DIR" \
  --ipc "$IPC" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --seed "$SEED"

echo ""
echo "============================================================"
echo "[Stage 3] Pipeline complete"
echo "  Output:         $OUTPUT_DIR"
echo "  Visual modes:   ${OUTPUT_DIR}/modes/visual_modes.pt"
echo "  Semantic modes: ${OUTPUT_DIR}/modes/semantic_modes.pt"
echo "  Modes index:    ${OUTPUT_DIR}/modes/modes_index.json"
echo "============================================================"
