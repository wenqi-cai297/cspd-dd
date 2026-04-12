#!/usr/bin/env bash
set -euo pipefail

# Run the full Stage 1 pipeline: extraction → normalization → render.
#
# Usage:
#   bash scripts/server/stage1/run_stage1_pipeline.sh <dataset_root> [backend]
#
# Examples:
#   # ImageNette with real VLM
#   bash scripts/server/stage1/run_stage1_pipeline.sh /media/4T_HDD/cai/datasets/ImageNette/train qwen_local
#
#   # ImageNet-1k 5-shot
#   bash scripts/server/stage1/run_stage1_pipeline.sh /media/4T_HDD/cai/datasets/ImageNet1k_5shot/train qwen_local
#
#   # Quick plumbing test with mock backend
#   bash scripts/server/stage1/run_stage1_pipeline.sh /media/4T_HDD/cai/datasets/ImageNette/train mock
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd           # conda env override
#   STAGE1_FLUSH_EVERY=50           # incremental flush interval
#   STAGE1_DISABLE_VLM_REVIEW=0     # set to 1 to skip VLM review in normalization
#   STAGE1_MAX_RETRIES=2            # extraction retry count

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/stage1/run_stage1_pipeline.sh <dataset_root> [backend]"
  exit 1
fi

DATASET_ROOT="$1"
BACKEND="${2:-qwen_local}"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
FLUSH_EVERY="${STAGE1_FLUSH_EVERY:-50}"
MAX_RETRIES="${STAGE1_MAX_RETRIES:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"
pip install -e . -q

# Derive dataset label for output paths
DATASET_LABEL="$(basename "$(dirname "$DATASET_ROOT")")"
SPLIT="$(basename "$DATASET_ROOT")"
case "$SPLIT" in
  train|val|valid|validation|test|testing)
    DATASET_LABEL="${DATASET_LABEL}_${SPLIT}"
    ;;
  *)
    DATASET_LABEL="$SPLIT"
    ;;
esac

# --- Stage 1A: Extraction ---
ATTR_TS="$(date +%Y-%m-%d_%H%M%S)"
ATTR_DIR="runs/stage1/attributes/${DATASET_LABEL}/${BACKEND}/${ATTR_TS}"

echo "============================================================"
echo "[Stage 1A] Extraction"
echo "  dataset_root:  $DATASET_ROOT"
echo "  backend:       $BACKEND"
echo "  output_dir:    $ATTR_DIR"
echo "============================================================"

EXTRACT_CMD=(
  cspd-stage1 run
  --dataset-root "$DATASET_ROOT"
  --output-dir "$ATTR_DIR"
  --backend "$BACKEND"
  --class-name-map classes.json
  --class-archetype-map configs/stage1/class_to_archetype_imagenet1k_manual.json
  --flush-every "$FLUSH_EVERY"
  --max-retries "$MAX_RETRIES"
)
"${EXTRACT_CMD[@]}"

echo "[Stage 1A] Extraction complete: $ATTR_DIR"

# --- Stage 1B: Normalization ---
NORM_TS="$(date +%Y-%m-%d_%H%M%S)"
NORM_DIR="${ATTR_DIR}/normalization/${NORM_TS}"

echo ""
echo "============================================================"
echo "[Stage 1B] Normalization"
echo "  input:      ${ATTR_DIR}/attributes.jsonl"
echo "  output_dir: $NORM_DIR"
echo "============================================================"

NORM_CMD=(
  cspd-stage1 normalize
  --input "${ATTR_DIR}/attributes.jsonl"
  --output-dir "$NORM_DIR"
)
if [[ "${STAGE1_DISABLE_VLM_REVIEW:-0}" == "1" ]]; then
  NORM_CMD+=(--disable-vlm-review)
fi
"${NORM_CMD[@]}"

echo "[Stage 1B] Normalization complete: $NORM_DIR"

# --- Stage 1C: Render ---
RENDER_TS="$(date +%Y-%m-%d_%H%M%S)"
RENDER_DIR="runs/stage1/render/${DATASET_LABEL}/${BACKEND}/${RENDER_TS}"

echo ""
echo "============================================================"
echo "[Stage 1C] Render"
echo "  input:      ${NORM_DIR}/attributes_normalized.jsonl"
echo "  output_dir: $RENDER_DIR"
echo "============================================================"

cspd-stage1 render \
  --input "${NORM_DIR}/attributes_normalized.jsonl" \
  --output-dir "$RENDER_DIR"

echo "[Stage 1C] Render complete: $RENDER_DIR"

# --- Summary ---
echo ""
echo "============================================================"
echo "[Stage 1] Pipeline complete"
echo "  Attributes:  $ATTR_DIR"
echo "  Normalized:  $NORM_DIR"
echo "  Rendered:    $RENDER_DIR"
echo "  Records:     ${RENDER_DIR}/records.jsonl"
echo "============================================================"
