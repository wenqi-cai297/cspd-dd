#!/usr/bin/env bash
set -euo pipefail

# Run Stage 1 normalization in-place relative to an attribute run directory.
# Usage:
#   bash scripts/server/run_stage1_normalization.sh <attribute_run_dir_or_attributes_jsonl> [review_backend]
#   bash scripts/server/run_stage1_normalization.sh <attribute_run_dir_or_attributes_jsonl> --disable-vlm-review
# Example:
#   bash scripts/server/run_stage1_normalization.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111
#   bash scripts/server/run_stage1_normalization.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111/attributes.jsonl qwen_local
#   bash scripts/server/run_stage1_normalization.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111 --disable-vlm-review
#
# The output directory is generated automatically as:
#   <attribute_run_dir>/normalization/<timestamp>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage1_normalization.sh <attribute_run_dir_or_attributes_jsonl> [review_backend]"
  echo "   or: bash scripts/server/run_stage1_normalization.sh <attribute_run_dir_or_attributes_jsonl> --disable-vlm-review"
  exit 1
fi

INPUT_ARG="$1"
REVIEW_BACKEND="qwen_local"
DISABLE_VLM_REVIEW=0
if [[ $# -ge 2 ]]; then
  if [[ "$2" == "--disable-vlm-review" ]]; then
    DISABLE_VLM_REVIEW=1
  else
    REVIEW_BACKEND="$2"
  fi
fi
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -d "$INPUT_ARG" ]]; then
  ATTRIBUTES_PATH="$INPUT_ARG/attributes.jsonl"
  RUN_DIR="$INPUT_ARG"
elif [[ -f "$INPUT_ARG" ]]; then
  ATTRIBUTES_PATH="$INPUT_ARG"
  RUN_DIR="$(dirname "$INPUT_ARG")"
else
  echo "[ERROR] Input path not found: $INPUT_ARG"
  exit 1
fi

if [[ ! -f "$ATTRIBUTES_PATH" ]]; then
  echo "[ERROR] attributes.jsonl not found: $ATTRIBUTES_PATH"
  exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="$RUN_DIR/normalization/$TIMESTAMP"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] stage1_attributes:   $ATTRIBUTES_PATH"
echo "[INFO] normalization_dir:  $OUTPUT_DIR"
if [[ "$DISABLE_VLM_REVIEW" -eq 1 ]]; then
  echo "[INFO] inline_vlm_review: disabled"
else
  echo "[INFO] inline_vlm_review: enabled ($REVIEW_BACKEND)"
fi

CMD=(
  python scripts/data/normalize_stage1_attributes.py
  --input "$ATTRIBUTES_PATH"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$DISABLE_VLM_REVIEW" -eq 1 ]]; then
  CMD+=(--disable-vlm-review)
else
  CMD+=(--review-backend "$REVIEW_BACKEND")
fi

"${CMD[@]}"

echo "[INFO] Stage 1 normalization complete."
echo "[INFO] normalized:   $OUTPUT_DIR/attributes_normalized.jsonl"
echo "[INFO] audit:        $OUTPUT_DIR/normalization_audit.jsonl"
echo "[INFO] review:       $OUTPUT_DIR/normalization_review_queue.jsonl"
echo "[INFO] review_vlm:   $OUTPUT_DIR/normalization_review_vlm.jsonl"
echo "[INFO] review_sum:   $OUTPUT_DIR/normalization_review_vlm_summary.json"
echo "[INFO] summary:      $OUTPUT_DIR/normalization_summary.json"
