#!/usr/bin/env bash
set -euo pipefail

# Run constrained VLM review over normalization-review rows only.
# Usage:
#   bash scripts/server/run_stage1_normalization_review_vlm.sh <attributes_normalized.jsonl> [backend] [max_new_tokens]
# Example:
#   bash scripts/server/run_stage1_normalization_review_vlm.sh runs/stage1/attributes/.../normalization/.../attributes_normalized.jsonl qwen_local 256

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage1_normalization_review_vlm.sh <attributes_normalized.jsonl> [backend] [max_new_tokens]"
  exit 1
fi

INPUT_PATH="$1"
BACKEND="${2:-mock}"
MAX_NEW_TOKENS="${3:-256}"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[ERROR] Input file does not exist: $INPUT_PATH"
  exit 1
fi

INPUT_DIR="$(cd "$(dirname "$INPUT_PATH")" && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="$INPUT_DIR/review_vlm/$TIMESTAMP"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] normalized_input: $INPUT_PATH"
echo "[INFO] review_backend:   $BACKEND"
echo "[INFO] review_output:    $OUTPUT_DIR"

python scripts/data/review_normalization_with_vlm.py \
  --input "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --backend "$BACKEND" \
  --max-new-tokens "$MAX_NEW_TOKENS"

echo "[INFO] Stage 1 normalization review complete."
echo "[INFO] review_jsonl: $OUTPUT_DIR/normalization_review_vlm.jsonl"
echo "[INFO] summary:     $OUTPUT_DIR/normalization_review_vlm_summary.json"
