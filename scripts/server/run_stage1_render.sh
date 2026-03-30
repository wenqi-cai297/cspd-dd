#!/usr/bin/env bash
set -euo pipefail

# Run Stage 1 canonical semantic rendering from normalized Stage 1 artifacts.
# Usage:
#   bash scripts/server/run_stage1_render.sh /path/to/attributes_normalized.jsonl [renderer_version]
# Example:
#   bash scripts/server/run_stage1_render.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111/normalization/2026-03-28_180021/attributes_normalized.jsonl
#
# The output directory is generated automatically as:
#   runs/stage1/render/<dataset_name>/<backend>/<timestamp>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage1_render.sh <normalized_attributes_jsonl> [renderer_version]"
  exit 1
fi

INPUT_PATH="$1"
RENDERER_VERSION="${2:-v1}"
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[ERROR] Input file not found: $INPUT_PATH"
  exit 1
fi

INPUT_DIR="$(dirname "$INPUT_PATH")"
PARENT_DIR="$(basename "$INPUT_DIR")"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"

# Supported inputs:
# 1) runs/stage1/attributes/<dataset>/<backend>/<run_ts>/normalization/<norm_ts>/attributes_normalized.jsonl
# 2) fallback generic path handling
if [[ "$PARENT_DIR" =~ ^20[0-9]{2}-[0-9]{2}-[0-9]{2}_[0-9]{6}$ ]] && [[ "$(basename "$(dirname "$INPUT_DIR")")" == "normalization" ]]; then
  ATTRIBUTE_RUN_DIR="$(dirname "$(dirname "$INPUT_DIR")")"
  BACKEND_NAME="$(basename "$(dirname "$ATTRIBUTE_RUN_DIR")")"
  DATASET_NAME="$(basename "$(dirname "$(dirname "$ATTRIBUTE_RUN_DIR")")")"
elif [[ "$PARENT_DIR" == normalized* ]]; then
  BACKEND_NAME="$(basename "$(dirname "$(dirname "$INPUT_DIR")")")"
  DATASET_NAME="$(basename "$(dirname "$(dirname "$(dirname "$INPUT_DIR")")")")"
else
  DATASET_NAME="$PARENT_DIR"
  BACKEND_NAME="stage1_render"
fi

OUTPUT_DIR="runs/stage1/render/${DATASET_NAME}/${BACKEND_NAME}/${TIMESTAMP}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] stage1_render_input:  $INPUT_PATH"
echo "[INFO] dataset_name:         $DATASET_NAME"
echo "[INFO] backend_name:         $BACKEND_NAME"
echo "[INFO] stage1_render_dir:    $OUTPUT_DIR"
echo "[INFO] renderer_version:     $RENDERER_VERSION"

CMD=(
  cspd-stage1 render
  --input "$INPUT_PATH"
  --output-dir "$OUTPUT_DIR"
  --renderer-version "$RENDERER_VERSION"
)

"${CMD[@]}"

echo "[INFO] Stage 1 canonical render complete."
echo "[INFO] records:  $OUTPUT_DIR/records.jsonl"
echo "[INFO] failures: $OUTPUT_DIR/failures.jsonl"
echo "[INFO] summary:  $OUTPUT_DIR/render_summary.json"
