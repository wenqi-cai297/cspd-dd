#!/usr/bin/env bash
set -euo pipefail

# Run Stage 2 canonical semantic rendering from normalized Stage 1 artifacts.
# Usage:
#   bash scripts/server/run_stage2_render.sh /path/to/attributes_normalized.jsonl [renderer_version]
# Example:
#   bash scripts/server/run_stage2_render.sh runs/stage1/attributes/ImageNette/qwen_local/2026-03-26_183111/normalized_v2/attributes_normalized.jsonl
#
# The output directory is generated automatically as:
#   runs/stage2/<dataset_name>/<backend>/<timestamp>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage2_render.sh <normalized_attributes_jsonl> [renderer_version]"
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
INPUT_PARENT="$(basename "$INPUT_DIR")"
BACKEND_CANDIDATE="$(basename "$(dirname "$(dirname "$INPUT_DIR")")")"
DATASET_CANDIDATE="$(basename "$(dirname "$(dirname "$(dirname "$INPUT_DIR")")")")"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"

if [[ "$INPUT_PARENT" == normalized* && "$BACKEND_CANDIDATE" != runs && -n "$DATASET_CANDIDATE" ]]; then
  DATASET_NAME="$DATASET_CANDIDATE"
  BACKEND_NAME="$BACKEND_CANDIDATE"
else
  DATASET_NAME="$INPUT_PARENT"
  BACKEND_NAME="stage2_render"
fi

OUTPUT_DIR="runs/stage2/${DATASET_NAME}/${BACKEND_NAME}/${TIMESTAMP}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] stage2_input:       $INPUT_PATH"
echo "[INFO] dataset_name:       $DATASET_NAME"
echo "[INFO] backend_name:       $BACKEND_NAME"
echo "[INFO] stage2_output_dir:  $OUTPUT_DIR"
echo "[INFO] renderer_version:   $RENDERER_VERSION"

CMD=(
  cspd-stage2 render
  --input "$INPUT_PATH"
  --output-dir "$OUTPUT_DIR"
  --renderer-version "$RENDERER_VERSION"
)

"${CMD[@]}"

echo "[INFO] Stage 2 render complete."
echo "[INFO] records:  $OUTPUT_DIR/records.jsonl"
echo "[INFO] failures: $OUTPUT_DIR/failures.jsonl"
echo "[INFO] summary:  $OUTPUT_DIR/render_summary.json"
