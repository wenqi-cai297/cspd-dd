#!/usr/bin/env bash
set -euo pipefail

# Run Stage 2 canonical semantic rendering from normalized Stage 1 artifacts.
# Usage:
#   bash scripts/server/run_stage2_render.sh /path/to/attributes_normalized.jsonl [output_dir] [renderer_version]
# Example:
#   bash scripts/server/run_stage2_render.sh runs/stage1/mock_normalized/attributes_normalized.jsonl
#   bash scripts/server/run_stage2_render.sh /data/cspd/runs/stage1/imagenet_train_normalized/attributes_normalized.jsonl runs/stage2/imagenet_train_render

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage2_render.sh <normalized_attributes_jsonl> [output_dir] [renderer_version]"
  exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="${2:-}"
RENDERER_VERSION="${3:-v1}"
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "[ERROR] Input file not found: $INPUT_PATH"
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  INPUT_PARENT="$(basename "$(dirname "$INPUT_PATH")")"
  TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
  OUTPUT_DIR="runs/stage2/${INPUT_PARENT}/${TIMESTAMP}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] stage2_input:       $INPUT_PATH"
echo "[INFO] stage2_output_dir:  $OUTPUT_DIR"
echo "[INFO] renderer_version:   $RENDERER_VERSION"

a=(
  cspd-stage2 render
  --input "$INPUT_PATH"
  --output-dir "$OUTPUT_DIR"
  --renderer-version "$RENDERER_VERSION"
)

"${a[@]}"

echo "[INFO] Stage 2 render complete."
echo "[INFO] records:  $OUTPUT_DIR/records.jsonl"
echo "[INFO] failures: $OUTPUT_DIR/failures.jsonl"
echo "[INFO] summary:  $OUTPUT_DIR/render_summary.json"
