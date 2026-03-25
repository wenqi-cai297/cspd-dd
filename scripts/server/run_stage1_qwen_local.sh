#!/usr/bin/env bash
set -euo pipefail

# Run Stage 1 attribute extraction on an ImageFolder-style dataset.
# Usage:
#   bash scripts/server/run_stage1_qwen_local.sh /path/to/dataset [max_new_tokens]
# Example:
#   bash scripts/server/run_stage1_qwen_local.sh /data/cifar10_small 256
#
# The output directory is generated automatically as:
#   runs/attributes/<dataset_name>/qwen_local/<timestamp>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage1_qwen_local.sh <dataset_root> [max_new_tokens]"
  exit 1
fi

DATASET_ROOT="$1"
MAX_NEW_TOKENS="${2:-256}"
ENV_NAME="cspd_vlm"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
TORCH_DTYPE="float16"
DEVICE_MAP="auto"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_NAME="$(basename "$DATASET_ROOT")"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/attributes/${DATASET_NAME}/qwen_local/${TIMESTAMP}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"

mkdir -p "$OUTPUT_DIR"

echo "[INFO] dataset_root:   $DATASET_ROOT"
echo "[INFO] output_dir:     $OUTPUT_DIR"
echo "[INFO] max_tokens:     $MAX_NEW_TOKENS"
if [[ -n "$CLASS_NAME_MAP" ]]; then
  echo "[INFO] class_name_map: $CLASS_NAME_MAP"
fi

CMD=(
  cspd-stage1 run
  --dataset-root "$DATASET_ROOT"
  --output-dir "$OUTPUT_DIR"
  --backend qwen_local
  --model-name "$MODEL_NAME"
  --torch-dtype "$TORCH_DTYPE"
  --device-map "$DEVICE_MAP"
  --max-new-tokens "$MAX_NEW_TOKENS"
)

if [[ -n "$CLASS_NAME_MAP" ]]; then
  CMD+=(--class-name-map "$CLASS_NAME_MAP")
fi

"${CMD[@]}"
