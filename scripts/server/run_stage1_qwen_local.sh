#!/usr/bin/env bash
set -euo pipefail

# Run Stage 1 attribute extraction on an ImageFolder-style dataset.
# Usage:
#   bash scripts/server/run_stage1_qwen_local.sh /path/to/dataset /path/to/output_dir
# Example:
#   bash scripts/server/run_stage1_qwen_local.sh /data/cifar10_small runs/stage1_qwen

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/run_stage1_qwen_local.sh <dataset_root> <output_dir> [max_new_tokens]"
  exit 1
fi

DATASET_ROOT="$1"
OUTPUT_DIR="$2"
MAX_NEW_TOKENS="${3:-256}"
ENV_NAME="cspd_vlm"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
TORCH_DTYPE="float16"
DEVICE_MAP="auto"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"

cspd-stage1 run \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --backend qwen_local \
  --model-name "$MODEL_NAME" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --max-new-tokens "$MAX_NEW_TOKENS"
