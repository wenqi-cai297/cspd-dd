#!/usr/bin/env bash
set -euo pipefail

# Run Stage 1 attribute extraction on an ImageFolder-style dataset.
# Usage:
#   bash scripts/server/run_stage1_qwen_local.sh /path/to/dataset [max_new_tokens] [class_name_map] [flush_every] [class_archetype_map]
# Example:
#   bash scripts/server/run_stage1_qwen_local.sh /data/cifar10_small 256
#   bash scripts/server/run_stage1_qwen_local.sh /data/imagenette 256 /data/imagenette/classes.json 10 /data/imagenette/class_to_archetype.json
#
# The output directory is generated automatically as:
#   runs/stage1/attributes/<dataset_name>/qwen_local/<timestamp>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/run_stage1_qwen_local.sh <dataset_root> [max_new_tokens] [class_name_map] [flush_every] [class_archetype_map]"
  exit 1
fi

DATASET_ROOT="$1"
MAX_NEW_TOKENS="${2:-256}"
CLASS_NAME_MAP="${3:-}"
FLUSH_EVERY="${4:-10}"
CLASS_ARCHETYPE_MAP="${5:-}"
ENV_NAME="cspd-dd"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
TORCH_DTYPE="float16"
DEVICE_MAP="auto"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_BASENAME="$(basename "$DATASET_ROOT")"
DATASET_PARENT_BASENAME="$(basename "$(dirname "$DATASET_ROOT")")"
if [[ "$DATASET_BASENAME" == "train" || "$DATASET_BASENAME" == "val" || "$DATASET_BASENAME" == "test" ]]; then
  DATASET_NAME="$DATASET_PARENT_BASENAME"
else
  DATASET_NAME="$DATASET_BASENAME"
fi

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/attributes/${DATASET_NAME}/qwen_local/${TIMESTAMP}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[INFO] dataset_root:        $DATASET_ROOT"
echo "[INFO] output_dir:          $OUTPUT_DIR"
echo "[INFO] max_tokens:          $MAX_NEW_TOKENS"
echo "[INFO] flush_every:         $FLUSH_EVERY"
if [[ -n "$CLASS_NAME_MAP" ]]; then
  echo "[INFO] class_name_map:      $CLASS_NAME_MAP"
fi
if [[ -n "$CLASS_ARCHETYPE_MAP" ]]; then
  echo "[INFO] class_archetype_map: $CLASS_ARCHETYPE_MAP"
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
  --flush-every "$FLUSH_EVERY"
)

if [[ -n "$CLASS_NAME_MAP" ]]; then
  CMD+=(--class-name-map "$CLASS_NAME_MAP")
fi
if [[ -n "$CLASS_ARCHETYPE_MAP" ]]; then
  CMD+=(--class-archetype-map "$CLASS_ARCHETYPE_MAP")
fi

"${CMD[@]}"
