#!/usr/bin/env bash
set -euo pipefail

# Prepare Stage 1 metadata files: classes.json and class_to_archetype.json.
# Usage:
#   bash scripts/server/prepare_stage1_metadata.sh <classes_py_or_json> [class_var_name] [archetype_mode=heuristic|vlm]

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/prepare_stage1_metadata.sh <classes_py_or_json> [class_var_name] [archetype_mode=heuristic|vlm]"
  exit 1
fi

CLASSES_SOURCE="$1"
CLASS_VAR_NAME="${2:-}"
ARCHETYPE_MODE="${3:-heuristic}"
ENV_NAME="cspd-dd"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
TORCH_DTYPE="float16"
DEVICE_MAP="auto"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
PREP_DIR="runs/prep/manual/${TIMESTAMP}"
CLASSES_JSON="$PREP_DIR/classes.json"
ARCHETYPE_JSON="$PREP_DIR/class_to_archetype.json"
ARCHETYPE_DETAIL_JSONL="$PREP_DIR/class_to_archetype_details.jsonl"

mkdir -p "$PREP_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

if [[ "$CLASSES_SOURCE" == *.json ]]; then
  cp "$CLASSES_SOURCE" "$CLASSES_JSON"
else
  CMD=(python scripts/data/convert_class_py_to_json.py --input "$CLASSES_SOURCE" --output "$CLASSES_JSON")
  if [[ -n "$CLASS_VAR_NAME" ]]; then
    CMD+=(--var-name "$CLASS_VAR_NAME")
  fi
  "${CMD[@]}"
fi

if [[ "$ARCHETYPE_MODE" == "heuristic" ]]; then
  python scripts/data/generate_class_to_archetype_map.py \
    --input "$CLASSES_JSON" \
    --output "$ARCHETYPE_JSON"
elif [[ "$ARCHETYPE_MODE" == "vlm" ]]; then
  python scripts/data/generate_class_to_archetype_map_vlm.py \
    --input "$CLASSES_JSON" \
    --output "$ARCHETYPE_JSON" \
    --detail-output "$ARCHETYPE_DETAIL_JSONL" \
    --model-name "$MODEL_NAME" \
    --torch-dtype "$TORCH_DTYPE" \
    --device-map "$DEVICE_MAP"
else
  echo "Unsupported archetype mode: $ARCHETYPE_MODE"
  exit 1
fi

echo "[OK] Metadata prepared."
echo "[INFO] classes_json:         $CLASSES_JSON"
echo "[INFO] class_archetype_json: $ARCHETYPE_JSON"
if [[ -f "$ARCHETYPE_DETAIL_JSONL" ]]; then
  echo "[INFO] archetype_detail:     $ARCHETYPE_DETAIL_JSONL"
fi
