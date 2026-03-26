#!/usr/bin/env bash
set -euo pipefail

# Prepare Stage 1 metadata files: classes.json and a fixed class_to_archetype.json.
# Usage:
#   bash scripts/server/prepare_stage1_metadata.sh <classes_py_or_json> <class_archetype_json> [class_var_name]

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/prepare_stage1_metadata.sh <classes_py_or_json> <class_archetype_json> [class_var_name]"
  exit 1
fi

CLASSES_SOURCE="$1"
ARCHETYPE_SOURCE="$2"
CLASS_VAR_NAME="${3:-}"
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
PREP_DIR="runs/prep/manual/${TIMESTAMP}"
CLASSES_JSON="$PREP_DIR/classes.json"
ARCHETYPE_JSON="$PREP_DIR/class_to_archetype.json"

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

cp "$ARCHETYPE_SOURCE" "$ARCHETYPE_JSON"

echo "[OK] Metadata prepared using fixed taxonomy/mapping files."
echo "[INFO] classes_json:         $CLASSES_JSON"
echo "[INFO] class_archetype_json: $ARCHETYPE_JSON"
