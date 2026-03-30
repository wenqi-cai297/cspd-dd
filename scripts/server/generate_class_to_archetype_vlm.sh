#!/usr/bin/env bash
set -euo pipefail

# Generate class_to_archetype.json with multimodal class evidence:
# class text + sampled class images.
# Usage:
#   bash scripts/server/generate_class_to_archetype_vlm.sh <dataset_root> [images_per_class] [taxonomy_json]
#
# By default this script uses the repo-bundled classes.json at the repo root.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/server/generate_class_to_archetype_vlm.sh <dataset_root> [images_per_class] [taxonomy_json]"
  exit 1
fi

DATASET_ROOT="$1"
IMAGES_PER_CLASS="${2:-5}"
TAXONOMY_JSON="${3:-configs/stage1/archetype_taxonomy_manual.json}"
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CLASSES_JSON="$REPO_ROOT/classes.json"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
PREP_DIR="runs/prep/multimodal/${TIMESTAMP}"
OUTPUT_JSON="$PREP_DIR/class_to_archetype.json"
DETAIL_JSONL="$PREP_DIR/class_to_archetype_details.jsonl"
CLASSES_COPY="$PREP_DIR/classes.json"

mkdir -p "$PREP_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"
cp "$CLASSES_JSON" "$CLASSES_COPY"

python scripts/data/generate_class_to_archetype_map_vlm.py \
  --input "$CLASSES_COPY" \
  --dataset-root "$DATASET_ROOT" \
  --output "$OUTPUT_JSON" \
  --detail-output "$DETAIL_JSONL" \
  --taxonomy "$TAXONOMY_JSON" \
  --images-per-class "$IMAGES_PER_CLASS"

echo "[OK] Multimodal class_to_archetype mapping generated."
echo "[INFO] dataset_root:        $DATASET_ROOT"
echo "[INFO] classes_json:        $CLASSES_COPY"
echo "[INFO] class_archetype:     $OUTPUT_JSON"
echo "[INFO] detail_jsonl:        $DETAIL_JSONL"
