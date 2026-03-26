#!/usr/bin/env bash
set -euo pipefail

# Full Stage 1 shell workflow from environment checks to final attribute extraction.
# Usage:
#   bash scripts/server/run_stage1_full_workflow.sh \
#     [--skip-smoke] \
#     <dataset_root> \
#     <classes_py_or_json> \
#     <class_archetype_json> \
#     [class_var_name] \
#     [max_new_tokens=256] \
#     [sample_image_for_single_test]

SKIP_SMOKE=0
if [[ "${1:-}" == "--skip-smoke" ]]; then
  SKIP_SMOKE=1
  shift
fi

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/server/run_stage1_full_workflow.sh [--skip-smoke] <dataset_root> <classes_py_or_json> <class_archetype_json> [class_var_name] [max_new_tokens=256] [sample_image_for_single_test]"
  exit 1
fi

DATASET_ROOT="$1"
CLASSES_SOURCE="$2"
ARCHETYPE_SOURCE="$3"
CLASS_VAR_NAME="${4:-}"
MAX_NEW_TOKENS="${5:-256}"
SAMPLE_IMAGE="${6:-}"
ENV_NAME="cspd-dd"
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
TORCH_DTYPE="float16"
DEVICE_MAP="auto"
FLUSH_EVERY="10"
SMOKE_NUM_CLASSES="3"
SMOKE_NUM_IMAGES_PER_CLASS="10"

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
PREP_DIR="runs/prep/${DATASET_NAME}/${TIMESTAMP}"
TEST_DIR="runs/tests/${DATASET_NAME}/${TIMESTAMP}"
ATTR_DIR="runs/attributes/${DATASET_NAME}/qwen_local/${TIMESTAMP}"
CLASSES_JSON="$PREP_DIR/classes.json"
ARCHETYPE_JSON="$PREP_DIR/class_to_archetype.json"
SMOKE_DATASET_DIR="$TEST_DIR/mock_subset"
SMOKE_CLASSES_JSON="$TEST_DIR/mock_subset_classes.json"
SMOKE_ARCHETYPE_JSON="$TEST_DIR/mock_subset_class_to_archetype.json"

mkdir -p "$PREP_DIR" "$TEST_DIR" "$ATTR_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

echo "[STEP 1/7] Environment checks"
python --version
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "from PIL import Image; print('PIL ok')"
pip install -e .

echo "[STEP 2/7] Prepare classes.json"
if [[ "$CLASSES_SOURCE" == *.json ]]; then
  cp "$CLASSES_SOURCE" "$CLASSES_JSON"
else
  CMD=(python scripts/data/convert_class_py_to_json.py --input "$CLASSES_SOURCE" --output "$CLASSES_JSON")
  if [[ -n "$CLASS_VAR_NAME" ]]; then
    CMD+=(--var-name "$CLASS_VAR_NAME")
  fi
  "${CMD[@]}"
fi

echo "[STEP 3/7] Use fixed class_to_archetype.json"
cp "$ARCHETYPE_SOURCE" "$ARCHETYPE_JSON"

echo "[STEP 4/7] Qwen load test"
python scripts/vlm/test_qwen_vl_load.py | tee "$TEST_DIR/qwen_load_test.log"

echo "[STEP 5/7] Single-image inference test"
if [[ -z "$SAMPLE_IMAGE" ]]; then
  SAMPLE_IMAGE="$(find "$DATASET_ROOT" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.webp' \) -print -quit)"
fi
if [[ -z "$SAMPLE_IMAGE" ]]; then
  echo "Could not find a sample image under dataset root: $DATASET_ROOT"
  exit 1
fi
python scripts/vlm/test_single_image_infer.py \
  --image "$SAMPLE_IMAGE" \
  --class-name unknown \
  --class-id -1 \
  --max-new-tokens "$MAX_NEW_TOKENS" | tee "$TEST_DIR/single_image_test.log"

echo "[STEP 6/7] Mock smoke run"
if [[ "$SKIP_SMOKE" == "1" ]]; then
  echo "[INFO] --skip-smoke enabled; skipping mock smoke run"
else
  rm -rf "$SMOKE_DATASET_DIR"
  mkdir -p "$SMOKE_DATASET_DIR"
  python - <<'PY' "$DATASET_ROOT" "$CLASSES_JSON" "$ARCHETYPE_JSON" "$SMOKE_DATASET_DIR" "$SMOKE_CLASSES_JSON" "$SMOKE_ARCHETYPE_JSON" "$SMOKE_NUM_CLASSES" "$SMOKE_NUM_IMAGES_PER_CLASS"
import json
import shutil
import sys
from pathlib import Path

image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jpeg', '.JPEG', '.JPG', '.PNG', '.BMP', '.WEBP'}

dataset_root = Path(sys.argv[1])
classes_json_path = Path(sys.argv[2])
archetype_json_path = Path(sys.argv[3])
smoke_dataset_dir = Path(sys.argv[4])
smoke_classes_json = Path(sys.argv[5])
smoke_archetype_json = Path(sys.argv[6])
num_classes = int(sys.argv[7])
num_images = int(sys.argv[8])

class_name_map = json.loads(classes_json_path.read_text(encoding='utf-8-sig'))
class_archetype_map = json.loads(archetype_json_path.read_text(encoding='utf-8-sig'))
selected_classes = {}
selected_archetypes = {}

class_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=lambda p: p.name)[:num_classes]
for class_dir in class_dirs:
    target_dir = smoke_dataset_dir / class_dir.name
    target_dir.mkdir(parents=True, exist_ok=True)
    images = [p for p in sorted(class_dir.rglob('*')) if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    for image_path in images[:num_images]:
        shutil.copy2(image_path, target_dir / image_path.name)
    selected_classes[class_dir.name] = class_name_map.get(class_dir.name, class_dir.name)
    if class_dir.name in class_archetype_map:
        selected_archetypes[class_dir.name] = class_archetype_map[class_dir.name]

smoke_classes_json.write_text(json.dumps(selected_classes, ensure_ascii=False, indent=2), encoding='utf-8')
smoke_archetype_json.write_text(json.dumps(selected_archetypes, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[INFO] Built smoke subset with {len(class_dirs)} classes at {smoke_dataset_dir}")
PY

  cspd-stage1 run \
    --dataset-root "$SMOKE_DATASET_DIR" \
    --output-dir "$TEST_DIR/mock_smoke_run" \
    --backend mock \
    --class-name-map "$SMOKE_CLASSES_JSON" \
    --class-archetype-map "$SMOKE_ARCHETYPE_JSON" \
    --flush-every 1 | tee "$TEST_DIR/mock_smoke_run.log"
fi

echo "[STEP 7/7] Qwen local attribute extraction run"
cspd-stage1 run \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$ATTR_DIR" \
  --backend qwen_local \
  --model-name "$MODEL_NAME" \
  --torch-dtype "$TORCH_DTYPE" \
  --device-map "$DEVICE_MAP" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --class-name-map "$CLASSES_JSON" \
  --class-archetype-map "$ARCHETYPE_JSON" \
  --flush-every "$FLUSH_EVERY"

echo

echo "[OK] Full Stage 1 workflow finished."
echo "[INFO] classes_json:          $CLASSES_JSON"
echo "[INFO] class_archetype_json:  $ARCHETYPE_JSON"
echo "[INFO] test_dir:              $TEST_DIR"
echo "[INFO] final_attribute_dir:   $ATTR_DIR"
