#!/usr/bin/env bash
set -euo pipefail

# Run CSPD Stage 2 v1 training helper with accelerate-based launch by default.
# Usage:
#   bash scripts/server/run_stage2_train.sh <dataset_root> <stage1_render_records_jsonl> [backbone_name] [batch_size] [epochs] [extra args...]
# Example:
#   bash scripts/server/run_stage2_train.sh /data/imagenette/train runs/stage1/render/imagenette/qwen_local/2026-04-02_010203/records.jsonl
#
# Output directory:
#   runs/stage2/train/<dataset_label>/<backbone_slug>/<timestamp>
#   - default label is basename(dataset_root)
#   - split-only roots like .../train become <parent>_train
#   - override with STAGE2_DATASET_LABEL=...
# Launch behavior:
#   - default: accelerate launch --num_processes ${STAGE2_NUM_PROCESSES:-all available GPUs}
#   - override with STAGE2_ACCELERATE_EXTRA_ARGS="..."
#   - set STAGE2_DISABLE_ACCELERATE=1 to fall back to direct single-process CLI launch

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/run_stage2_train.sh <dataset_root> <stage1_render_records_jsonl> [backbone_name] [batch_size] [epochs] [extra args...]"
  exit 1
fi

DATASET_ROOT="$1"
RENDER_INPUT="$2"
shift 2

BACKBONE_NAME="black-forest-labs/FLUX.1-Kontext-dev"
BATCH_SIZE="4"
EPOCHS="1"
EXTRA_ARGS=()
ENV_NAME="cspd-dd"
USE_ACCELERATE="${STAGE2_DISABLE_ACCELERATE:-0}"
ACCELERATE_EXTRA_ARGS_STRING="${STAGE2_ACCELERATE_EXTRA_ARGS:-}"

if [[ $# -gt 0 && "$1" != -* ]]; then
  BACKBONE_NAME="$1"
  shift
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  BATCH_SIZE="$1"
  shift
fi

if [[ $# -gt 0 && "$1" != -* ]]; then
  EPOCHS="$1"
  shift
fi

EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi

if [[ ! -f "$RENDER_INPUT" ]]; then
  echo "[ERROR] Stage 1 render input not found: $RENDER_INPUT"
  exit 1
fi

derive_dataset_label() {
  local dataset_root="$1"
  local base_name
  local parent_name

  base_name="$(basename "$dataset_root")"
  parent_name="$(basename "$(dirname "$dataset_root")")"

  case "$base_name" in
    train|val|valid|validation|test|testing)
      if [[ -n "$parent_name" && "$parent_name" != "." && "$parent_name" != "/" ]]; then
        printf '%s_%s\n' "$parent_name" "$base_name"
        return
      fi
      ;;
  esac

  printf '%s\n' "$base_name"
}

DATASET_LABEL="${STAGE2_DATASET_LABEL:-$(derive_dataset_label "$DATASET_ROOT")}"
BACKBONE_SLUG="$(echo "$BACKBONE_NAME" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}/${TIMESTAMP}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

CMD=(
  cspd-stage2 train
  --dataset-root "$DATASET_ROOT"
  --render-input "$RENDER_INPUT"
  --output-dir "$OUTPUT_DIR"
  --backbone-name "$BACKBONE_NAME"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

ACCELERATE_CMD=()
if [[ "$USE_ACCELERATE" != "1" ]]; then
  ACCELERATE_CMD=(accelerate launch)
  if [[ -n "${STAGE2_NUM_PROCESSES:-}" ]]; then
    ACCELERATE_CMD+=(--num_processes "$STAGE2_NUM_PROCESSES")
  fi
  if [[ -n "$ACCELERATE_EXTRA_ARGS_STRING" ]]; then
    # shellcheck disable=SC2206
    ACCELERATE_EXTRA_ARGS=($ACCELERATE_EXTRA_ARGS_STRING)
    ACCELERATE_CMD+=("${ACCELERATE_EXTRA_ARGS[@]}")
  fi
fi

echo "[INFO] dataset_root:        $DATASET_ROOT"
echo "[INFO] dataset_label:       $DATASET_LABEL"
echo "[INFO] render_input:        $RENDER_INPUT"
echo "[INFO] backbone_name:       $BACKBONE_NAME"
echo "[INFO] batch_size:          $BATCH_SIZE"
echo "[INFO] epochs:              $EPOCHS"
echo "[INFO] stage2_output_dir:   $OUTPUT_DIR"
if [[ "$USE_ACCELERATE" != "1" ]]; then
  echo "[INFO] launch_mode:         accelerate"
  echo "[INFO] accelerate_cmd:      ${ACCELERATE_CMD[*]}"
else
  echo "[INFO] launch_mode:         direct_single_process"
fi

if [[ "$USE_ACCELERATE" != "1" ]]; then
  "${ACCELERATE_CMD[@]}" "${CMD[@]}"
else
  "${CMD[@]}" --disable-accelerate
fi

echo "[INFO] Stage 2 helper run complete."
echo "[INFO] manifest:            $OUTPUT_DIR/train_manifest.jsonl"
echo "[INFO] manifest_summary:    $OUTPUT_DIR/train_manifest_summary.json"
echo "[INFO] run_summary:         $OUTPUT_DIR/stage2_run_summary.json"
echo "[INFO] trainer_plan:        $OUTPUT_DIR/trainer_plan.json"
