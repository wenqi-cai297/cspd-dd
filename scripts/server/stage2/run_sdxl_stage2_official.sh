#!/usr/bin/env bash
set -euo pipefail

# Run the SDXL Stage 2 official-diffusers wrapper path.
# Usage:
#   bash scripts/server/stage2/run_sdxl_stage2_official.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [extra args...]
# Environment helpers:
#   - CSPD_ENV_NAME=cspd-dd                      # conda env name override
#   - CSPD_STAGE2_SDXL_SCRIPT=/path/to/train_text_to_image_lora_sdxl.py
#   - DIFFUSERS_REPO_ROOT=/path/to/diffusers     # script auto-resolves under examples/text_to_image/
#   - STAGE2_DATASET_LABEL=imagenette_train      # optional output label override
#   - STAGE2_NUM_PROCESSES=2                     # forwarded to --sdxl-num-processes
#   - STAGE2_ACCELERATE_EXTRA_ARGS="--mixed_precision fp16"   # forwarded to --sdxl-accelerate-extra-arg
#   - STAGE2_OUTPUT_DIR=/custom/run/dir          # optional fixed run directory override
# Example:
#   STAGE2_NUM_PROCESSES=2 bash scripts/server/stage2/run_sdxl_stage2_official.sh /data/imagenette/train runs/stage1/render/imagenette/qwen_local/2026-04-02_010203/records.jsonl 1 1 --learning-rate 1e-4

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/stage2/run_sdxl_stage2_official.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [extra args...]"
  exit 1
fi

DATASET_ROOT="$1"
RENDER_INPUT="$2"
shift 2
BATCH_SIZE="${1:-1}"
if [[ $# -gt 0 ]]; then shift; fi
EPOCHS="${1:-1}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi

if [[ ! -f "$RENDER_INPUT" ]]; then
  echo "[ERROR] Stage 1 render input not found: $RENDER_INPUT"
  exit 1
fi

resolve_sdxl_script() {
  local requested="${CSPD_STAGE2_SDXL_SCRIPT:-}"
  local candidate=""

  if [[ -n "$requested" && -f "$requested" ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  if [[ -n "${DIFFUSERS_REPO_ROOT:-}" ]]; then
    candidate="${DIFFUSERS_REPO_ROOT}/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if [[ -n "${DIFFUSERS_HOME:-}" ]]; then
    candidate="${DIFFUSERS_HOME}/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if command -v train_text_to_image_lora_sdxl.py >/dev/null 2>&1; then
    command -v train_text_to_image_lora_sdxl.py
    return 0
  fi

  return 1
}

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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

bash scripts/server/check_stage2_sdxl_env.sh "${CSPD_STAGE2_SDXL_SCRIPT:-}" >/dev/null

DATASET_LABEL="${STAGE2_DATASET_LABEL:-$(derive_dataset_label "$DATASET_ROOT")}"
BACKBONE_NAME="stabilityai/stable-diffusion-xl-base-1.0"
BACKBONE_SLUG="$(echo "$BACKBONE_NAME" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="${STAGE2_OUTPUT_DIR:-runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}/${TIMESTAMP}}"

CMD=(
  cspd-stage2 train
  --dataset-root "$DATASET_ROOT"
  --render-input "$RENDER_INPUT"
  --output-dir "$OUTPUT_DIR"
  --backbone-name "$BACKBONE_NAME"
  --training-parameterization lora
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --resolution 512
  --sdxl-mixed-precision fp16
  --sdxl-report-to none
)

STAGE2_NUM_PROCESSES="${STAGE2_NUM_PROCESSES:-2}"
CMD+=(--sdxl-num-processes "$STAGE2_NUM_PROCESSES")

if [[ -n "${STAGE2_ACCELERATE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  ACCELERATE_EXTRA_ARGS=( ${STAGE2_ACCELERATE_EXTRA_ARGS} )
  for value in "${ACCELERATE_EXTRA_ARGS[@]}"; do
    CMD+=(--sdxl-accelerate-extra-arg "$value")
  done
fi

if RESOLVED_SCRIPT="$(resolve_sdxl_script)"; then
  CMD+=(--sdxl-official-script "$RESOLVED_SCRIPT")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] dataset_root:        $DATASET_ROOT"
echo "[INFO] dataset_label:       $DATASET_LABEL"
echo "[INFO] render_input:        $RENDER_INPUT"
echo "[INFO] output_dir:          $OUTPUT_DIR"
echo "[INFO] backbone_name:       $BACKBONE_NAME"
echo "[INFO] batch_size:          $BATCH_SIZE"
echo "[INFO] epochs:              $EPOCHS"
echo "[INFO] sdxl_script:         ${RESOLVED_SCRIPT:-train_text_to_image_lora_sdxl.py (PATH lookup / Python preflight)}"
echo "[INFO] launch:              ${CMD[*]}"

"${CMD[@]}"

echo "[INFO] SDXL Stage 2 helper run complete."
echo "[INFO] run_summary:         $OUTPUT_DIR/stage2_run_summary.json"
echo "[INFO] trainer_plan:        $OUTPUT_DIR/trainer_plan.json"
echo "[INFO] launch_plan:         $OUTPUT_DIR/sdxl_official_launch_plan.json"
echo "[INFO] materialized_data:   $OUTPUT_DIR/sdxl_materialized_dataset"
echo "[INFO] official_output:     $OUTPUT_DIR/official_output"
