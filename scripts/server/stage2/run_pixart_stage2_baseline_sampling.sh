#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run standalone pretrained PixArt baseline sampling outside the Stage 2
#   training loop, using the same prompt-file flow as training-time step-0 /
#   periodic sampling for direct comparison.
#
# Usage:
#   bash scripts/server/stage2/run_pixart_stage2_baseline_sampling.sh
#
# Optional environment overrides:
#   DATASET_ROOT=/media/4T_HDD/cai/datasets/ImageNette/train
#   BACKBONE_NAME=PixArt-alpha/PixArt-Sigma-XL-2-512-MS
#   SAMPLE_PROMPT_FILE=configs/stage2/sample_prompts_imagenette.txt
#   SAMPLE_NUM_PROMPTS=8
#   SAMPLE_NUM_INFERENCE_STEPS=50
#   SAMPLE_GUIDANCE_SCALE=7
#   SAMPLE_SEED=42
#   RESOLUTION=512
#   BACKBONE_TORCH_DTYPE=float16
#   BACKBONE_DEVICE=cuda
#   BACKBONE_LOCAL_FILES_ONLY=1
#   OUTPUT_DIR=runs/stage2/baseline_samples/custom_run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_NAME="${ENV_NAME:-cspd-dd}"

DATASET_ROOT="${DATASET_ROOT:-/media/4T_HDD/cai/datasets/ImageNette/train}"
BACKBONE_NAME="${BACKBONE_NAME:-PixArt-alpha/PixArt-Sigma-XL-2-512-MS}"
SAMPLE_PROMPT_FILE="${SAMPLE_PROMPT_FILE:-configs/stage2/sample_prompts_imagenette.txt}"
SAMPLE_NUM_PROMPTS="${SAMPLE_NUM_PROMPTS:-8}"
SAMPLE_NUM_INFERENCE_STEPS="${SAMPLE_NUM_INFERENCE_STEPS:-50}"
SAMPLE_GUIDANCE_SCALE="${SAMPLE_GUIDANCE_SCALE:-7}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
RESOLUTION="${RESOLUTION:-512}"
BACKBONE_TORCH_DTYPE="${BACKBONE_TORCH_DTYPE:-float16}"
BACKBONE_DEVICE="${BACKBONE_DEVICE:-}"
BACKBONE_DEVICE_MAP="${BACKBONE_DEVICE_MAP:-}"
BACKBONE_LOCAL_FILES_ONLY="${BACKBONE_LOCAL_FILES_ONLY:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

EXTRA_ARGS=("$@")

resolve_repo_path() {
  local raw_path="$1"
  if [[ -z "$raw_path" ]]; then
    printf '%s\n' "$raw_path"
    return
  fi
  if [[ "$raw_path" = /* ]]; then
    printf '%s\n' "$raw_path"
    return
  fi
  printf '%s\n' "$REPO_ROOT/$raw_path"
}

cd "$REPO_ROOT"

RESOLVED_SAMPLE_PROMPT_FILE="$(resolve_repo_path "$SAMPLE_PROMPT_FILE")"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi

if [[ ! -f "$RESOLVED_SAMPLE_PROMPT_FILE" ]]; then
  echo "[ERROR] Sample prompt file not found: $SAMPLE_PROMPT_FILE"
  echo "[ERROR] Resolved path: $RESOLVED_SAMPLE_PROMPT_FILE"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

CMD=(
  python -m cspd_stage2.cli sample-baseline
  --dataset-root "$DATASET_ROOT"
  --backbone-name "$BACKBONE_NAME"
  --sample-prompt-file "$RESOLVED_SAMPLE_PROMPT_FILE"
  --sample-num-prompts "$SAMPLE_NUM_PROMPTS"
  --sample-num-inference-steps "$SAMPLE_NUM_INFERENCE_STEPS"
  --sample-guidance-scale "$SAMPLE_GUIDANCE_SCALE"
  --sample-seed "$SAMPLE_SEED"
  --resolution "$RESOLUTION"
  --backbone-torch-dtype "$BACKBONE_TORCH_DTYPE"
)

if [[ -n "$BACKBONE_DEVICE" ]]; then
  CMD+=(--backbone-device "$BACKBONE_DEVICE")
fi

if [[ -n "$BACKBONE_DEVICE_MAP" ]]; then
  CMD+=(--backbone-device-map "$BACKBONE_DEVICE_MAP")
fi

if [[ "$BACKBONE_LOCAL_FILES_ONLY" != "0" ]]; then
  CMD+=(--backbone-local-files-only)
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  CMD+=(--output-dir "$OUTPUT_DIR")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] repo_root:                 $REPO_ROOT"
echo "[INFO] dataset_root:              $DATASET_ROOT"
echo "[INFO] backbone_name:             $BACKBONE_NAME"
echo "[INFO] sample_prompt_file:        $SAMPLE_PROMPT_FILE"
echo "[INFO] resolved_sample_prompt:    $RESOLVED_SAMPLE_PROMPT_FILE"
echo "[INFO] sample_num_prompts:        $SAMPLE_NUM_PROMPTS"
echo "[INFO] sample_num_inference_steps:$SAMPLE_NUM_INFERENCE_STEPS"
echo "[INFO] sample_guidance_scale:     $SAMPLE_GUIDANCE_SCALE"
echo "[INFO] output_dir_override:       ${OUTPUT_DIR:-<auto>}"
echo "[INFO] launch cmd:                ${CMD[*]}"

"${CMD[@]}"
