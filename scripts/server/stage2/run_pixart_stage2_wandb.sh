#!/usr/bin/env bash
set -euo pipefail

# Purpose:
#   Run the current PixArt Stage 2 LoRA training recipe with W&B logging
#   and periodic sampling enabled, without forcing the user to copy a long CLI.
#
# Usage:
#   bash scripts/server/stage2/run_pixart_stage2_wandb.sh
#
# Optional environment overrides:
#   STAGE2_NUM_PROCESSES=2
#   WANDB_PROJECT=cspd-stage2
#   WANDB_RUN_NAME=custom-name
#   WANDB_MODE=online|offline|disabled
#   SAMPLE_EVERY=100
#   SAMPLE_NUM_PROMPTS=8
#   SAMPLE_NUM_INFERENCE_STEPS=20
#   SAMPLE_GUIDANCE_SCALE=4.5
#   SAMPLE_SEED=42
#   EPOCHS=20
#   LEARNING_RATE=2e-5
#   BATCH_SIZE=32
#   GRADIENT_ACCUMULATION_STEPS=4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_NAME="${ENV_NAME:-cspd-dd}"

DATASET_ROOT="${DATASET_ROOT:-/media/4T_HDD/cai/datasets/ImageNette/train}"
RENDER_INPUT="${RENDER_INPUT:-runs/stage1/render/ImageNette/qwen_local/2026-04-02_212610/records.jsonl}"
BACKBONE_NAME="${BACKBONE_NAME:-PixArt-alpha/PixArt-Sigma-XL-2-512-MS}"
RESOLUTION="${RESOLUTION:-512}"
BACKBONE_TORCH_DTYPE="${BACKBONE_TORCH_DTYPE:-float16}"
TRAINING_PARAMETERIZATION="${TRAINING_PARAMETERIZATION:-lora}"
TRAINABLE_COMPONENT_GROUP="${TRAINABLE_COMPONENT_GROUP:-full_transformer}"
ADAPTER_RANK="${ADAPTER_RANK:-64}"
ADAPTER_ALPHA="${ADAPTER_ALPHA:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1000}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.01}"
ADAM_WEIGHT_DECAY="${ADAM_WEIGHT_DECAY:-0.0}"
PIXART_SIGMA_PROMPT_DROPOUT_PROB="${PIXART_SIGMA_PROMPT_DROPOUT_PROB:-0.1}"
EPOCHS="${EPOCHS:-20}"

WANDB_PROJECT="${WANDB_PROJECT:-cspd-stage2}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-pixart-imagenette-b32ga4-e${EPOCHS}-$(date +%Y%m%d_%H%M%S)}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_TAGS=(pixart stage2 imagenette lora)

SAMPLE_EVERY="${SAMPLE_EVERY:-100}"
SAMPLE_PROMPT_FILE="${SAMPLE_PROMPT_FILE:-configs/stage2/sample_prompts_imagenette.txt}"
SAMPLE_NUM_PROMPTS="${SAMPLE_NUM_PROMPTS:-8}"
SAMPLE_NUM_INFERENCE_STEPS="${SAMPLE_NUM_INFERENCE_STEPS:-20}"
SAMPLE_GUIDANCE_SCALE="${SAMPLE_GUIDANCE_SCALE:-4.5}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"

EXTRA_ARGS=("$@")

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi

cd "$REPO_ROOT"

if [[ ! -f "$RENDER_INPUT" ]]; then
  echo "[ERROR] Stage 1 render input not found: $RENDER_INPUT"
  exit 1
fi

if [[ ! -f "$SAMPLE_PROMPT_FILE" ]]; then
  echo "[ERROR] Sample prompt file not found: $SAMPLE_PROMPT_FILE"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

CMD=(
  accelerate launch
  --num_processes "${STAGE2_NUM_PROCESSES:-2}"
  -m cspd_stage2.cli train
  --dataset-root "$DATASET_ROOT"
  --render-input "$RENDER_INPUT"
  --backbone-name "$BACKBONE_NAME"
  --resolution "$RESOLUTION"
  --backbone-torch-dtype "$BACKBONE_TORCH_DTYPE"
  --backbone-local-files-only
  --training-parameterization "$TRAINING_PARAMETERIZATION"
  --trainable-component-group "$TRAINABLE_COMPONENT_GROUP"
  --adapter-rank "$ADAPTER_RANK"
  --adapter-alpha "$ADAPTER_ALPHA"
  --batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning-rate "$LEARNING_RATE"
  --lr-scheduler "$LR_SCHEDULER"
  --lr-warmup-steps "$LR_WARMUP_STEPS"
  --max-grad-norm "$MAX_GRAD_NORM"
  --adam-weight-decay "$ADAM_WEIGHT_DECAY"
  --pixart-sigma-prompt-dropout-prob "$PIXART_SIGMA_PROMPT_DROPOUT_PROB"
  --epochs "$EPOCHS"
  --wandb
  --wandb-project "$WANDB_PROJECT"
  --wandb-run-name "$WANDB_RUN_NAME"
  --wandb-mode "$WANDB_MODE"
  --sample-every "$SAMPLE_EVERY"
  --sample-prompt-file "$SAMPLE_PROMPT_FILE"
  --sample-num-prompts "$SAMPLE_NUM_PROMPTS"
  --sample-num-inference-steps "$SAMPLE_NUM_INFERENCE_STEPS"
  --sample-guidance-scale "$SAMPLE_GUIDANCE_SCALE"
  --sample-seed "$SAMPLE_SEED"
)

if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb-entity "$WANDB_ENTITY")
fi

if [[ -n "$WANDB_DIR" ]]; then
  CMD+=(--wandb-dir "$WANDB_DIR")
fi

if [[ -n "$WANDB_RESUME" ]]; then
  CMD+=(--wandb-resume "$WANDB_RESUME")
fi

if [[ -n "$WANDB_RUN_ID" ]]; then
  CMD+=(--wandb-run-id "$WANDB_RUN_ID")
fi

for tag in "${WANDB_TAGS[@]}"; do
  CMD+=(--wandb-tag "$tag")
done

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] repo_root:                 $REPO_ROOT"
echo "[INFO] dataset_root:              $DATASET_ROOT"
echo "[INFO] render_input:              $RENDER_INPUT"
echo "[INFO] backbone_name:             $BACKBONE_NAME"
echo "[INFO] batch_size:                $BATCH_SIZE"
echo "[INFO] gradient_accumulation:     $GRADIENT_ACCUMULATION_STEPS"
echo "[INFO] epochs:                    $EPOCHS"
echo "[INFO] learning_rate:             $LEARNING_RATE"
echo "[INFO] wandb_project:             $WANDB_PROJECT"
echo "[INFO] wandb_run_name:            $WANDB_RUN_NAME"
echo "[INFO] sample_every:              $SAMPLE_EVERY"
echo "[INFO] sample_prompt_file:        $SAMPLE_PROMPT_FILE"
echo "[INFO] launch cmd:                ${CMD[*]}"

"${CMD[@]}"
