#!/usr/bin/env bash
set -euo pipefail

# Run the full Stage 2 pipeline: SDXL LoRA training + inference sampling.
#
# Usage:
#   bash scripts/stage2/run_stage2_pipeline.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [rank]
#
# Examples:
#   # ImageNette with default config (best known: rank=64, epoch=15, batch=8)
#   bash scripts/stage2/run_stage2_pipeline.sh \
#     /media/4T_HDD/cai/datasets/ImageNette/train \
#     runs/stage1/render/ImageNette_train/qwen_local/2026-04-12_XXXXXX/records.jsonl \
#     8 20 64
#
# Notes:
#   - Trains SDXL LoRA with checkpoints every 5 epochs (auto-calculated)
#   - Runs inference sampling on all checkpoints + final weights for comparison
#   - Best known config from ImageNette experiments: rank=64, epoch=15
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   STAGE2_NUM_PROCESSES=2           # number of GPUs
#   DIFFUSERS_REPO_ROOT=./diffusers  # path to cloned diffusers repo

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/stage2/run_stage2_pipeline.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [rank]"
  exit 1
fi

DATASET_ROOT="$1"
RENDER_INPUT="$2"
BATCH_SIZE="${3:-8}"
EPOCHS="${4:-20}"
RANK="${5:-64}"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
NUM_PROCESSES="${STAGE2_NUM_PROCESSES:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "[ERROR] Dataset root not found: $DATASET_ROOT"
  exit 1
fi
if [[ ! -f "$RENDER_INPUT" ]]; then
  echo "[ERROR] Stage 1 render records not found: $RENDER_INPUT"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

# Derive dataset label
derive_dataset_label() {
  local dataset_root="$1"
  local base_name parent_name
  base_name="$(basename "$dataset_root")"
  parent_name="$(basename "$(dirname "$dataset_root")")"
  case "$base_name" in
    train|val|valid|validation|test|testing)
      if [[ -n "$parent_name" && "$parent_name" != "." && "$parent_name" != "/" ]]; then
        printf '%s_%s\n' "$parent_name" "$base_name"; return
      fi ;;
  esac
  printf '%s\n' "$base_name"
}

DATASET_LABEL="$(derive_dataset_label "$DATASET_ROOT")"
BACKBONE_NAME="stabilityai/stable-diffusion-xl-base-1.0"
BACKBONE_SLUG="$(echo "$BACKBONE_NAME" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}/${TIMESTAMP}"

# Calculate checkpoint interval (every 5 epochs)
# steps_per_epoch ≈ num_pairs / (batch_size * num_processes)
NUM_PAIRS=$(wc -l < "$RENDER_INPUT")
STEPS_PER_EPOCH=$(( NUM_PAIRS / (BATCH_SIZE * NUM_PROCESSES) ))
CHECKPOINT_INTERVAL=$(( STEPS_PER_EPOCH * 5 ))
if [[ $CHECKPOINT_INTERVAL -lt 1 ]]; then CHECKPOINT_INTERVAL=100; fi

# Resolve SDXL training script
resolve_sdxl_script() {
  local candidate=""
  if [[ -n "${CSPD_STAGE2_SDXL_SCRIPT:-}" && -f "${CSPD_STAGE2_SDXL_SCRIPT}" ]]; then
    printf '%s\n' "$CSPD_STAGE2_SDXL_SCRIPT"; return 0
  fi
  for root in "${DIFFUSERS_REPO_ROOT:-}" "${DIFFUSERS_HOME:-}"; do
    candidate="${root}/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    if [[ -n "$root" && -f "$candidate" ]]; then printf '%s\n' "$candidate"; return 0; fi
  done
  return 1
}

echo "============================================================"
echo "[Stage 2] SDXL LoRA Training"
echo "  dataset_root:   $DATASET_ROOT"
echo "  render_input:   $RENDER_INPUT"
echo "  output_dir:     $OUTPUT_DIR"
echo "  backbone:       $BACKBONE_NAME"
echo "  rank:           $RANK"
echo "  batch_size:     $BATCH_SIZE"
echo "  epochs:         $EPOCHS"
echo "  num_processes:  $NUM_PROCESSES"
echo "  checkpoint_interval: $CHECKPOINT_INTERVAL steps (~5 epochs)"
echo "============================================================"

TRAIN_CMD=(
  cspd-stage2 train
  --dataset-root "$DATASET_ROOT"
  --render-input "$RENDER_INPUT"
  --output-dir "$OUTPUT_DIR"
  --backbone-name "$BACKBONE_NAME"
  --training-parameterization lora
  --adapter-rank "$RANK"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --resolution 512
  --save-every "$CHECKPOINT_INTERVAL"
  --sdxl-mixed-precision fp16
  --sdxl-lr-scheduler constant
  --sdxl-num-processes "$NUM_PROCESSES"
)

if RESOLVED_SCRIPT="$(resolve_sdxl_script)"; then
  TRAIN_CMD+=(--sdxl-official-script "$RESOLVED_SCRIPT")
fi

"${TRAIN_CMD[@]}"

echo "[Stage 2] Training complete: $OUTPUT_DIR"

# --- Inference sampling on checkpoints ---
SAMPLES_BASE="runs/stage2/samples/${DATASET_LABEL}/${TIMESTAMP}"

echo ""
echo "============================================================"
echo "[Stage 2] Inference sampling"
echo "============================================================"

# Baseline (no LoRA)
echo "[Sampling] baseline (no LoRA)..."
python scripts/stage2/sample_sdxl_lora.py \
  --no-lora \
  --output-dir "${SAMPLES_BASE}/baseline"

# Final weights
if [[ -f "${OUTPUT_DIR}/official_output/pytorch_lora_weights.safetensors" ]]; then
  echo "[Sampling] final weights..."
  python scripts/stage2/sample_sdxl_lora.py \
    --lora-weights "${OUTPUT_DIR}/official_output/pytorch_lora_weights.safetensors" \
    --output-dir "${SAMPLES_BASE}/final"
fi

# Intermediate checkpoints
for ckpt_dir in "${OUTPUT_DIR}/official_output"/checkpoint-*/; do
  if [[ -d "$ckpt_dir" ]]; then
    ckpt_name="$(basename "$ckpt_dir")"
    weights="${ckpt_dir}pytorch_lora_weights.safetensors"
    if [[ -f "$weights" ]]; then
      echo "[Sampling] ${ckpt_name}..."
      python scripts/stage2/sample_sdxl_lora.py \
        --lora-weights "$weights" \
        --output-dir "${SAMPLES_BASE}/${ckpt_name}"
    fi
  fi
done

echo ""
echo "============================================================"
echo "[Stage 2] Pipeline complete"
echo "  Training:  $OUTPUT_DIR"
echo "  Samples:   $SAMPLES_BASE"
echo "============================================================"
