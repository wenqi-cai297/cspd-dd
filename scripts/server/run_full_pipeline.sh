#!/usr/bin/env bash
set -euo pipefail

# Full CSPD pipeline: Prep → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Eval.
# Each stage checks if output already exists and skips if so.
#
# Usage:
#   bash scripts/server/run_full_pipeline.sh
#
# Environment overrides:
#   CSPD_ENV_NAME=cspd-dd
#   STAGE2_NUM_PROCESSES=2
#   DIFFUSERS_REPO_ROOT=./diffusers
#   EVAL_REPEAT=3
#   PIPELINE_IPC="10 20 50"           # IPC values to sweep

# ============================================================
# Configuration — edit these for different datasets
# ============================================================
DATASET_ROOT="/media/4T_HDD/cai/datasets/ImageNette/train"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
NCLASS=10
BACKEND="qwen_local"

# Stage 2 training config
STAGE2_BATCH_SIZE=8
STAGE2_EPOCHS=15
STAGE2_RANK=64
STAGE2_BEST_EPOCH=9  # which epoch checkpoint to use (0 = use final weights)

# Stage 4 / Eval config
IPC_LIST="${PIPELINE_IPC:-10 20 50}"
EVAL_ARCH="resnet_ap"
EVAL_REPEAT="${EVAL_REPEAT:-3}"

# ============================================================
# Derived paths
# ============================================================
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

# Dataset label
BASE_NAME="$(basename "$DATASET_ROOT")"
PARENT_NAME="$(basename "$(dirname "$DATASET_ROOT")")"
case "$BASE_NAME" in
  train|val|valid|validation|test|testing)
    DATASET_LABEL="${PARENT_NAME}_${BASE_NAME}" ;;
  *)
    DATASET_LABEL="$BASE_NAME" ;;
esac

BACKBONE_NAME="stabilityai/stable-diffusion-xl-base-1.0"
BACKBONE_SLUG="$(echo "$BACKBONE_NAME" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"

echo "============================================================"
echo " CSPD Full Pipeline"
echo "  dataset:    $DATASET_LABEL"
echo "  nclass:     $NCLASS"
echo "  ipc_list:   $IPC_LIST"
echo "  eval_arch:  $EVAL_ARCH"
echo "============================================================"

# ============================================================
# Stage 1: Extraction → Normalization → Render
# ============================================================
# Find latest render records.jsonl
RENDER_DIR="runs/stage1/render/${DATASET_LABEL}/${BACKEND}"
RENDER_INPUT=""
if [[ -d "$RENDER_DIR" ]]; then
  # Find the latest timestamp directory with records.jsonl
  LATEST_RENDER="$(ls -1d "${RENDER_DIR}"/*/ 2>/dev/null | sort | tail -1)"
  if [[ -n "$LATEST_RENDER" && -f "${LATEST_RENDER}records.jsonl" ]]; then
    RENDER_INPUT="${LATEST_RENDER}records.jsonl"
  fi
fi

if [[ -n "$RENDER_INPUT" && -f "$RENDER_INPUT" ]]; then
  echo ""
  echo "[Stage 1] Render output found: $RENDER_INPUT, skipping."
else
  echo ""
  echo "============================================================"
  echo "[Stage 1] Running full Stage 1 pipeline..."
  echo "============================================================"
  bash scripts/server/stage1/run_stage1_pipeline.sh "$DATASET_ROOT" "$BACKEND"

  # Find the newly created render output
  LATEST_RENDER="$(ls -1d "${RENDER_DIR}"/*/ 2>/dev/null | sort | tail -1)"
  RENDER_INPUT="${LATEST_RENDER}records.jsonl"
  if [[ ! -f "$RENDER_INPUT" ]]; then
    echo "[ERROR] Stage 1 render output not found after running pipeline"
    exit 1
  fi
fi
echo "[Stage 1] Using render: $RENDER_INPUT"

# ============================================================
# Stage 2: SDXL LoRA Training
# ============================================================
# Find latest training run with the target checkpoint
STAGE2_TRAIN_DIR="runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}"
LORA_WEIGHTS=""

if [[ -d "$STAGE2_TRAIN_DIR" ]]; then
  # Search for the target checkpoint across all training runs (latest first)
  for run_dir in $(ls -1d "${STAGE2_TRAIN_DIR}"/*/ 2>/dev/null | sort -r); do
    if [[ $STAGE2_BEST_EPOCH -gt 0 ]]; then
      # Calculate step number: steps_per_epoch * best_epoch
      # Read from the training plan if available, otherwise estimate
      NUM_PAIRS=$(wc -l < "$RENDER_INPUT" 2>/dev/null || echo "13000")
      NUM_PROCESSES="${STAGE2_NUM_PROCESSES:-2}"
      STEPS_PER_EPOCH=$(( NUM_PAIRS / (STAGE2_BATCH_SIZE * NUM_PROCESSES) ))
      TARGET_STEP=$(( STEPS_PER_EPOCH * STAGE2_BEST_EPOCH ))
      CANDIDATE="${run_dir}official_output/checkpoint-${TARGET_STEP}/pytorch_lora_weights.safetensors"
    else
      CANDIDATE="${run_dir}official_output/pytorch_lora_weights.safetensors"
    fi
    if [[ -f "$CANDIDATE" ]]; then
      LORA_WEIGHTS="$CANDIDATE"
      break
    fi
  done
fi

if [[ -n "$LORA_WEIGHTS" ]]; then
  echo ""
  echo "[Stage 2] LoRA weights found: $LORA_WEIGHTS, skipping training."
else
  echo ""
  echo "============================================================"
  echo "[Stage 2] Running SDXL LoRA training..."
  echo "============================================================"

  # Calculate checkpoint interval (every epoch)
  NUM_PAIRS=$(wc -l < "$RENDER_INPUT")
  NUM_PROCESSES="${STAGE2_NUM_PROCESSES:-2}"
  STEPS_PER_EPOCH=$(( NUM_PAIRS / (STAGE2_BATCH_SIZE * NUM_PROCESSES) ))
  if [[ $STEPS_PER_EPOCH -lt 1 ]]; then STEPS_PER_EPOCH=100; fi

  bash scripts/server/stage2/run_sdxl_stage2_official.sh \
    "$DATASET_ROOT" "$RENDER_INPUT" "$STAGE2_BATCH_SIZE" "$STAGE2_EPOCHS" \
    --adapter-rank "$STAGE2_RANK" \
    --save-every "$STEPS_PER_EPOCH"

  # Find the newly created checkpoint
  LATEST_RUN="$(ls -1d "${STAGE2_TRAIN_DIR}"/*/ 2>/dev/null | sort | tail -1)"
  TARGET_STEP=$(( STEPS_PER_EPOCH * STAGE2_BEST_EPOCH ))
  LORA_WEIGHTS="${LATEST_RUN}official_output/checkpoint-${TARGET_STEP}/pytorch_lora_weights.safetensors"
  if [[ ! -f "$LORA_WEIGHTS" ]]; then
    # Fallback to final weights
    LORA_WEIGHTS="${LATEST_RUN}official_output/pytorch_lora_weights.safetensors"
  fi
  if [[ ! -f "$LORA_WEIGHTS" ]]; then
    echo "[ERROR] LoRA weights not found after training"
    exit 1
  fi
fi
echo "[Stage 2] Using LoRA: $LORA_WEIGHTS"

# ============================================================
# Stage 3A: DINOv2 Encode
# ============================================================
ENCODE_DIR="runs/stage3/${DATASET_LABEL}/encoded"

if [[ -f "${ENCODE_DIR}/dino_embeds.pt" && -f "${ENCODE_DIR}/encode_index.json" ]]; then
  echo ""
  echo "[Stage 3A] Encode output found at ${ENCODE_DIR}, skipping."
else
  echo ""
  echo "============================================================"
  echo "[Stage 3A] Encoding dataset with DINOv2..."
  echo "============================================================"
  cspd-stage3 encode \
    --dataset-root "$DATASET_ROOT" \
    --render-input "$RENDER_INPUT" \
    --output-dir "$ENCODE_DIR"
fi

# ============================================================
# Stage 3B + Stage 4 + Eval: per-IPC sweep
# ============================================================
for IPC in $IPC_LIST; do
  echo ""
  echo "############################################################"
  echo "# IPC=$IPC"
  echo "############################################################"

  # --- Stage 3B: Cluster ---
  MODES_DIR="runs/stage3/${DATASET_LABEL}/ipc${IPC}/modes_kmeans"
  if [[ -f "${MODES_DIR}/modes_index.json" ]]; then
    echo "[IPC=$IPC] Stage 3B: modes found at ${MODES_DIR}, skipping."
  else
    echo "[IPC=$IPC] Stage 3B: clustering..."
    cspd-stage3 cluster \
      --encode-dir "$ENCODE_DIR" \
      --output-dir "$MODES_DIR" \
      --ipc "$IPC"
  fi

  # --- Stage 4: Generate ---
  TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
  STAGE4_OUT="runs/stage4/${DATASET_LABEL}/ipc${IPC}/lora/${TIMESTAMP}"
  echo "[IPC=$IPC] Stage 4: generating → $STAGE4_OUT"
  cspd-stage4 generate \
    --modes-dir "$MODES_DIR" \
    --output-dir "$STAGE4_OUT" \
    --lora-weights "$LORA_WEIGHTS" \
    --visual-mode none

  # --- Eval ---
  echo "[IPC=$IPC] Eval: ${EVAL_ARCH}, repeat=${EVAL_REPEAT}"
  EVAL_REPEAT=$EVAL_REPEAT bash scripts/server/eval/run_eval_pipeline.sh \
    "${STAGE4_OUT}/images" \
    "$VAL_DIR" \
    "$NCLASS" "$IPC" "$EVAL_ARCH"

  echo "[IPC=$IPC] Done."
done

echo ""
echo "############################################################"
echo " Full pipeline complete."
echo "  Dataset:    $DATASET_LABEL"
echo "  Render:     $RENDER_INPUT"
echo "  LoRA:       $LORA_WEIGHTS"
echo "  Encode:     $ENCODE_DIR"
echo "############################################################"
