#!/usr/bin/env bash
set -euo pipefail

# End-to-end CSPD dataset-distillation pipeline.
#
#   Stage 1 (extract -> normalize -> render)
#   -> Stage 2 (SDXL LoRA training, pick best-epoch checkpoint)
#   -> Stage 3A (DINOv2 encode, shared across IPC sweep)
#   -> per IPC: Stage 3B (HDBSCAN cluster) + Stage 4 (text2img generate) + Eval
#
# Every stage is idempotent: if its output already exists on disk the stage
# is skipped. Delete the corresponding run directory to force a rebuild.
#
# Usage:
#   bash scripts/pipelines/run_full_pipeline.sh <train_root> [val_root] [nclass]
#
# Positional args:
#   train_root  ImageFolder-style training split root (required).
#   val_root    ImageFolder-style validation split root (optional; defaults
#               to <parent(train_root)>/val when train_root ends in "train").
#   nclass      Number of classes (optional; defaults to the count of
#               subdirectories under train_root).
#
# Environment overrides (all optional):
#   CSPD_ENV_NAME=cspd-dd
#   STAGE1_BACKEND=qwen_local              # or "mock" for a plumbing smoke run
#   STAGE2_NUM_PROCESSES=2                 # accelerate --num_processes
#   STAGE2_BATCH_SIZE=8
#   STAGE2_EPOCHS=9                        # total training epochs.
#                                          # Epoch 9 was empirically best on
#                                          # ImageNette with cosine LR; training
#                                          # beyond that overfits.
#   STAGE2_RANK=64                         # LoRA rank
#   STAGE2_BEST_EPOCH=9                    # which epoch checkpoint to consume
#                                          # (0 => final weights). When this
#                                          # equals STAGE2_EPOCHS the target
#                                          # checkpoint is the final one.
#   DIFFUSERS_REPO_ROOT=./diffusers        # required for the Stage 2 trainer
#   PIPELINE_IPC="10"                      # space-separated IPC values to sweep
#   EVAL_ARCH=resnet_ap                    # single arch; use "all" for 3-arch
#   EVAL_REPEAT=3
#
# Prep assumption: classes.json + class_to_archetype.json must already exist
# in-tree (scripts/prep/prepare_stage1_metadata.sh covers this). The bundled
# ImageNet-1k manual mapping at configs/stage1/class_to_archetype_imagenet1k_manual.json
# is the default; new datasets with classes outside ImageNet-1k must run Prep
# first.

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/pipelines/run_full_pipeline.sh <train_root> [val_root] [nclass]"
  exit 1
fi

TRAIN_ROOT="$1"
VAL_ROOT="${2:-}"
NCLASS_ARG="${3:-}"

if [[ ! -d "$TRAIN_ROOT" ]]; then
  echo "[ERROR] train_root does not exist: $TRAIN_ROOT"
  exit 1
fi

# Auto-derive val_root if not provided
if [[ -z "$VAL_ROOT" ]]; then
  case "$(basename "$TRAIN_ROOT")" in
    train|Train|TRAIN)
      VAL_ROOT="$(dirname "$TRAIN_ROOT")/val" ;;
    *)
      VAL_ROOT="$(dirname "$TRAIN_ROOT")/val" ;;
  esac
fi
if [[ ! -d "$VAL_ROOT" ]]; then
  echo "[ERROR] val_root does not exist: $VAL_ROOT"
  echo "       Pass it explicitly as the 2nd positional arg."
  exit 1
fi

# Auto-detect nclass if not provided
NCLASS="${NCLASS_ARG:-$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')}"
if [[ "$NCLASS" -lt 1 ]]; then
  echo "[ERROR] could not detect any class subdirectories under $TRAIN_ROOT"
  exit 1
fi

# Configuration
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
STAGE1_BACKEND="${STAGE1_BACKEND:-qwen_local}"
STAGE2_NUM_PROCESSES="${STAGE2_NUM_PROCESSES:-2}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-8}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-9}"
STAGE2_RANK="${STAGE2_RANK:-64}"
STAGE2_BEST_EPOCH="${STAGE2_BEST_EPOCH:-9}"
IPC_LIST="${PIPELINE_IPC:-10}"
EVAL_ARCH="${EVAL_ARCH:-resnet_ap}"
EVAL_REPEAT="${EVAL_REPEAT:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

# Derive dataset label (matches Stage 2 output convention)
BASE_NAME="$(basename "$TRAIN_ROOT")"
PARENT_NAME="$(basename "$(dirname "$TRAIN_ROOT")")"
case "$BASE_NAME" in
  train|val|valid|validation|test|testing)
    DATASET_LABEL="${PARENT_NAME}_${BASE_NAME}" ;;
  *)
    DATASET_LABEL="$BASE_NAME" ;;
esac

BACKBONE_NAME="stabilityai/stable-diffusion-xl-base-1.0"
BACKBONE_SLUG="stabilityai_stable-diffusion-xl-base-1.0"

echo "============================================================"
echo " CSPD full pipeline"
echo "  dataset:    $DATASET_LABEL"
echo "  train_root: $TRAIN_ROOT"
echo "  val_root:   $VAL_ROOT"
echo "  nclass:     $NCLASS"
echo "  ipc_list:   $IPC_LIST"
echo "  backend:    $STAGE1_BACKEND"
echo "  eval:       $EVAL_ARCH x $EVAL_REPEAT"
echo "============================================================"

# ============================================================
# Stage 1: Extract -> Normalize -> Render (skip if render already exists)
# ============================================================
RENDER_DIR_ROOT="runs/stage1/render/${DATASET_LABEL}/${STAGE1_BACKEND}"
RENDER_INPUT=""
if [[ -d "$RENDER_DIR_ROOT" ]]; then
  LATEST_RENDER="$(ls -1d "${RENDER_DIR_ROOT}"/*/ 2>/dev/null | sort | tail -1)"
  if [[ -n "$LATEST_RENDER" && -f "${LATEST_RENDER}records.jsonl" ]]; then
    RENDER_INPUT="${LATEST_RENDER}records.jsonl"
  fi
fi

if [[ -n "$RENDER_INPUT" ]]; then
  echo ""
  echo "[Stage 1] Render output found at $RENDER_INPUT — skipping."
else
  echo ""
  echo "============================================================"
  echo "[Stage 1] Running Stage 1 pipeline (extract -> normalize -> render)..."
  echo "============================================================"
  bash scripts/stage1/run_stage1_pipeline.sh "$TRAIN_ROOT" "$STAGE1_BACKEND"

  LATEST_RENDER="$(ls -1d "${RENDER_DIR_ROOT}"/*/ 2>/dev/null | sort | tail -1)"
  RENDER_INPUT="${LATEST_RENDER}records.jsonl"
  if [[ ! -f "$RENDER_INPUT" ]]; then
    echo "[ERROR] Stage 1 render output not found after running pipeline"
    exit 1
  fi
fi
echo "[Stage 1] render: $RENDER_INPUT"

# ============================================================
# Stage 2: SDXL LoRA training (skip if target checkpoint already exists)
# ============================================================
STAGE2_TRAIN_DIR="runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}"
LORA_WEIGHTS=""

# Estimate steps/epoch for checkpoint path derivation
NUM_PAIRS="$(wc -l < "$RENDER_INPUT" 2>/dev/null | tr -d ' ' || echo 0)"
if [[ "$NUM_PAIRS" -lt 1 ]]; then NUM_PAIRS=1; fi
# Steps-per-epoch with ceiling division (accelerate + diffusers pads the
# last batch rather than dropping it, so NUM_PAIRS=12894 with batch=8 and
# num_processes=2 gives 806 steps/epoch, not 805). Getting this exact
# matters for the checkpoint path derivation below.
EFFECTIVE_BATCH=$(( STAGE2_BATCH_SIZE * STAGE2_NUM_PROCESSES ))
STEPS_PER_EPOCH=$(( (NUM_PAIRS + EFFECTIVE_BATCH - 1) / EFFECTIVE_BATCH ))
if [[ $STEPS_PER_EPOCH -lt 1 ]]; then STEPS_PER_EPOCH=100; fi
TARGET_STEP=$(( STEPS_PER_EPOCH * STAGE2_BEST_EPOCH ))

# Search existing training runs for the target checkpoint (latest first)
if [[ -d "$STAGE2_TRAIN_DIR" ]]; then
  for run_dir in $(ls -1d "${STAGE2_TRAIN_DIR}"/*/ 2>/dev/null | sort -r); do
    if [[ "$STAGE2_BEST_EPOCH" -gt 0 ]]; then
      CAND="${run_dir}official_output/checkpoint-${TARGET_STEP}/pytorch_lora_weights.safetensors"
    else
      CAND="${run_dir}official_output/pytorch_lora_weights.safetensors"
    fi
    if [[ -f "$CAND" ]]; then
      LORA_WEIGHTS="$CAND"
      break
    fi
  done
fi

if [[ -n "$LORA_WEIGHTS" ]]; then
  echo ""
  echo "[Stage 2] LoRA checkpoint found at $LORA_WEIGHTS — skipping training."
else
  echo ""
  echo "============================================================"
  echo "[Stage 2] Training SDXL LoRA (rank=$STAGE2_RANK, epochs=$STAGE2_EPOCHS)..."
  echo "  steps/epoch (estimated): $STEPS_PER_EPOCH"
  echo "============================================================"

  STAGE2_NUM_PROCESSES="$STAGE2_NUM_PROCESSES" bash scripts/stage2/run_sdxl_stage2_official.sh \
    "$TRAIN_ROOT" "$RENDER_INPUT" "$STAGE2_BATCH_SIZE" "$STAGE2_EPOCHS" \
    --adapter-rank "$STAGE2_RANK" \
    --save-every "$STEPS_PER_EPOCH"

  LATEST_RUN="$(ls -1d "${STAGE2_TRAIN_DIR}"/*/ 2>/dev/null | sort | tail -1)"
  if [[ "$STAGE2_BEST_EPOCH" -gt 0 ]]; then
    LORA_WEIGHTS="${LATEST_RUN}official_output/checkpoint-${TARGET_STEP}/pytorch_lora_weights.safetensors"
  fi
  if [[ ! -f "$LORA_WEIGHTS" ]]; then
    # Fall back to final weights
    LORA_WEIGHTS="${LATEST_RUN}official_output/pytorch_lora_weights.safetensors"
  fi
  if [[ ! -f "$LORA_WEIGHTS" ]]; then
    echo "[ERROR] LoRA weights not found after training"
    exit 1
  fi
fi
echo "[Stage 2] LoRA: $LORA_WEIGHTS"

# ============================================================
# Stage 3A: DINOv2 encode (shared across the IPC sweep)
# ============================================================
ENCODE_DIR="runs/stage3/${DATASET_LABEL}/encoded"

if [[ -f "${ENCODE_DIR}/dino_embeds.pt" && -f "${ENCODE_DIR}/encode_index.json" ]]; then
  echo ""
  echo "[Stage 3A] Encode output found at $ENCODE_DIR — skipping."
else
  echo ""
  echo "============================================================"
  echo "[Stage 3A] Encoding dataset with DINOv2..."
  echo "============================================================"
  cspd-stage3 encode \
    --dataset-root "$TRAIN_ROOT" \
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

  MODES_DIR="runs/stage3/${DATASET_LABEL}/ipc${IPC}/modes_hdbscan"
  if [[ -f "${MODES_DIR}/modes_index.json" ]]; then
    echo "[IPC=$IPC] Stage 3B modes found at $MODES_DIR — skipping cluster."
  else
    echo "[IPC=$IPC] Stage 3B clustering..."
    cspd-stage3 cluster \
      --encode-dir "$ENCODE_DIR" \
      --output-dir "$MODES_DIR" \
      --ipc "$IPC"
  fi

  TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
  STAGE4_OUT="runs/stage4/${DATASET_LABEL}/ipc${IPC}/lora/${TIMESTAMP}"
  echo "[IPC=$IPC] Stage 4 text2img -> $STAGE4_OUT"
  cspd-stage4 generate \
    --modes-dir "$MODES_DIR" \
    --output-dir "$STAGE4_OUT" \
    --lora-weights "$LORA_WEIGHTS" \
    --model-name "$BACKBONE_NAME" \
    --visual-mode none

  echo "[IPC=$IPC] Eval ($EVAL_ARCH, repeat=$EVAL_REPEAT)"
  EVAL_REPEAT="$EVAL_REPEAT" bash scripts/eval/run_eval_pipeline.sh \
    "${STAGE4_OUT}/images" \
    "$VAL_ROOT" \
    "$NCLASS" "$IPC" "$EVAL_ARCH"
done

echo ""
echo "############################################################"
echo " Full pipeline complete."
echo "  dataset:    $DATASET_LABEL"
echo "  render:     $RENDER_INPUT"
echo "  LoRA:       $LORA_WEIGHTS"
echo "  encode:     $ENCODE_DIR"
echo "  IPC sweep:  $IPC_LIST"
echo "############################################################"
