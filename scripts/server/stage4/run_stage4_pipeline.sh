#!/usr/bin/env bash
set -euo pipefail

# Run Stage 4: dual-anchor conditioned distilled dataset generation.
#
# Usage:
#   bash scripts/server/stage4/run_stage4_pipeline.sh <stage3_modes_dir> <stage2_lora_weights> [strength]
#
# Examples:
#   # ImageNette, IPC=10, strength=0.5
#   bash scripts/server/stage4/run_stage4_pipeline.sh \
#     runs/stage3/ImageNette_train/ipc10/2026-04-13_XXXXXX/modes \
#     runs/stage2/train/ImageNette_train/.../official_output/checkpoint-8050/pytorch_lora_weights.safetensors \
#     0.5
#
#   # Without LoRA (baseline SDXL)
#   bash scripts/server/stage4/run_stage4_pipeline.sh \
#     runs/stage3/ImageNette_train/ipc10/2026-04-13_XXXXXX/modes \
#     none \
#     0.5
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   STAGE4_STEPS=50                  # inference steps
#   STAGE4_GUIDANCE=7.5              # guidance scale
#   STAGE4_SEED=42                   # RNG seed
#   STAGE4_DEVICE=cuda
#   STAGE4_DTYPE=float16

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/stage4/run_stage4_pipeline.sh <stage3_modes_dir> <stage2_lora_weights|none> [strength] [semantic_mode] [visual_mode]"
  exit 1
fi

MODES_DIR="$1"
LORA_WEIGHTS="$2"
STRENGTH="${3:-0.5}"
SEMANTIC_MODE="${4:-caption}"
VISUAL_MODE="${5:-centroid}"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
STEPS="${STAGE4_STEPS:-50}"
GUIDANCE="${STAGE4_GUIDANCE:-7.5}"
SEED="${STAGE4_SEED:-42}"
DEVICE="${STAGE4_DEVICE:-cuda}"
DTYPE="${STAGE4_DTYPE:-float16}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$MODES_DIR" ]]; then
  echo "[ERROR] Stage 3 modes directory not found: $MODES_DIR"
  exit 1
fi

if [[ "$LORA_WEIGHTS" != "none" && ! -f "$LORA_WEIGHTS" ]]; then
  echo "[ERROR] LoRA weights not found: $LORA_WEIGHTS"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
# Derive output dir from modes_dir path
MODES_PARENT="$(dirname "$MODES_DIR")"
DATASET_LABEL="$(basename "$(dirname "$(dirname "$MODES_PARENT")")")"
IPC_LABEL="$(basename "$(dirname "$MODES_PARENT")")"
LORA_TAG="lora"
if [[ "$LORA_WEIGHTS" == "none" ]]; then
  LORA_TAG="baseline"
fi
OUTPUT_DIR="runs/stage4/${DATASET_LABEL}/${IPC_LABEL}/s${STRENGTH}_${LORA_TAG}/${TIMESTAMP}"

echo "============================================================"
echo "[Stage 4] Dual-Anchor Distilled Generation"
echo "  modes_dir:     $MODES_DIR"
echo "  lora_weights:  $LORA_WEIGHTS"
echo "  output_dir:    $OUTPUT_DIR"
echo "  strength:      $STRENGTH"
echo "  steps:         $STEPS"
echo "  guidance:      $GUIDANCE"
echo "  seed:          $SEED"
echo "============================================================"

CMD=(
  cspd-stage4 generate
  --modes-dir "$MODES_DIR"
  --output-dir "$OUTPUT_DIR"
  --strength "$STRENGTH"
  --num-inference-steps "$STEPS"
  --guidance-scale "$GUIDANCE"
  --seed "$SEED"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --semantic-mode "$SEMANTIC_MODE"
  --visual-mode "$VISUAL_MODE"
)

if [[ "$LORA_WEIGHTS" != "none" ]]; then
  CMD+=(--lora-weights "$LORA_WEIGHTS")
fi

"${CMD[@]}"

echo ""
echo "============================================================"
echo "[Stage 4] Pipeline complete"
echo "  Output:     $OUTPUT_DIR"
echo "  Images:     ${OUTPUT_DIR}/images/"
echo "  Metadata:   ${OUTPUT_DIR}/distilled_metadata.json"
echo "  Summary:    ${OUTPUT_DIR}/stage4_summary.json"
echo "============================================================"
