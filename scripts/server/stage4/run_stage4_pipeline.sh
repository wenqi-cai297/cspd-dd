#!/usr/bin/env bash
set -euo pipefail

# Run Stage 4: generate distilled dataset from Stage 3 modes + Stage 2 LoRA.
#
# Usage:
#   bash scripts/server/stage4/run_stage4_pipeline.sh <stage3_modes_dir> <stage2_lora_weights|none>
#
# Examples:
#   bash scripts/server/stage4/run_stage4_pipeline.sh \
#     runs/stage3/ImageNette_train/ipc10/2026-04-14_XXXXXX/modes \
#     runs/stage2/train/ImageNette_train/.../checkpoint-8050/pytorch_lora_weights.safetensors
#
#   # Without LoRA (baseline SDXL)
#   bash scripts/server/stage4/run_stage4_pipeline.sh \
#     runs/stage3/ImageNette_train/ipc10/2026-04-14_XXXXXX/modes \
#     none
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   STAGE4_STEPS=50
#   STAGE4_GUIDANCE=9.0
#   STAGE4_SEED=42
#   STAGE4_STRENGTH=0.8  (img2img denoising strength)
#   STAGE4_VISUAL_MODE=medoid  (medoid|centroid|none)
#   STAGE4_REFINER=stabilityai/stable-diffusion-xl-refiner-1.0  (optional)
#   STAGE4_REFINER_STRENGTH=0.3  (optional)

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/stage4/run_stage4_pipeline.sh <stage3_modes_dir> <stage2_lora_weights|none>"
  exit 1
fi

MODES_DIR="$1"
LORA_WEIGHTS="$2"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
STEPS="${STAGE4_STEPS:-50}"
GUIDANCE="${STAGE4_GUIDANCE:-9.0}"
SEED="${STAGE4_SEED:-42}"
STRENGTH="${STAGE4_STRENGTH:-0.8}"
VISUAL_MODE="${STAGE4_VISUAL_MODE:-medoid}"
REFINER="${STAGE4_REFINER:-}"
REFINER_STRENGTH="${STAGE4_REFINER_STRENGTH:-0.3}"
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
MODES_PARENT="$(dirname "$MODES_DIR")"
DATASET_LABEL="$(basename "$(dirname "$(dirname "$MODES_PARENT")")")"
IPC_LABEL="$(basename "$(dirname "$MODES_PARENT")")"
LORA_TAG="lora"
if [[ "$LORA_WEIGHTS" == "none" ]]; then
  LORA_TAG="baseline"
fi
OUTPUT_DIR="runs/stage4/${DATASET_LABEL}/${IPC_LABEL}/${LORA_TAG}/${TIMESTAMP}"

echo "============================================================"
echo "[Stage 4] Distilled Dataset Generation"
echo "  modes_dir:     $MODES_DIR"
echo "  lora_weights:  $LORA_WEIGHTS"
echo "  output_dir:    $OUTPUT_DIR"
echo "  steps:         $STEPS"
echo "  guidance:      $GUIDANCE"
echo "  seed:          $SEED"
echo "  visual_mode:   $VISUAL_MODE"
echo "  strength:      $STRENGTH"
echo "  refiner:       ${REFINER:-none}"
echo "============================================================"

CMD=(
  cspd-stage4 generate
  --modes-dir "$MODES_DIR"
  --output-dir "$OUTPUT_DIR"
  --num-inference-steps "$STEPS"
  --guidance-scale "$GUIDANCE"
  --seed "$SEED"
  --visual-mode "$VISUAL_MODE"
  --strength "$STRENGTH"
)

if [[ "$LORA_WEIGHTS" != "none" ]]; then
  CMD+=(--lora-weights "$LORA_WEIGHTS")
fi

if [[ -n "$REFINER" ]]; then
  CMD+=(--refiner-model "$REFINER" --refiner-strength "$REFINER_STRENGTH")
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
