#!/usr/bin/env bash
set -euo pipefail

# Phase 3 set-level candidate selection — A/B vs HDBSCAN + medoid baseline (62.33%).
#
# Uses the same HDBSCAN modes as the baseline, same LoRA, same seed. The only
# difference is Stage 4 generates N candidates per mode and greedy-selects
# one per mode to minimize set-level distance to the real class distribution
# (D3HR-style moment matching).
#
# Expected: comparable or better than 62.33% on ImageNette IPC=10 (ResNetAP-10,
# 3 repeats).
#
# Usage:
#   bash scripts/server/run_setlevel_phase3.sh
#
# Environment:
#   SETLEVEL_NUM_CANDIDATES=10   (N candidates per mode, default 10)
#   SETLEVEL_OBJECTIVE=moments   (moments | mmd, default moments)
#   SETLEVEL_SEED=42
#   EVAL_REPEAT=3

ENCODE_DIR="runs/stage3/ImageNette_train/encoded_with_vae"
MODES_DIR="runs/stage3/ImageNette_train/ipc10/hdbscan_medoid"
LORA="runs/stage2/train/ImageNette_train/stabilityai_stable-diffusion-xl-base-1.0/2026-04-14_181645/official_output/checkpoint-7254/pytorch_lora_weights.safetensors"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
NCLASS=10
IPC=10

NUM_CANDIDATES="${SETLEVEL_NUM_CANDIDATES:-10}"
OBJECTIVE="${SETLEVEL_OBJECTIVE:-moments}"
SEED="${SETLEVEL_SEED:-42}"
EVAL_REPEAT="${EVAL_REPEAT:-3}"

ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ ! -f "${MODES_DIR}/modes_index.json" ]]; then
  echo "[ERROR] HDBSCAN modes not found at ${MODES_DIR}. Run Stage 3 first."
  exit 1
fi

if [[ ! -f "$LORA" ]]; then
  echo "[ERROR] LoRA weights not found: $LORA"
  exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
STAGE4_OUT="runs/stage4/ImageNette_train/ipc${IPC}/lora/setlevel_${OBJECTIVE}_n${NUM_CANDIDATES}_${TIMESTAMP}"

echo "============================================================"
echo "[Phase 3 set-level] Generation"
echo "  modes_dir:        $MODES_DIR (HDBSCAN baseline 62.33%)"
echo "  lora:             $LORA"
echo "  num_candidates:   $NUM_CANDIDATES"
echo "  set_objective:    $OBJECTIVE"
echo "  seed:             $SEED"
echo "  output:           $STAGE4_OUT"
echo "============================================================"

cspd-stage4 generate \
  --modes-dir "$MODES_DIR" \
  --output-dir "$STAGE4_OUT" \
  --lora-weights "$LORA" \
  --model-name "stabilityai/stable-diffusion-xl-base-1.0" \
  --visual-mode none \
  --resolution 512 \
  --guidance-scale 7.5 \
  --num-inference-steps 50 \
  --seed "$SEED" \
  --num-candidates "$NUM_CANDIDATES" \
  --set-level-selection \
  --set-objective "$OBJECTIVE" \
  --candidate-probe-dir "$ENCODE_DIR" \
  --eval-representativeness

echo ""
echo "============================================================"
echo "[Phase 3 set-level] Eval (resnet_ap, repeat=$EVAL_REPEAT)"
echo "============================================================"
EVAL_REPEAT="$EVAL_REPEAT" bash scripts/server/eval/run_eval_pipeline.sh \
  "${STAGE4_OUT}/images" \
  "$VAL_DIR" \
  "$NCLASS" "$IPC" resnet_ap

echo ""
echo "============================================================"
echo "[Phase 3 set-level] Done."
echo "  Stage4:           $STAGE4_OUT"
echo "  Set-level report: ${STAGE4_OUT}/set_level_selection_report.json"
echo "  Repr report:      ${STAGE4_OUT}/representativeness_report.json"
echo "  Baseline (HDBSCAN + medoid, 3 repeats): 62.33% ± 1.47"
echo "============================================================"
