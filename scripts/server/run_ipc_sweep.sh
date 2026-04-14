#!/usr/bin/env bash
set -euo pipefail

# Run Stage 3 cluster + Stage 4 generate + Eval for multiple IPC values.
# Assumes Stage 3A encode is already done.
#
# Usage:
#   bash scripts/server/run_ipc_sweep.sh

ENCODE_DIR="runs/stage3/ImageNette_train/ipc10/dino_2026-04-14_141307/encoded"
LORA="runs/stage2/train/ImageNette_train/stabilityai_stable-diffusion-xl-base-1.0/2026-04-14_181645/official_output/checkpoint-7254/pytorch_lora_weights.safetensors"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
NCLASS=10
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for IPC in 10 20 50; do
  echo ""
  echo "############################################################"
  echo "# IPC=$IPC"
  echo "############################################################"

  # Stage 3: cluster
  MODES_DIR="runs/stage3/ImageNette_train/ipc${IPC}/modes_kmeans"
  echo "[IPC=$IPC] Stage 3 cluster → $MODES_DIR"
  cspd-stage3 cluster \
    --encode-dir "$ENCODE_DIR" \
    --output-dir "$MODES_DIR" \
    --ipc "$IPC"

  # Stage 4: text2img
  TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
  STAGE4_OUT="runs/stage4/ImageNette_train/ipc${IPC}/lora/${TIMESTAMP}"
  echo "[IPC=$IPC] Stage 4 generate → $STAGE4_OUT"
  cspd-stage4 generate \
    --modes-dir "$MODES_DIR" \
    --output-dir "$STAGE4_OUT" \
    --lora-weights "$LORA" \
    --visual-mode none

  # Eval
  echo "[IPC=$IPC] Eval (resnet_ap, repeat=3)"
  EVAL_REPEAT=3 bash scripts/server/eval/run_eval_pipeline.sh \
    "${STAGE4_OUT}/images" \
    "$VAL_DIR" \
    "$NCLASS" "$IPC" resnet_ap

  echo "[IPC=$IPC] Done."
done

echo ""
echo "############################################################"
echo "# All IPC sweep complete."
echo "############################################################"
