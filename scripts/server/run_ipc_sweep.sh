#!/usr/bin/env bash
set -euo pipefail

# Run Stage 3 (encode + cluster) + Stage 4 generate + Eval for multiple IPC values.
# If encode output already exists, skips re-encoding.
#
# Usage:
#   bash scripts/server/run_ipc_sweep.sh

DATASET_ROOT="/media/4T_HDD/cai/datasets/ImageNette/train"
RENDER_INPUT="runs/stage1/render/ImageNette_train/qwen_local/2026-04-13_111606/records.jsonl"
ENCODE_DIR="runs/stage3/ImageNette_train/encoded"
LORA="runs/stage2/train/ImageNette_train/stabilityai_stable-diffusion-xl-base-1.0/2026-04-14_181645/official_output/checkpoint-7254/pytorch_lora_weights.safetensors"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
NCLASS=10
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Stage 3A: Encode (skip if already exists)
if [[ -f "${ENCODE_DIR}/dino_embeds.pt" && -f "${ENCODE_DIR}/encode_index.json" ]]; then
  echo "[Stage 3A] Encode output already exists at ${ENCODE_DIR}, skipping."
else
  echo "[Stage 3A] Encoding dataset..."
  cspd-stage3 encode \
    --dataset-root "$DATASET_ROOT" \
    --render-input "$RENDER_INPUT" \
    --output-dir "$ENCODE_DIR"
fi

for IPC in 10 20 50; do
  echo ""
  echo "############################################################"
  echo "# IPC=$IPC"
  echo "############################################################"

  # Stage 3B: cluster
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
