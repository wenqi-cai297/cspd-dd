#!/usr/bin/env bash
set -euo pipefail

# Run Stage 3 cluster + Stage 4 (multi-candidate) + Eval for IPC=10,20,50.
# Uses existing encode results. Generates 10 candidates per mode and selects best.

ENCODE_DIR="runs/stage3/ImageNette_train/encoded_with_vae"
LORA="runs/stage2/train/ImageNette_train/stabilityai_stable-diffusion-xl-base-1.0/2026-04-14_181645/official_output/checkpoint-7254/pytorch_lora_weights.safetensors"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
NCLASS=10
NUM_CANDIDATES=10
CANDIDATE_BETA=0.5
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for IPC in 10 20 50; do
  echo ""
  echo "############################################################"
  echo "# IPC=$IPC, candidates=$NUM_CANDIDATES"
  echo "############################################################"

  # Stage 3: cluster
  MODES_DIR="runs/stage3/ImageNette_train/ipc${IPC}/modes_kmeans_candidates"
  if [[ -f "${MODES_DIR}/modes_index.json" ]]; then
    echo "[IPC=$IPC] Stage 3: modes found, skipping."
  else
    echo "[IPC=$IPC] Stage 3: clustering..."
    cspd-stage3 cluster \
      --encode-dir "$ENCODE_DIR" \
      --output-dir "$MODES_DIR" \
      --ipc "$IPC"
  fi

  # Stage 4: text2img with candidate selection
  TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
  STAGE4_OUT="runs/stage4/ImageNette_train/ipc${IPC}/lora/candidates${NUM_CANDIDATES}_${TIMESTAMP}"
  echo "[IPC=$IPC] Stage 4: generating ${NUM_CANDIDATES} candidates/mode → $STAGE4_OUT"
  cspd-stage4 generate \
    --modes-dir "$MODES_DIR" \
    --output-dir "$STAGE4_OUT" \
    --lora-weights "$LORA" \
    --visual-mode none \
    --num-candidates "$NUM_CANDIDATES" \
    --candidate-beta "$CANDIDATE_BETA" \
    --candidate-probe-dir "$ENCODE_DIR"

  # Eval
  echo "[IPC=$IPC] Eval (resnet_ap, repeat=3)"
  EVAL_REPEAT=3 bash scripts/eval/run_eval_pipeline.sh \
    "${STAGE4_OUT}/images" \
    "$VAL_DIR" \
    "$NCLASS" "$IPC" resnet_ap

  echo "[IPC=$IPC] Done."
done

echo ""
echo "############################################################"
echo "# All candidate sweep complete."
echo "############################################################"
