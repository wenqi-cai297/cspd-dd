#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/stage2/run_sdxl_stage2_official.sh <dataset_root> <stage1_render_records_jsonl> [batch_size] [epochs] [extra args...]"
  exit 1
fi

DATASET_ROOT="$1"
RENDER_INPUT="$2"
shift 2
BATCH_SIZE="${1:-1}"
if [[ $# -gt 0 ]]; then shift; fi
EPOCHS="${1:-1}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

CMD=(
  cspd-stage2 train
  --dataset-root "$DATASET_ROOT"
  --render-input "$RENDER_INPUT"
  --backbone-name stabilityai/stable-diffusion-xl-base-1.0
  --training-parameterization lora
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --resolution 1024
  --sdxl-mixed-precision fp16
  --sdxl-lr-scheduler constant
  --sdxl-report-to none
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

"${CMD[@]}"
