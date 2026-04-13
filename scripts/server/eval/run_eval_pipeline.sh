#!/usr/bin/env bash
set -euo pipefail

# Evaluate a distilled dataset by training classifiers and testing on real validation set.
#
# Usage:
#   bash scripts/server/eval/run_eval_pipeline.sh <distilled_dir> <val_dir> <nclass> <ipc> [arch|all]
#
# Examples:
#   # ImageNette, IPC=10, all architectures
#   bash scripts/server/eval/run_eval_pipeline.sh \
#     runs/stage4/ImageNette_train/ipc10/.../images \
#     /media/4T_HDD/cai/datasets/ImageNette/val \
#     10 10 all
#
#   # Single architecture
#   bash scripts/server/eval/run_eval_pipeline.sh \
#     runs/stage4/.../images \
#     /media/4T_HDD/cai/datasets/ImageNette/val \
#     10 10 convnet
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   EVAL_REPEAT=3                    # independent runs per arch
#   EVAL_SIZE=224                    # image resolution
#   EVAL_BATCH_SIZE=64
#   EVAL_SEED=0

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/server/eval/run_eval_pipeline.sh <distilled_dir> <val_dir> <nclass> <ipc> [arch|all]"
  exit 1
fi

DISTILLED_DIR="$1"
VAL_DIR="$2"
NCLASS="$3"
IPC="$4"
ARCH="${5:-all}"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
REPEAT="${EVAL_REPEAT:-3}"
SIZE="${EVAL_SIZE:-224}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
SEED="${EVAL_SEED:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ ! -d "$DISTILLED_DIR" ]]; then
  echo "[ERROR] Distilled dataset not found: $DISTILLED_DIR"
  exit 1
fi
if [[ ! -d "$VAL_DIR" ]]; then
  echo "[ERROR] Validation dataset not found: $VAL_DIR"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
SAVE_DIR="runs/eval/${TIMESTAMP}_ipc${IPC}_${ARCH}"

echo "============================================================"
echo "[Eval] Distilled Dataset Evaluation"
echo "  distilled_dir: $DISTILLED_DIR"
echo "  val_dir:       $VAL_DIR"
echo "  nclass:        $NCLASS"
echo "  ipc:           $IPC"
echo "  arch:          $ARCH"
echo "  repeat:        $REPEAT"
echo "  size:          $SIZE"
echo "  save_dir:      $SAVE_DIR"
echo "============================================================"

if [[ "$ARCH" == "all" ]]; then
  cspd-eval run-all \
    --distilled-dir "$DISTILLED_DIR" \
    --val-dir "$VAL_DIR" \
    --nclass "$NCLASS" \
    --ipc "$IPC" \
    --size "$SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --repeat "$REPEAT" \
    --save-dir "$SAVE_DIR"
else
  cspd-eval run \
    --distilled-dir "$DISTILLED_DIR" \
    --val-dir "$VAL_DIR" \
    --arch "$ARCH" \
    --nclass "$NCLASS" \
    --ipc "$IPC" \
    --size "$SIZE" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --repeat "$REPEAT" \
    --save-dir "$SAVE_DIR"
fi

echo ""
echo "============================================================"
echo "[Eval] Complete. Results: $SAVE_DIR"
echo "============================================================"
