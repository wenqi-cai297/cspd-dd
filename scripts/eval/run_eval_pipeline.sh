#!/usr/bin/env bash
set -euo pipefail

# Train one (or all three) eval classifiers on a distilled dataset and report
# top-1 / top-5 on the real validation set.
#
# Usage:
#   bash scripts/eval/run_eval_pipeline.sh <distilled_dir> <val_dir> <nclass> <ipc> [arch|all]
#
# Output is placed at:
#   runs/eval/<dataset>/ipc<IPC>/<arch>/<stage4_tag>/<eval_timestamp>/eval_<arch>.json
#
# where:
#   <dataset>     is parsed out of the distilled_dir path (the segment right
#                 after "runs/stage4/"), falling back to "unknown_dataset".
#   <stage4_tag>  preserves the full Stage 4 lineage (everything under
#                 "runs/stage4/<dataset>/ipc<IPC>/" with slashes replaced by "__",
#                 so for example
#                   runs/stage4/ImageNette_train/ipc10/lora/pipeline_TS/gen_seed42/images
#                 ->
#                   runs/eval/ImageNette_train/ipc10/resnet_ap/lora__pipeline_TS__gen_seed42/<ts>/
#                 ).
#
# Examples:
#   # ImageNette, IPC=10, single architecture
#   bash scripts/eval/run_eval_pipeline.sh \
#     runs/stage4/.../images \
#     /media/4T_HDD/cai/datasets/ImageNette/val \
#     10 10 resnet_ap
#
#   # All three architectures on the same distilled dataset
#   bash scripts/eval/run_eval_pipeline.sh \
#     runs/stage4/.../images \
#     /media/4T_HDD/cai/datasets/ImageNette/val \
#     10 10 all
#
# Environment:
#   CSPD_ENV_NAME=cspd-dd
#   EVAL_REPEAT=3                    # independent runs per arch
#   EVAL_SIZE=224                    # image resolution
#   EVAL_BATCH_SIZE=64
#   EVAL_SEED=0
#   EVAL_SAVE_DIR=<path>             # explicit override for the computed SAVE_DIR

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/eval/run_eval_pipeline.sh <distilled_dir> <val_dir> <nclass> <ipc> [arch|all]"
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
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

# --- Derive dataset label + stage4 tag from distilled_dir ---
# Accept both "<...>/runs/stage4/<ds>/ipc<IPC>/<rest>/images" and the same
# without the trailing /images.
REL_DIR="${DISTILLED_DIR%/}"
if [[ "$(basename "$REL_DIR")" == "images" ]]; then
  REL_DIR="$(dirname "$REL_DIR")"
fi

DATASET_LABEL=""
STAGE4_TAG=""
if [[ "$REL_DIR" == *"runs/stage4/"* ]]; then
  SUFFIX="${REL_DIR##*runs/stage4/}"
  DATASET_LABEL="${SUFFIX%%/*}"
  REST="${SUFFIX#*/}"
  case "$REST" in
    ipc*/*)
      STAGE4_TAG="${REST#ipc*/}"
      ;;
    *)
      STAGE4_TAG="$REST"
      ;;
  esac
  STAGE4_TAG="${STAGE4_TAG//\//__}"
fi

if [[ -z "$DATASET_LABEL" ]]; then
  DATASET_LABEL="unknown_dataset"
fi
if [[ -z "$STAGE4_TAG" ]]; then
  STAGE4_TAG="$(basename "$REL_DIR")"
fi

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
SAVE_DIR="${EVAL_SAVE_DIR:-runs/eval/${DATASET_LABEL}/ipc${IPC}/${ARCH}/${STAGE4_TAG}/${TIMESTAMP}}"

echo "============================================================"
echo "[Eval] Distilled dataset evaluation"
echo "  distilled_dir: $DISTILLED_DIR"
echo "  val_dir:       $VAL_DIR"
echo "  dataset:       $DATASET_LABEL"
echo "  stage4_tag:    $STAGE4_TAG"
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
