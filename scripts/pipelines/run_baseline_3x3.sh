#!/usr/bin/env bash
set -euo pipefail

# 3x3 measurement protocol for the HDBSCAN + medoid baseline.
#
#   For each seed in {42, 123, 456}:
#     Stage 3B cluster (--seed $SEED)  ->  Stage 4 text2img (--seed $SEED)
#     ->  Eval with EVAL_REPEAT independent runs
#
#   Aggregation:
#     per-seed = max(EVAL_REPEAT eval repeats)  # best-of-3
#     overall  = mean / std / min / max across the 3 per-seed bests
#
# Assumes Stage 1 (render records) + Stage 2 (LoRA checkpoint) + Stage 3A
# (DINOv2 encode) are already on disk. Run
#   bash scripts/pipelines/run_full_pipeline.sh <train_root>
# first if not.
#
# Usage:
#   bash scripts/pipelines/run_baseline_3x3.sh <train_root> [val_root] [nclass]
#
# Positional args:
#   train_root  ImageFolder training split root (required, used only to derive
#               the dataset label and locate the existing Stage 1 / 2 / 3 runs).
#   val_root    Validation split root (optional; default <parent>/val).
#   nclass      Number of classes (optional; default = subdir count).
#
# Environment overrides:
#   CSPD_ENV_NAME=cspd-dd
#   IPC=10                                 # single IPC value
#   BASELINE_SEEDS="42 123 456"            # paired across Stage 3 + Stage 4
#   EVAL_REPEAT=3
#   LORA_WEIGHTS=<path>                    # explicit override. If unset,
#                                          # the newest pytorch_lora_weights.safetensors
#                                          # under the dataset-specific
#                                          # Stage 2 train dir is used.
#   ENCODE_DIR=<path>                      # explicit override of auto-detect

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/pipelines/run_baseline_3x3.sh <train_root> [val_root] [nclass]"
  exit 1
fi

TRAIN_ROOT="$1"
VAL_ROOT="${2:-}"
NCLASS_ARG="${3:-}"

if [[ ! -d "$TRAIN_ROOT" ]]; then
  echo "[ERROR] train_root does not exist: $TRAIN_ROOT"
  exit 1
fi

if [[ -z "$VAL_ROOT" ]]; then
  VAL_ROOT="$(dirname "$TRAIN_ROOT")/val"
fi
if [[ ! -d "$VAL_ROOT" ]]; then
  echo "[ERROR] val_root does not exist: $VAL_ROOT"
  exit 1
fi

NCLASS="${NCLASS_ARG:-$(find "$TRAIN_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')}"

ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
IPC="${IPC:-10}"
SEEDS="${BASELINE_SEEDS:-42 123 456}"
EVAL_REPEAT="${EVAL_REPEAT:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

# Dataset label
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

# --- Resolve ENCODE_DIR ---
if [[ -z "${ENCODE_DIR:-}" ]]; then
  ENCODE_DIR="runs/stage3/${DATASET_LABEL}/encoded"
fi
if [[ ! -f "${ENCODE_DIR}/dino_embeds.pt" ]]; then
  echo "[ERROR] Stage 3A encode output not found: ${ENCODE_DIR}/dino_embeds.pt"
  echo "       Run scripts/pipelines/run_full_pipeline.sh first, or set ENCODE_DIR."
  exit 1
fi

# --- Resolve LoRA checkpoint ---
# Pick the newest pytorch_lora_weights.safetensors on disk under the
# dataset-specific Stage 2 train dir. Accepts either the trainer's final
# `official_output/pytorch_lora_weights.safetensors` or any intermediate
# `official_output/checkpoint-N/pytorch_lora_weights.safetensors`.
# LORA_WEIGHTS=<path> in env overrides this entirely.
if [[ -z "${LORA_WEIGHTS:-}" ]]; then
  STAGE2_TRAIN_DIR="runs/stage2/train/${DATASET_LABEL}/${BACKBONE_SLUG}"
  if [[ ! -d "$STAGE2_TRAIN_DIR" ]]; then
    echo "[ERROR] No Stage 2 training runs under $STAGE2_TRAIN_DIR"
    echo "       Run scripts/pipelines/run_full_pipeline.sh first, or set LORA_WEIGHTS."
    exit 1
  fi
  LORA_WEIGHTS="$(
    find "$STAGE2_TRAIN_DIR" -type f -name 'pytorch_lora_weights.safetensors' -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -1 | cut -d' ' -f2-
  )"
fi
if [[ -z "${LORA_WEIGHTS:-}" || ! -f "$LORA_WEIGHTS" ]]; then
  echo "[ERROR] Could not locate a LoRA checkpoint under $STAGE2_TRAIN_DIR."
  echo "       Pass LORA_WEIGHTS=<path> explicitly."
  exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_ROOT="runs/stage4/${DATASET_LABEL}/ipc${IPC}/lora/baseline_3x3_${TIMESTAMP}"
SUMMARY_FILE="${RUN_ROOT}/summary.txt"
mkdir -p "$RUN_ROOT"

echo "============================================================"
echo " 3x3 baseline measurement"
echo "  dataset:   $DATASET_LABEL"
echo "  val_root:  $VAL_ROOT"
echo "  nclass:    $NCLASS"
echo "  IPC:       $IPC"
echo "  seeds:     $SEEDS  (shared between Stage 3 and Stage 4)"
echo "  encode:    $ENCODE_DIR"
echo "  LoRA:      $LORA_WEIGHTS"
echo "  eval rep:  $EVAL_REPEAT per seed"
echo "  run root:  $RUN_ROOT"
echo "============================================================"

{
  echo "# 3x3 baseline (Stage 3 + Stage 4 + Eval, paired seeds)"
  echo "# dataset=$DATASET_LABEL  IPC=$IPC"
  echo "# encode=$ENCODE_DIR"
  echo "# lora=$LORA_WEIGHTS"
  echo "# within-round: image i uses seed + mode_idx"
  echo "# eval_repeat=$EVAL_REPEAT  arch=resnet_ap"
  echo ""
} > "$SUMMARY_FILE"

for SEED in $SEEDS; do
  echo ""
  echo "############################################################"
  echo "# Seed=$SEED   (Stage 3 cluster -> Stage 4 generate -> Eval)"
  echo "############################################################"

  MODES_DIR="runs/stage3/${DATASET_LABEL}/ipc${IPC}/hdbscan_medoid_s${SEED}"
  if [[ -f "${MODES_DIR}/modes_index.json" ]]; then
    echo "[seed=$SEED] Stage 3B modes exist at $MODES_DIR — skipping cluster."
  else
    echo "[seed=$SEED] Stage 3B clustering with --seed $SEED"
    cspd-stage3 cluster \
      --encode-dir "$ENCODE_DIR" \
      --output-dir "$MODES_DIR" \
      --ipc "$IPC" \
      --seed "$SEED"
  fi

  STAGE4_OUT="${RUN_ROOT}/gen_seed${SEED}"
  mkdir -p "$STAGE4_OUT"
  echo "[seed=$SEED] Stage 4 text2img -> $STAGE4_OUT"
  cspd-stage4 generate \
    --modes-dir "$MODES_DIR" \
    --output-dir "$STAGE4_OUT" \
    --lora-weights "$LORA_WEIGHTS" \
    --model-name "$BACKBONE_NAME" \
    --visual-mode none \
    --resolution 512 \
    --guidance-scale 7.5 \
    --num-inference-steps 50 \
    --seed "$SEED"

  echo "[seed=$SEED] Eval (resnet_ap, repeat=$EVAL_REPEAT)"
  EVAL_REPEAT="$EVAL_REPEAT" bash scripts/eval/run_eval_pipeline.sh \
    "${STAGE4_OUT}/images" \
    "$VAL_ROOT" \
    "$NCLASS" "$IPC" resnet_ap
done

echo ""
echo "============================================================"
echo " Aggregating..."
echo "============================================================"

python - <<PYEOF | tee -a "$SUMMARY_FILE"
import json, glob, os, statistics

run_root = "$RUN_ROOT"
seeds = "$SEEDS".split()
ipc = $IPC

# Find eval JSON files that match each gen_seed directory under run_root.
# Structure as of 2026-04-18:
#   runs/eval/<dataset>/ipc<IPC>/<arch>/<stage4_tag>/<eval_ts>/eval_<arch>.json
# The eval JSON's "distilled_dir" field carries the original Stage 4 images
# path; that's what we match on (so the script keeps working even if the
# on-disk convention shifts again).
eval_candidates = glob.glob(
    f"runs/eval/**/ipc{ipc}/resnet_ap/**/eval_resnet_ap.json", recursive=True,
)
# Also accept the older flat layout for back-compat on rigs that still have
# old eval dirs around.
eval_candidates += glob.glob(f"runs/eval/*_ipc{ipc}_resnet_ap/eval_resnet_ap.json")

per_seed_best = []  # (seed, best_acc1, runs, eval_path)
for s in seeds:
    gen_dir = os.path.join(run_root, f"gen_seed{s}")
    cand = []
    for p in eval_candidates:
        try:
            d = json.load(open(p))
            if d.get("distilled_dir", "").startswith(gen_dir):
                cand.append((os.path.getmtime(p), p, d))
        except Exception:
            pass
    if not cand:
        print(f"[WARN] no eval found for gen_seed={s}")
        continue
    cand.sort()
    _, p, d = cand[-1]
    runs_acc = [r.get("best_acc1") for r in d.get("runs", [])]
    best = max(runs_acc) if runs_acc else None
    per_seed_best.append((s, best, runs_acc, p))

print("")
print("# Per-generation-seed results (best-of-3 across eval repeats)")
for s, best, runs, p in per_seed_best:
    print(f"  gen_seed={s:>4}: best={best:.2f}  runs={runs}  (eval={p})")

bests = [b for _, b, _, _ in per_seed_best if b is not None]
if bests:
    print("")
    print("# Aggregate across per-seed bests (n=3)")
    print(f"  mean = {statistics.mean(bests):.2f}")
    if len(bests) > 1:
        print(f"  std  = {statistics.pstdev(bests):.2f}")
    print(f"  min  = {min(bests):.2f}")
    print(f"  max  = {max(bests):.2f}")
PYEOF

echo ""
echo "============================================================"
echo " 3x3 baseline complete. Summary: $SUMMARY_FILE"
echo "============================================================"
