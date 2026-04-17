#!/usr/bin/env bash
set -euo pipefail

# HDBSCAN + medoid baseline, 3x3 protocol:
#   - Stage 4 runs 3 times with master seeds {42, 123, 456}
#   - Each run: all IPC*num_classes images share the run's master seed
#     (per-round shared-seed generation; no +mode_idx offset)
#   - Each resulting dataset is eval'd with 3 independent seeds (EVAL_REPEAT=3)
#   - Total: 9 accuracy numbers (3 generation seeds x 3 eval repeats)
#
# Usage:
#   bash scripts/server/run_baseline_3x3.sh
#
# Environment:
#   BASELINE_SEEDS="42 123 456"   (override to change master seeds)
#   EVAL_REPEAT=3

MODES_DIR="runs/stage3/ImageNette_train/ipc10/hdbscan_medoid"
LORA="runs/stage2/train/ImageNette_train/stabilityai_stable-diffusion-xl-base-1.0/2026-04-14_181645/official_output/checkpoint-7254/pytorch_lora_weights.safetensors"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
NCLASS=10
IPC=10

SEEDS="${BASELINE_SEEDS:-42 123 456}"
EVAL_REPEAT="${EVAL_REPEAT:-3}"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ ! -f "${MODES_DIR}/modes_index.json" ]]; then
  echo "[ERROR] HDBSCAN modes not found at ${MODES_DIR}"
  exit 1
fi
if [[ ! -f "$LORA" ]]; then
  echo "[ERROR] LoRA weights not found: $LORA"
  exit 1
fi

TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
RUN_ROOT="runs/stage4/ImageNette_train/ipc${IPC}/lora/baseline_3x3_${TIMESTAMP}"
SUMMARY_FILE="${RUN_ROOT}/summary.txt"
mkdir -p "$RUN_ROOT"

echo "============================================================"
echo "[baseline 3x3] HDBSCAN + medoid, per-round shared-seed protocol"
echo "  modes:      $MODES_DIR"
echo "  lora:       $LORA"
echo "  seeds:      $SEEDS"
echo "  eval rep:   $EVAL_REPEAT per generation seed"
echo "  run root:   $RUN_ROOT"
echo "============================================================"

echo "# baseline 3x3 protocol" > "$SUMMARY_FILE"
echo "# modes=$MODES_DIR" >> "$SUMMARY_FILE"
echo "# lora=$LORA" >> "$SUMMARY_FILE"
echo "# per-round shared master seed; eval_repeat=$EVAL_REPEAT" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for SEED in $SEEDS; do
  echo ""
  echo "############################################################"
  echo "# Generation seed=$SEED"
  echo "############################################################"

  STAGE4_OUT="${RUN_ROOT}/gen_seed${SEED}"
  mkdir -p "$STAGE4_OUT"

  cspd-stage4 generate \
    --modes-dir "$MODES_DIR" \
    --output-dir "$STAGE4_OUT" \
    --lora-weights "$LORA" \
    --model-name "$MODEL_NAME" \
    --visual-mode none \
    --resolution 512 \
    --guidance-scale 7.5 \
    --num-inference-steps 50 \
    --seed "$SEED"

  echo ""
  echo "# Eval for gen_seed=$SEED (repeat=$EVAL_REPEAT)"
  EVAL_REPEAT="$EVAL_REPEAT" bash scripts/server/eval/run_eval_pipeline.sh \
    "${STAGE4_OUT}/images" \
    "$VAL_DIR" \
    "$NCLASS" "$IPC" resnet_ap
done

echo ""
echo "============================================================"
echo "[baseline 3x3] Aggregating..."
echo "============================================================"

python - <<PYEOF | tee -a "$SUMMARY_FILE"
import json, glob, os, statistics

run_root = "$RUN_ROOT"
seeds = "$SEEDS".split()

per_seed = []  # list of (seed, mean, std, runs)
all_acc1 = []
for s in seeds:
    gen_dir = os.path.join(run_root, f"gen_seed{s}")
    # The eval dir is saved under runs/eval/<ts>_ipc10_resnet_ap; find the most
    # recent one whose distilled_dir points to this gen dir.
    cand = []
    for p in glob.glob("runs/eval/*_ipc10_resnet_ap/eval_resnet_ap.json"):
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
    per_seed.append((s, d["mean_best_acc1"], d["std_best_acc1"], runs_acc, p))
    all_acc1.extend(runs_acc)

print("")
print("# Per-generation-seed results")
for s, m, sd, runs, p in per_seed:
    print(f"  gen_seed={s:>4}: mean={m:.2f}  std={sd:.2f}  runs={runs}  (eval={p})")

if all_acc1:
    print("")
    print("# Aggregate across all 9 numbers")
    print(f"  grand_mean = {statistics.mean(all_acc1):.2f}")
    print(f"  grand_std  = {statistics.pstdev(all_acc1):.2f}")
    print(f"  n          = {len(all_acc1)}")
    print(f"  min        = {min(all_acc1):.2f}")
    print(f"  max        = {max(all_acc1):.2f}")
PYEOF

echo ""
echo "============================================================"
echo "[baseline 3x3] Done. Summary: $SUMMARY_FILE"
echo "============================================================"
