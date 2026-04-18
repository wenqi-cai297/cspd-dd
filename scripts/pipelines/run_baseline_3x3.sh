#!/usr/bin/env bash
set -euo pipefail

# HDBSCAN + medoid baseline, full 3x3 protocol (Stage 3 + Stage 4 paired seeds):
#   - For each seed in {42, 123, 456}:
#       * Stage 3 cluster with --seed: produces its own modes_hdbscan_s<seed>
#         (HDBSCAN itself is deterministic, but the PCA preprocessing and the
#         K-Means fallback / sub-clustering used when HDBSCAN finds <IPC modes
#         are seeded, so different seeds yield different medoid selections
#         for the same real images).
#       * Stage 4 generate with --seed: all IPC*num_classes images in this
#         round share the master seed (per-round shared-seed convention,
#         no +mode_idx offset).
#       * Eval with 3 independent eval seeds (EVAL_REPEAT=3).
#   - Aggregation: for each seed take max(3 eval repeats) = best-of-3,
#     then report mean / std / min / max across the 3 per-seed bests.
#
# Usage:
#   bash scripts/pipelines/run_baseline_3x3.sh           # IPC=10 (default)
#   IPC=20 bash scripts/pipelines/run_baseline_3x3.sh   # IPC=20
#   IPC=50 bash scripts/pipelines/run_baseline_3x3.sh   # IPC=50
#
# Environment:
#   IPC=10                        (images per class; 10/20/50 recommended)
#   BASELINE_SEEDS="42 123 456"   (override to change seeds; paired across stages)
#   EVAL_REPEAT=3

# Legacy directory name from when we also encoded VAE latents. Only
# dino_embeds.pt + encode_index.json are consumed now.
ENCODE_DIR="runs/stage3/ImageNette_train/encoded_with_vae"
LORA="runs/stage2/train/ImageNette_train/stabilityai_stable-diffusion-xl-base-1.0/2026-04-14_181645/official_output/checkpoint-7254/pytorch_lora_weights.safetensors"
VAL_DIR="/media/4T_HDD/cai/datasets/ImageNette/val"
MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
NCLASS=10
IPC="${IPC:-10}"

SEEDS="${BASELINE_SEEDS:-42 123 456}"
EVAL_REPEAT="${EVAL_REPEAT:-3}"
ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if [[ ! -d "$ENCODE_DIR" ]]; then
  echo "[ERROR] Stage 3 encode dir not found: $ENCODE_DIR"
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
echo "[baseline 3x3] HDBSCAN + medoid, full 3x3 (Stage 3 + Stage 4 paired)"
echo "  IPC:        $IPC"
echo "  encode:     $ENCODE_DIR"
echo "  lora:       $LORA"
echo "  seeds:      $SEEDS  (shared across Stage 3 and Stage 4)"
echo "  eval rep:   $EVAL_REPEAT per seed"
echo "  run root:   $RUN_ROOT"
echo "============================================================"

echo "# baseline 3x3 protocol (Stage 3 + Stage 4 + Eval, paired seeds)" > "$SUMMARY_FILE"
echo "# IPC=$IPC" >> "$SUMMARY_FILE"
echo "# encode=$ENCODE_DIR" >> "$SUMMARY_FILE"
echo "# lora=$LORA" >> "$SUMMARY_FILE"
echo "# within-round: image i uses seed + mode_idx; Stage 3 re-clustered per seed; eval_repeat=$EVAL_REPEAT" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for SEED in $SEEDS; do
  echo ""
  echo "############################################################"
  echo "# Seed=$SEED   (Stage 3 cluster -> Stage 4 generate -> Eval)"
  echo "############################################################"

  MODES_DIR="runs/stage3/ImageNette_train/ipc${IPC}/hdbscan_medoid_s${SEED}"
  if [[ -f "${MODES_DIR}/modes_index.json" ]]; then
    echo "[seed=$SEED] Stage 3: modes already exist at $MODES_DIR, skipping cluster"
  else
    echo "[seed=$SEED] Stage 3: clustering with --seed $SEED"
    cspd-stage3 cluster \
      --encode-dir "$ENCODE_DIR" \
      --output-dir "$MODES_DIR" \
      --ipc "$IPC" \
      --seed "$SEED"
  fi

  STAGE4_OUT="${RUN_ROOT}/gen_seed${SEED}"
  mkdir -p "$STAGE4_OUT"

  echo "[seed=$SEED] Stage 4: generating into $STAGE4_OUT"
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

  echo "[seed=$SEED] Eval (repeat=$EVAL_REPEAT)"
  EVAL_REPEAT="$EVAL_REPEAT" bash scripts/eval/run_eval_pipeline.sh \
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
ipc = $IPC

# Aggregation rule: for each gen_seed, take MAX over its 3 eval repeats
# (best-of-3). Then compute mean/std/min/max across the 3 per-seed bests.
per_seed_best = []  # list of (seed, best_of_3, runs, eval_path)
for s in seeds:
    gen_dir = os.path.join(run_root, f"gen_seed{s}")
    # Eval dirs are under runs/eval/<ts>_ipc<IPC>_resnet_ap; match by distilled_dir
    # prefix and take the most recent if multiple.
    cand = []
    for p in glob.glob(f"runs/eval/*_ipc{ipc}_resnet_ap/eval_resnet_ap.json"):
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
echo "[baseline 3x3] Done. Summary: $SUMMARY_FILE"
echo "============================================================"
