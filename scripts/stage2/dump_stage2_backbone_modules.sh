#!/usr/bin/env bash
set -euo pipefail

# Dump real Stage 2 backbone module names to text artifacts under runs/stage2/inspect/.
# Usage:
#   bash scripts/stage2/dump_stage2_backbone_modules.sh [backbone_name] [component] [extra args...]
# Example:
#   bash scripts/stage2/dump_stage2_backbone_modules.sh black-forest-labs/FLUX.1-Kontext-dev transformer --local-files-only

BACKBONE_NAME="${1:-black-forest-labs/FLUX.1-Kontext-dev}"
COMPONENT="${2:-transformer}"
shift $(( $# >= 2 ? 2 : $# )) || true
EXTRA_ARGS=("$@")
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKBONE_SLUG="$(echo "$BACKBONE_NAME" | tr '/ ' '__' | tr -cd '[:alnum:]_.-')"
TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"
OUTPUT_DIR="runs/stage2/inspect/${BACKBONE_SLUG}/${TIMESTAMP}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

CMD=(
  cspd-stage2 dump-modules
  --backbone-name "$BACKBONE_NAME"
  --load-backbone
  --component "$COMPONENT"
  --output-dir "$OUTPUT_DIR"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] backbone_name:      $BACKBONE_NAME"
echo "[INFO] component:          $COMPONENT"
echo "[INFO] inspect_output_dir: $OUTPUT_DIR"

"${CMD[@]}"

echo "[INFO] Dump complete."
echo "[INFO] summary:            $OUTPUT_DIR/dump_summary.json"
echo "[INFO] top-level:          $OUTPUT_DIR/pipeline_top_level_components.txt"
echo "[INFO] raw children:       $OUTPUT_DIR/pipeline_named_children.txt"
echo "[INFO] focus children:     $OUTPUT_DIR/${COMPONENT}_named_children.txt"
echo "[INFO] focus modules:      $OUTPUT_DIR/${COMPONENT}_named_modules.txt"
echo "[INFO] filtered dir:       $OUTPUT_DIR/filtered"
