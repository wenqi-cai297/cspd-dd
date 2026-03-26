#!/usr/bin/env bash
set -euo pipefail

# Run Stage 1 with the mock backend for quick plumbing checks.
# Usage:
#   bash scripts/server/run_stage1_mock.sh /path/to/dataset /path/to/output_dir

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/server/run_stage1_mock.sh <dataset_root> <output_dir>"
  exit 1
fi

DATASET_ROOT="$1"
OUTPUT_DIR="$2"
ENV_NAME="cspd-dd"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"

cspd-stage1 run \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --backend mock
