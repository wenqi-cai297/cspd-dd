#!/usr/bin/env bash
set -euo pipefail

# Server setup script for CSPD-DD Stage 1.
# Usage:
#   bash scripts/server/setup_cspd_stage1.sh
# Assumptions:
#   - conda is installed
#   - the environment name is cspd_vlm
#   - this script is executed from anywhere after the repo is cloned

ENV_NAME="cspd_vlm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Initialize conda in non-interactive shells.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO_ROOT"
pip install -e .

# Print a quick sanity check so the user can see the CLI is available.
cspd-stage1 --help

echo
echo "[OK] CSPD-DD Stage 1 is installed in conda env: $ENV_NAME"
echo "[OK] Repo root: $REPO_ROOT"
