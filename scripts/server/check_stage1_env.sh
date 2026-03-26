#!/usr/bin/env bash
set -euo pipefail

# Check and prepare the server environment for CSPD-DD Stage 1.
# Usage:
#   bash scripts/server/check_stage1_env.sh

ENV_NAME="cspd_vlm"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

echo "[INFO] Python"
python --version

echo "[INFO] Torch / CUDA"
python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_device_count', torch.cuda.device_count())"

echo "[INFO] Installing runtime dependencies"
pip install transformers pillow accelerate sentencepiece
pip install -e .

echo "[INFO] Transformers"
python -c "import transformers; print(transformers.__version__)"

echo "[INFO] PIL"
python -c "from PIL import Image; print('PIL ok')"

echo "[OK] Stage 1 environment check complete."
