#!/usr/bin/env bash
set -euo pipefail

# Check and prepare the server environment for CSPD Stage 2 SDXL official-diffusers runs.
# Usage:
#   bash scripts/server/check_stage2_sdxl_env.sh [optional_explicit_sdxl_script_path]

ENV_NAME="${CSPD_ENV_NAME:-cspd-dd}"
REQUESTED_SCRIPT="${1:-${CSPD_STAGE2_SDXL_SCRIPT:-}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

resolve_sdxl_script() {
  local requested="$1"
  local candidate=""

  if [[ -n "$requested" && -f "$requested" ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  if [[ -n "${DIFFUSERS_REPO_ROOT:-}" ]]; then
    candidate="${DIFFUSERS_REPO_ROOT}/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if [[ -n "${DIFFUSERS_HOME:-}" ]]; then
    candidate="${DIFFUSERS_HOME}/examples/text_to_image/train_text_to_image_lora_sdxl.py"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi

  if command -v train_text_to_image_lora_sdxl.py >/dev/null 2>&1; then
    command -v train_text_to_image_lora_sdxl.py
    return 0
  fi

  return 1
}

echo "[INFO] Python"
python --version

echo "[INFO] accelerate / diffusers imports"
python - <<'PY'
import importlib.util
modules = ["torch", "diffusers", "accelerate", "transformers", "peft"]
for name in modules:
    print(name, importlib.util.find_spec(name) is not None)
PY

echo "[INFO] editable install"
pip install -e .

echo "[INFO] version checks"
python - <<'PY'
import accelerate, diffusers, transformers
print('accelerate', accelerate.__version__)
print('diffusers', diffusers.__version__)
print('transformers', transformers.__version__)
PY

echo "[INFO] Torch / CUDA"
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f'cuda_device_{idx}', torch.cuda.get_device_name(idx))
PY

echo "[INFO] accelerate CLI"
command -v accelerate
accelerate env || true

echo "[INFO] cspd-stage2 CLI"
command -v cspd-stage2
cspd-stage2 --help >/dev/null

if RESOLVED_SCRIPT="$(resolve_sdxl_script "$REQUESTED_SCRIPT")"; then
  echo "[INFO] resolved_sdxl_script: $RESOLVED_SCRIPT"
else
  echo "[WARN] Could not resolve train_text_to_image_lora_sdxl.py"
  echo "[WARN] Set one of:"
  echo "[WARN]   export CSPD_STAGE2_SDXL_SCRIPT=/path/to/train_text_to_image_lora_sdxl.py"
  echo "[WARN]   export DIFFUSERS_REPO_ROOT=/path/to/diffusers"
  echo "[WARN]   bash scripts/server/check_stage2_sdxl_env.sh /explicit/path/to/train_text_to_image_lora_sdxl.py"
fi

echo "[OK] Stage 2 SDXL environment check complete."
