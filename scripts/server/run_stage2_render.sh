#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper.
# Canonical rendering is now classified as part of Stage 1.
# Prefer:
#   bash scripts/server/run_stage1_render.sh <normalized_attributes_jsonl> [renderer_version]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_stage1_render.sh" "$@"
