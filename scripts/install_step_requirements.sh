#!/usr/bin/env bash
# Step 3 only: diffusers stack (no torch). Run after install_step_torch.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
pip install --no-cache-dir -r requirements.txt
echo "[step 3/3 done] Next:  bash scripts/smoke_import.sh"
