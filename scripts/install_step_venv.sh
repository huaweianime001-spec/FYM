#!/usr/bin/env bash
# Step 1 only: create venv + upgrade pip (fast). Safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ ! -d .venv ]] && python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
echo "[step 1/3 done] Next in EXTERNAL terminal:  bash scripts/install_step_torch.sh"
