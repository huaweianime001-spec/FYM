#!/usr/bin/env bash
# Full install (all steps). Prefer running OUTSIDE Cursor — see README if the IDE freezes.
#
# Resume-friendly alternative:
#   bash scripts/install_step_venv.sh
#   bash scripts/install_step_torch.sh
#   bash scripts/install_step_requirements.sh
#   bash scripts/smoke_import.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

bash "$ROOT/scripts/install_step_venv.sh"
bash "$ROOT/scripts/install_step_torch.sh"
bash "$ROOT/scripts/install_step_requirements.sh"
bash "$ROOT/scripts/smoke_import.sh"

echo "Done. Activate later with:  source $(pwd)/.venv/bin/activate"
