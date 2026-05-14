#!/usr/bin/env bash
# Quick import check (seconds). Does not download models.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .venv/bin/activate ]]; then
  echo "ERROR: No virtualenv at $ROOT/.venv"
  echo "  Run:  bash scripts/install_step_venv.sh"
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

if ! python -c "import sys; sys.exit(0)" 2>/dev/null; then
  echo "ERROR: python in .venv is broken."
  exit 1
fi

python "$ROOT/scripts/smoke_check.py"
echo "Smoke OK."
