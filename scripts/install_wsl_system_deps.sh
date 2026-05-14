#!/usr/bin/env bash
# One-time system packages on Ubuntu/WSL so bitsandbytes/triton can compile helpers (Python.h).
# Run once: bash scripts/install_wsl_system_deps.sh

set -euo pipefail
if ! command -v apt-get >/dev/null 2>&1; then
  echo "This script is for Debian/Ubuntu (WSL). On other distros, install python dev headers + gcc yourself."
  exit 1
fi

sudo apt-get update
sudo apt-get install -y python3-dev python3-venv build-essential

# Match venv Python minor if possible (e.g. 3.12)
PY_MINOR="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 12)"
if apt-cache show "python3.${PY_MINOR}-dev" >/dev/null 2>&1; then
  sudo apt-get install -y "python3.${PY_MINOR}-dev" || true
fi

echo "Done. Re-run: bash scripts/smoke_import.sh"
