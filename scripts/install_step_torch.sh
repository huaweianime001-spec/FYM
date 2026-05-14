#!/usr/bin/env bash
# Step 2 only: PyTorch CUDA wheels (~2GB). Run OUTSIDE Cursor to avoid IDE freezes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
echo "==> Installing PyTorch (CUDA 12.4 index). Change URL at pytorch.org if needed."
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[step 2/3 done] Next:  bash scripts/install_step_requirements.sh"
