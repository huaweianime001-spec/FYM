#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
echo "=== venv torch ==="
if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
  python - <<'PY' || true
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)
PY
else
  echo "no .venv"
fi
echo "=== model (WAN22_MODEL_ID or Hub default) ==="
M="${WAN22_MODEL_ID:-}"
if [[ -z "$M" ]]; then
  M="${HOME}/models/Wan2.2-TI2V-5B-Diffusers"
fi
if [[ -f "$M/model_index.json" ]]; then
  echo "found model_index.json under $M"
else
  echo "no local snapshot at $M (set WAN22_MODEL_ID or use HF cache for Wan-AI/Wan2.2-TI2V-5B-Diffusers)"
fi
echo "=== outputs ==="
ls -la outputs 2>/dev/null || echo "no outputs dir"
