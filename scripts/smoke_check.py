#!/usr/bin/env python3
"""Import smoke test with actionable error messages."""
from __future__ import annotations

import sys
import traceback


def main() -> int:
    print("Python:", sys.version.split()[0])

    try:
        import torch
    except ImportError as e:
        print("FAIL: PyTorch is not installed.\n  Fix: bash scripts/install_step_torch.sh")
        print(" ", e)
        return 1

    print("torch:", torch.__version__, "| cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    else:
        print(
            "NOTE: CUDA not visible (WSL needs NVIDIA drivers; CPU-only torch is enough for this import test)."
        )

    try:
        import diffusers
        from diffusers import WanImageToVideoPipeline
    except ImportError as e:
        print(
            "FAIL: diffusers / Wan pipeline import failed.\n"
            "  Fix: bash scripts/install_step_requirements.sh\n"
            f"  ImportError: {e}"
        )
        return 1
    except Exception as e:
        tb = traceback.format_exc()
        if "Python.h" in tb or "fatal error" in tb.lower():
            py = f"{sys.version_info.major}.{sys.version_info.minor}"
            print(
                "FAIL: a native extension tried to compile but Python development headers are missing.\n"
                "  (Usually: diffusers → bitsandbytes → triton compiles a small CUDA helper.)\n"
                "  Fix on Ubuntu WSL:\n"
                "    sudo apt-get update\n"
                "    sudo apt-get install -y python3-dev python3-venv build-essential\n"
                f"  If you use Python {py} only, also: sudo apt-get install -y python{py}-dev\n"
                "  Then re-run: bash scripts/smoke_import.sh"
            )
            return 1
        if "bitsandbytes" in tb.lower() or "triton" in tb.lower():
            print(
                "FAIL: bitsandbytes / triton failed while importing diffusers.\n"
                "  Try installing system build deps (see Python.h message above), or:\n"
                "    sudo apt-get install -y python3-dev build-essential\n"
                "  Full traceback:\n" + tb
            )
            return 1
        print("FAIL: unexpected error while importing diffusers:\n" + tb)
        return 1

    ver = getattr(diffusers, "__version__", "0")
    print("diffusers:", ver)
    print("WanImageToVideoPipeline: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
