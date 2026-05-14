"""bitsandbytes quantization presets for the Wan UMT5 text encoder."""

from __future__ import annotations

import torch
from transformers import BitsAndBytesConfig


def umt5_bnb_config(mode: str, *, compute_dtype: torch.dtype = torch.bfloat16) -> BitsAndBytesConfig | None:
    """
    Return a ``BitsAndBytesConfig`` for ``UMT5EncoderModel``, or ``None`` for full precision.

    ``int4`` uses NF4 — lowest VRAM, good for smoke tests; ``int8`` is the default balance.
    """
    m = (mode or "none").lower().strip()
    aliases = {"off": "none", "no": "none", "fp": "none", "fp16": "none", "bf16": "none", "false": "none"}
    m = aliases.get(m, m)
    if m in ("none", ""):
        return None
    if m == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if m == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unknown text encoder quant {mode!r}; expected none, int8, int4.")
