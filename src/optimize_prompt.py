#!/usr/bin/env python3
"""
Optimize a *residual* motion embedding so ``base(coarse_prompt) + delta`` matches a teacher
embedding built from ``coarse_prompt + motion_description(reference_video)``.

This avoids backprop through the 5B transformer (cheap, stable) while still tying the prompt
embedding to observed motion in ``reference_video``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

from encoding_utils import encode_umt5_prompts
from motion_stats import motion_description_from_video
from umt5_quant import umt5_bnb_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize motion prompt embedding residual (UMT5 space).")
    p.add_argument("--model-id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    p.add_argument("--reference-video", type=Path, required=True)
    p.add_argument("--motion-prompt", type=str, required=True, help="Coarse motion description (starting point).")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--output", type=Path, default=Path("outputs/prompt_optim.pt"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--text-encoder-quant",
        type=str,
        choices=("none", "int8", "int4"),
        default=os.environ.get("WAN22_TEXT_ENCODER_QUANT", "int8"),
        help="UMT5 bitsandbytes: int4 (lowest VRAM, good for tests), int8 (default), none (full precision).",
    )
    p.add_argument("--no-text-encoder-8bit", action="store_true", help="Alias for --text-encoder-quant none.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    suffix = motion_description_from_video(args.reference_video)
    teacher_prompt = f"{args.motion_prompt.strip()} {suffix}"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", local_files_only=args.local_files_only)

    te_mode = "none" if args.no_text_encoder_8bit else args.text_encoder_quant

    te_kw: dict = dict(local_files_only=args.local_files_only, low_cpu_mem_usage=True)
    quant_cfg = None
    if device.type == "cuda" and te_mode != "none":
        try:
            quant_cfg = umt5_bnb_config(te_mode, compute_dtype=dtype)
        except Exception:
            quant_cfg = None
    if quant_cfg is not None:
        te_kw["quantization_config"] = quant_cfg
        te_kw["device_map"] = "auto"
        te_kw["torch_dtype"] = torch.float32
        encoder_dtype = torch.float32
    else:
        te_kw["torch_dtype"] = dtype
        encoder_dtype = dtype

    try:
        text_encoder = UMT5EncoderModel.from_pretrained(args.model_id, subfolder="text_encoder", **te_kw)
    except FileNotFoundError as e:
        err = str(e)
        if "safetensors" in err or ".bin" in err:
            mid = args.model_id
            local_hint = (
                f'  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \\\n'
                f'    --local-dir "{mid}" --resume-download\n'
                if not str(mid).startswith("Wan-AI/")
                else "  huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \\\n    --local-dir /path/to/Wan2.2-TI2V-5B-Diffusers --resume-download\n"
            )
            raise FileNotFoundError(
                "Local Wan checkpoint is incomplete (a weight shard is missing).\n"
                "Under text_encoder/ you need model-00001-of-00003.safetensors (and siblings), not only the index JSON.\n\n"
                "Resume into the same folder (Hugging Face CLI):\n"
                f"{local_hint}\n"
                "Then point --model-id / WAN22_MODEL_ID at that directory.\n\n"
                f"Original error: {err}"
            ) from e
        raise
    if "device_map" not in te_kw:
        text_encoder = text_encoder.to(device)

    with torch.no_grad():
        teacher = encode_umt5_prompts(
            tokenizer,
            text_encoder,
            [teacher_prompt],
            device=None,
            dtype=encoder_dtype,
            max_sequence_length=args.max_seq_len,
        )
        base = encode_umt5_prompts(
            tokenizer,
            text_encoder,
            [args.motion_prompt],
            device=None,
            dtype=encoder_dtype,
            max_sequence_length=args.max_seq_len,
        )

    # encode_umt5_prompts is @torch.inference_mode(); outputs cannot participate in autograd.
    teacher = teacher.detach().clone()
    base = base.detach().clone()

    delta = torch.zeros_like(base, dtype=encoder_dtype, requires_grad=True)
    opt = torch.optim.AdamW([delta], lr=args.lr, weight_decay=1e-4)

    text_encoder.eval()
    for step in range(args.steps):
        pred = base + delta
        loss = F.mse_loss(pred, teacher)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step:5d}  mse {loss.item():.6f}")

    payload = {
        "prompt_embeds": (base + delta).detach().cpu(),
        "delta": delta.detach().cpu(),
        "teacher_prompt": teacher_prompt,
        "coarse_prompt": args.motion_prompt,
        "max_sequence_length": args.max_seq_len,
        "model_id": args.model_id,
        "motion_stats_suffix": suffix,
    }
    torch.save(payload, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
