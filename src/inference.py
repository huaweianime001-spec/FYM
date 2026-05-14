#!/usr/bin/env python3
"""TI2V inference: corgi (or any) subject image + optimized motion prompt embeddings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from diffusers.utils import export_to_video, load_image

from load_pipeline import DEFAULT_MODEL_ID, load_wan22_ti2v_pipeline

DEFAULT_NEGATIVE_ZH = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _env_flag(name: str) -> bool:
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _wan_denormalize_latents_for_decode(pipe, latents: torch.Tensor) -> torch.Tensor:
    """Match ``pipeline_wan_i2v`` before ``vae.decode`` (latent mean / std in VAE dtype)."""
    latents = latents.to(dtype=pipe.vae.dtype)
    z_dim = int(pipe.vae.config.z_dim)
    lm = torch.tensor(pipe.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
        1, z_dim, 1, 1, 1
    )
    ls_inv = (1.0 / torch.tensor(pipe.vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(
        1, z_dim, 1, 1, 1
    )
    return latents / ls_inv + lm


def decode_latents_on_cpu_to_numpy(pipe, latents: torch.Tensor):
    """Run VAE decode (+ postprocess) on CPU after freeing CUDA weights to avoid decode-time OOM."""
    latents = _wan_denormalize_latents_for_decode(pipe, latents)
    pipe.maybe_free_model_hooks()
    if getattr(pipe, "transformer", None) is not None:
        pipe.transformer.to("cpu")
    if getattr(pipe, "text_encoder", None) is not None:
        try:
            pipe.text_encoder.to("cpu")
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pipe.vae.to("cpu")
    latents = latents.to("cpu")
    with torch.inference_mode():
        video = pipe.vae.decode(latents, return_dict=False)[0]
    stacked = pipe.video_processor.postprocess_video(video, output_type="np")
    if isinstance(stacked, np.ndarray) and stacked.ndim == 5:
        return stacked[0]
    return stacked


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wan 2.2 TI2V 5B image-to-video with saved prompt embeddings.")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--prompt-embeddings", type=Path, required=True, help=".pt from optimize_prompt.py")
    p.add_argument("--subject-image", type=Path, required=True)
    p.add_argument("--output-video", type=Path, default=Path("outputs/ti2v_out.mp4"))
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=65, help="Must be 4*k+1 (e.g. 49, 65, 81). Lower saves VRAM.")
    p.add_argument("--steps", type=int, default=35, help="Denoising steps; lower=faster/less VRAM pressure.")
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_ZH)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument(
        "--text-encoder-quant",
        type=str,
        choices=("none", "int8", "int4"),
        default=os.environ.get("WAN22_TEXT_ENCODER_QUANT", "int8"),
        help="UMT5 bitsandbytes: int4 (lowest VRAM, good for smoke tests), int8 (default), none (full precision).",
    )
    p.add_argument("--no-text-encoder-8bit", action="store_true", help="Alias for --text-encoder-quant none.")
    p.add_argument(
        "--transformer-dtype",
        type=str.lower,
        choices=("bf16", "fp16"),
        default=(os.environ.get("WAN22_TRANSFORMER_DTYPE") or "bf16").lower(),
        help="DiT weight/activation dtype: fp16 uses less VRAM than bf16 (slight quality risk).",
    )
    p.add_argument(
        "--vae-tile-min",
        type=int,
        default=int(os.environ["WAN22_VAE_TILE_MIN"])
        if os.environ.get("WAN22_VAE_TILE_MIN", "").strip().isdigit()
        else 0,
        help="If >0, spatial VAE tile size (pixels); smaller = lower VRAM during decode, slower. Env: WAN22_VAE_TILE_MIN.",
    )
    p.add_argument(
        "--cpu-vae-decode",
        action="store_true",
        default=_env_flag("WAN22_CPU_VAE_DECODE"),
        help="After denoising, move DiT/TE off GPU and decode on CPU (slow; avoids CUDA OOM at VAE). Env: WAN22_CPU_VAE_DECODE=1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_video.parent.mkdir(parents=True, exist_ok=True)

    bundle = torch.load(args.prompt_embeddings, map_location="cpu", weights_only=False)
    max_seq = int(bundle.get("max_sequence_length", 512))

    te_q = "none" if args.no_text_encoder_8bit else args.text_encoder_quant
    transformer_dtype = torch.float16 if args.transformer_dtype == "fp16" else torch.bfloat16
    vae_tile_min = args.vae_tile_min if args.vae_tile_min > 0 else None
    pipe = load_wan22_ti2v_pipeline(
        args.model_id,
        local_files_only=args.local_files_only,
        text_encoder_quant=te_q,
        transformer_dtype=transformer_dtype,
        vae_tile_min=vae_tile_min,
    )

    dtype = pipe.transformer.dtype
    prompt_embeds = bundle["prompt_embeds"].to(device=pipe._execution_device, dtype=dtype)

    image = load_image(str(args.subject_image))

    pipe_kw = dict(
        image=image,
        prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=max_seq,
    )
    if args.cpu_vae_decode:
        latents = pipe(**pipe_kw, output_type="latent").frames
        if isinstance(latents, (list, tuple)):
            latents = latents[0]
        out = decode_latents_on_cpu_to_numpy(pipe, latents)
    else:
        out = pipe(**pipe_kw).frames[0]

    export_to_video(out, str(args.output_video), fps=args.fps)
    print(f"Wrote {args.output_video}")


if __name__ == "__main__":
    main()
