"""
Load Wan 2.2 TI2V 5B with Diffusers using WanImageToVideoPipeline + expand_timesteps.

Hub repos expose ``model_index.json`` as ``WanPipeline`` even though TI2V I2V matches
``WanImageToVideoPipeline``. Loading each submodule with ``from_pretrained`` respects
``*.safetensors.index.json`` sharding automatically (no manual stitch).

VRAM-oriented defaults for RTX 3090 (24 GB): bf16 transformer, fp32 VAE tiling,
8-bit / 4-bit UMT5 via bitsandbytes when requested, sequential CPU offload when no bitsandbytes
weights are present (hooks break on quantized params).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel

from umt5_quant import umt5_bnb_config


DEFAULT_MODEL_ID = os.environ.get("WAN22_MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")


def load_wan22_ti2v_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    torch_dtype: torch.dtype | None = None,
    transformer_dtype: torch.dtype | None = None,
    local_files_only: bool = False,
    text_encoder_quant: str = "int8",
    sequential_offload: bool = True,
    vae_tiling: bool = True,
    vae_tile_min: int | None = None,
    low_cpu_mem_usage: bool = True,
) -> WanImageToVideoPipeline:
    torch_dtype = torch_dtype or torch.bfloat16
    transformer_dtype = transformer_dtype or torch_dtype
    common = {"local_files_only": local_files_only, "low_cpu_mem_usage": low_cpu_mem_usage}

    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32, **common
    )

    transformer_kwargs: dict[str, Any] = dict(common)
    transformer_kwargs["torch_dtype"] = transformer_dtype
    transformer_kwargs["device_map"] = None

    transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", **transformer_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", **common)

    quant_cfg = None
    if torch.cuda.is_available():
        try:
            quant_cfg = umt5_bnb_config(text_encoder_quant, compute_dtype=transformer_dtype)
        except Exception:
            quant_cfg = None

    te_kw: dict[str, Any] = dict(common)
    if quant_cfg is not None:
        te_kw["quantization_config"] = quant_cfg
        te_kw["torch_dtype"] = torch.float32
        te_kw["device_map"] = "auto"
    else:
        te_kw["torch_dtype"] = transformer_dtype

    text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", **te_kw)

    scheduler = UniPCMultistepScheduler.from_pretrained(model_id, subfolder="scheduler")

    pipe = WanImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        image_encoder=None,
        image_processor=None,
        transformer_2=None,
        boundary_ratio=None,
        expand_timesteps=True,
    )

    if hasattr(pipe.vae, "enable_tiling") and vae_tiling:
        if vae_tile_min is not None and vae_tile_min > 0:
            # Smaller tiles → lower peak VRAM during decode (more passes, slower).
            stride = max(int(vae_tile_min * 0.75), 32)
            pipe.vae.enable_tiling(
                tile_sample_min_height=vae_tile_min,
                tile_sample_min_width=vae_tile_min,
                tile_sample_stride_height=stride,
                tile_sample_stride_width=stride,
            )
        else:
            pipe.vae.enable_tiling()

    # accelerate sequential CPU offload is incompatible with bitsandbytes params (int8 / int4).
    use_sequential = (
        sequential_offload
        and torch.cuda.is_available()
        and quant_cfg is None
    )
    if use_sequential:
        pipe.enable_sequential_cpu_offload()
    elif torch.cuda.is_available():
        pipe.to("cuda")

    return pipe
