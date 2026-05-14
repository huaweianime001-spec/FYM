"""UMT5 prompt encoding aligned with diffusers Wan pipelines (trim + pad to max_length)."""

from __future__ import annotations

import html

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

try:
    import ftfy
except ImportError:
    ftfy = None


def basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prompt_clean(text: str) -> str:
    return whitespace_clean(basic_clean(text))


@torch.inference_mode()
def encode_umt5_prompts(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompts: list[str],
    *,
    device: torch.device | None,
    dtype: torch.dtype,
    max_sequence_length: int = 512,
) -> torch.Tensor:
    if device is None:
        device = next(text_encoder.parameters()).device

    prompts = [prompt_clean(p) for p in prompts]
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    pieces = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
    stacked = torch.stack(
        [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in pieces], dim=0
    )
    return stacked
