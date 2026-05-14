#!/usr/bin/env bash
# End-to-end motion-conditioned TI2V.
# Prereq: Wan2.2 TI2V 5B Diffusers weights (Hub cache or WAN22_MODEL_ID) — see README.
# Prereq: SUBJECT_IMAGE + MOTION_VIDEO env vars (existing files).
#
# Defaults target RTX 3090 (24 GB): NUM_FRAMES=65, STEPS=35 (override for quality).
# num_frames must stay 4*k+1 (49, 65, 81, 97, 121, …).
# WAN22_LOW_VRAM=1 → smaller H/W/frames/steps + int4 TE + fp16 DiT + VAE tile min 128 (README “GPU memory”).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source ".venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Smaller spatial / temporal / steps + int4 TE + fp16 DiT when probing on a 24 GB card.
if [[ "${WAN22_LOW_VRAM:-0}" == "1" ]]; then
  export NUM_FRAMES="${NUM_FRAMES:-33}"
  export HEIGHT="${HEIGHT:-480}"
  export WIDTH="${WIDTH:-832}"
  export STEPS="${STEPS:-20}"
  export WAN22_TEXT_ENCODER_QUANT="${WAN22_TEXT_ENCODER_QUANT:-int4}"
  export WAN22_TRANSFORMER_DTYPE="${WAN22_TRANSFORMER_DTYPE:-fp16}"
  # Smaller spatial VAE tiles reduce peak VRAM during decode (Diffusers default tile is 256).
  export WAN22_VAE_TILE_MIN="${WAN22_VAE_TILE_MIN:-128}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs}"
SUBJECT_IMAGE="${SUBJECT_IMAGE:-}"
MOTION_VIDEO="${MOTION_VIDEO:-}"
if [[ -z "${WAN22_MODEL_ID:-}" ]] && [[ -f "$HOME/models/Wan2.2-TI2V-5B-Diffusers/model_index.json" ]]; then
  WAN22_MODEL_ID="$HOME/models/Wan2.2-TI2V-5B-Diffusers"
elif [[ -z "${WAN22_MODEL_ID:-}" ]]; then
  WAN22_MODEL_ID="Wan-AI/Wan2.2-TI2V-5B-Diffusers"
fi

mkdir -p "$OUTPUT_DIR"

if [[ -z "$SUBJECT_IMAGE" || -z "$MOTION_VIDEO" ]]; then
  echo "Set SUBJECT_IMAGE and MOTION_VIDEO to existing files (PNG/JPEG subject, MP4 reference motion)."
  echo "Example:  export SUBJECT_IMAGE=/path/to/subject.png MOTION_VIDEO=/path/to/motion.mp4"
  exit 1
fi
if [[ ! -f "$SUBJECT_IMAGE" ]]; then
  echo "SUBJECT_IMAGE is not a file: $SUBJECT_IMAGE"
  exit 1
fi
if [[ ! -f "$MOTION_VIDEO" ]]; then
  echo "MOTION_VIDEO is not a file: $MOTION_VIDEO"
  exit 1
fi

# Optional: WAN22_TEXT_ENCODER_QUANT=int4|int8|none — UMT5 bitsandbytes (int4 = lightest VRAM for quick tests).
TE_QUANT_ARGS=()
if [[ -n "${WAN22_TEXT_ENCODER_QUANT:-}" ]]; then
  TE_QUANT_ARGS+=(--text-encoder-quant "$WAN22_TEXT_ENCODER_QUANT")
fi

XFORM_ARGS=()
if [[ -n "${WAN22_TRANSFORMER_DTYPE:-}" ]]; then
  XFORM_ARGS+=(--transformer-dtype "$WAN22_TRANSFORMER_DTYPE")
fi

VAE_ARGS=()
if [[ "${WAN22_CPU_VAE_DECODE:-0}" == "1" ]]; then
  VAE_ARGS+=(--cpu-vae-decode)
fi
if [[ -n "${WAN22_VAE_TILE_MIN:-}" ]] && [[ "${WAN22_VAE_TILE_MIN}" =~ ^[0-9]+$ ]] && [[ "${WAN22_VAE_TILE_MIN}" != "0" ]]; then
  VAE_ARGS+=(--vae-tile-min "$WAN22_VAE_TILE_MIN")
fi

MOTION_PROMPT="${MOTION_PROMPT:-A cute corgi lifts its paw playfully, repeating the motion smoothly.}"

echo "=== Step 1/2: optimize motion prompt embedding (UMT5 residual vs teacher) ==="
python "$ROOT/src/optimize_prompt.py" \
  --model-id "$WAN22_MODEL_ID" \
  --reference-video "$MOTION_VIDEO" \
  --motion-prompt "$MOTION_PROMPT" \
  --output "$OUTPUT_DIR/prompt_optim.pt" \
  "${TE_QUANT_ARGS[@]}"

echo "=== Step 2/2: TI2V generate (subject image + saved embeddings) ==="
python "$ROOT/src/inference.py" \
  --model-id "$WAN22_MODEL_ID" \
  --prompt-embeddings "$OUTPUT_DIR/prompt_optim.pt" \
  --subject-image "$SUBJECT_IMAGE" \
  --output-video "$OUTPUT_DIR/corgi_motion_transfer.mp4" \
  --height "${HEIGHT:-704}" \
  --width "${WIDTH:-1280}" \
  --num-frames "${NUM_FRAMES:-65}" \
  --steps "${STEPS:-35}" \
  --guidance-scale "${GUIDANCE_SCALE:-5.0}" \
  --fps "${FPS:-24}" \
  "${TE_QUANT_ARGS[@]}" \
  "${XFORM_ARGS[@]}" \
  "${VAE_ARGS[@]}"

echo "Wrote: $OUTPUT_DIR/corgi_motion_transfer.mp4"
