# Follow-Your-Motion style TI2V on Wan 2.2 (5B Diffusers)

Motion transfer from a **reference video** onto a **subject image** using **`Wan-AI/Wan2.2-TI2V-5B-Diffusers`** and Diffusers **`WanImageToVideoPipeline`** (`expand_timesteps=True`). Inspired by [Follow-Your-Motion](https://arxiv.org/abs/2506.05207); this repo does **not** train DiT LoRAs—it fits a **small UMT5 embedding residual** from optical-flow statistics vs a coarse text prompt, then runs full TI2V decode.

**Upstream vs here:** the paper trains spatial/temporal LoRAs with custom tooling. Here we keep the official **TI2V 5B** backbone and add **optimizable motion prompt embeddings** only.

## Model weights

This repository **does not** ship or download checkpoints. You need **`Wan-AI/Wan2.2-TI2V-5B-Diffusers`** (sharded `safetensors` + `model_index.json`) either:

- In the Hugging Face cache (default `--model-id`), or  
- On disk: `export WAN22_MODEL_ID=/path/to/Wan2.2-TI2V-5B-Diffusers`

Example (any machine with the CLI):

```bash
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --local-dir "$HOME/models/Wan2.2-TI2V-5B-Diffusers"
export WAN22_MODEL_ID="$HOME/models/Wan2.2-TI2V-5B-Diffusers"
```

`run_pipeline.sh` auto-picks **`$HOME/models/Wan2.2-TI2V-5B-Diffusers`** when `WAN22_MODEL_ID` is unset and that folder contains `model_index.json`; otherwise it uses the Hub id.

## Setup

**Python 3.10+**, NVIDIA driver + CUDA PyTorch matching your GPU.

```bash
cd follow_motion_wan22
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
# Pick the cu12x wheel index that matches your install: https://pytorch.org/get-started/locally/
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install --no-cache-dir -r requirements.txt
bash scripts/smoke_import.sh
```

On **Debian/Ubuntu** (including WSL), if `bitsandbytes` / `triton` fail with `Python.h: No such file`:

```bash
bash scripts/install_wsl_system_deps.sh   # installs python3-dev, build-essential
```

Optional **chunked** install (same steps, scripted):

```bash
bash scripts/install_wsl_chunked.sh
```

Syntax-only check (no GPU needed):

```bash
python3 -m py_compile src/*.py && echo OK
```

## Run motion transfer

**Required:** `SUBJECT_IMAGE` (PNG/JPEG) and `MOTION_VIDEO` (MP4). Optional: `WAN22_MODEL_ID`, `MOTION_PROMPT`, `OUTPUT_DIR`.

```bash
cd follow_motion_wan22
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

export SUBJECT_IMAGE=/path/to/subject.png
export MOTION_VIDEO=/path/to/reference_motion.mp4
export MOTION_PROMPT="A short coarse description of the motion you want."
# export WAN22_MODEL_ID=/path/to/Wan2.2-TI2V-5B-Diffusers   # if not using Hub cache

bash scripts/run_pipeline.sh
```

Artifacts: **`outputs/prompt_optim.pt`** and **`outputs/corgi_motion_transfer.mp4`**.

### GPU memory

- **`--text-encoder-quant int4`** only shrinks **UMT5**; the **5B DiT** and **activations** (resolution × frames × steps) dominate VRAM.
- **VAE decode** is a second spike; **`WAN22_VAE_TILE_MIN`** (e.g. `128`) or **`WAN22_CPU_VAE_DECODE=1`** help on tight cards.
- **`WAN22_LOW_VRAM=1`** applies smaller defaults (frames, resolution, steps, int4 TE, fp16 DiT, smaller VAE tiles).

`num_frames` must be **`4*k+1`** (e.g. 49, 65, 81).

### Tunables (environment)

| Variable | Meaning |
|----------|---------|
| `SUBJECT_IMAGE` / `MOTION_VIDEO` | **Required** for `run_pipeline.sh` |
| `MOTION_PROMPT` | Coarse motion sentence (default in script is a corgi example) |
| `HEIGHT` / `WIDTH` | e.g. 704×1280 or 1280×704 |
| `NUM_FRAMES` | `4k+1` |
| `STEPS` | Denoise steps |
| `WAN22_MODEL_ID` | Hub id or local Diffusers folder |
| `WAN22_TEXT_ENCODER_QUANT` | `int4`, `int8` (default), `none` |
| `WAN22_TRANSFORMER_DTYPE` | `fp16` or `bf16` (inference only) |
| `WAN22_LOW_VRAM` | `1` → smaller defaults |
| `WAN22_VAE_TILE_MIN` | Smaller spatial VAE tiles (e.g. `128`) |
| `WAN22_CPU_VAE_DECODE` | `1` → decode on CPU after denoising |

## Manual commands

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

python src/optimize_prompt.py \
  --model-id "${WAN22_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}" \
  --reference-video /path/to/motion.mp4 \
  --motion-prompt "Your coarse motion description." \
  --output outputs/prompt_optim.pt

python src/inference.py \
  --model-id "${WAN22_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}" \
  --prompt-embeddings outputs/prompt_optim.pt \
  --subject-image /path/to/subject.png \
  --output-video outputs/out.mp4
```

## Layout

| Path | Role |
|------|------|
| `src/load_pipeline.py` | TI2V pipeline load; offload; UMT5 quant; optional VAE tile size |
| `src/umt5_quant.py` | bitsandbytes presets for UMT5 |
| `src/motion_stats.py` | Optical-flow stats → teacher prompt suffix |
| `src/optimize_prompt.py` | Fits embedding residual |
| `src/inference.py` | TI2V decode; optional CPU VAE decode |
| `src/encoding_utils.py` | UMT5 encode helper |
| `scripts/run_pipeline.sh` | optimize → infer |
| `scripts/install_step_*.sh` / `install_wsl_chunked.sh` | Optional staged venv + torch + deps |
| `scripts/smoke_import.sh` | Import check |

## Limitations

Not a full replication of the paper’s LoRA training stack; this is a **Diffusers-first** recipe for **motion-aware embeddings + TI2V**.
