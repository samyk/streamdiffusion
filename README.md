# StreamDiffusion TouchDesigner Bridge

Python bridge that receives video from TouchDesigner over NDI, runs real-time
diffusion on a Linux NVIDIA GPU box, sends processed frames back over NDI, and
accepts live controls from TouchDesigner over WebSocket or the Daydream-style
REST API.

Supported inference paths:

- **StreamDiffusion** — SD Turbo / SDXL Turbo / LCM-LoRA (default stack)
- **FLUX.2 Klein** — optional `Flux2KleinPipeline` path (4B / 9B)
- **Passthrough** — NDI loopback for network validation

Post-processing:

- **NVIDIA Maxine VSR** — default GPU upscale (`maxine-vsr`)
- **Real-ESRGAN** — sharper, slower (`realesrgan`)
- **Bicubic** — fastest (`bicubic`)

## Linux Setup (Blackwell)

On the RTX 6000 Blackwell Linux machine:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev avahi-daemon curl
./scripts/setup_blackwell_linux.sh
```

Blackwell needs CUDA 13.2 PyTorch nightly (`cu132`) with `sm_120`:

```bash
sdtd-verify-gpu
sdtd-verify-inference
```

Repair mismatched wheels:

```bash
./scripts/install_pytorch_cu132.sh
./scripts/fix_inference_deps.sh
```

StreamDiffusion on Blackwell uses **TensorRT** by default (`acceleration=tensorrt` in
`streamdiffusion_td_bridge/defaults.py` and TD `hal_control`). FLUX.2 Klein still uses
`acceleration=none` (separate pipeline).

### Optional stacks

```bash
# GPU upscale (Maxine VSR + Real-ESRGAN fallback)
./scripts/install_upscaler_deps.sh

# FLUX.2 Klein (upgrades diffusers; accept HF model license first)
./scripts/install_flux2_klein_deps.sh
```

To return to the SD-only diffusers pin after trying FLUX:

```bash
./scripts/fix_inference_deps.sh
```

## Quick Start

### Passthrough (no model)

```bash
./scripts/run_passthrough.sh
```

TD: send `td_streamdiffusion_in`, receive `streamdiffusion_out`.

### StreamDiffusion (default)

```bash
./scripts/run_bridge_screen.sh "your prompt"
# or
sdtd-bridge --preset sdxl_turbo_fast --width 768 --height 768 \
  --acceleration tensorrt --prompt "your prompt"
```

### FLUX.2 Klein

```bash
./scripts/install_flux2_klein_deps.sh

sdtd-bridge \
  --preset flux2_klein_fast \
  --width 768 --height 768 \
  --acceleration none \
  --prompt "your prompt"
```

### With GPU upscale

```bash
SDTD_UPSCALE=1 ./scripts/run_bridge_screen.sh
# default: maxine-vsr, quality medium, x2 upscale
```

## Resolutions

| Setting | What it controls |
|---|---|
| `--width` / `--height` | Inference resolution. NDI **in** is resized to this. |
| `SDTD_UPSCALE=1` | Enable post-inference upscale |
| `SDTD_UPSCALE_FACTOR=2` | NDI **out** = infer × factor (2 or 4) |

Examples:

| Infer | Upscale | NDI out |
|---|---|---|
| 768×768 | off | 768×768 |
| 768×768 | x2 | 1536×1536 |
| 512×512 | x2 | 1024×1024 |

Env vars (instance A):

```bash
SDTD_WIDTH=512 SDTD_HEIGHT=512 SDTD_UPSCALE=1 ./scripts/run_bridge_screen.sh
```

Instance B overrides: `SDTD_WIDTH_B`, `SDTD_HEIGHT_B`, etc.

Runtime check:

```bash
curl -s http://hal:8780/v1/streams/remote-1 | jq '.runtime | {width, height, extra}'
```

## Presets

| Preset | Pipeline | Model | Notes |
|---|---|---|---|
| `sdxl_turbo_fast` | StreamDiffusion | SDXL Turbo | Default on hal |
| `sdxl_turbo_quality` | StreamDiffusion | SDXL Turbo | 2-step |
| `sd_turbo_fast` | StreamDiffusion | SD Turbo | |
| `sd_turbo_quality` | StreamDiffusion | SD Turbo | 2-step |
| `lcm_lora_style` | StreamDiffusion | SD1.5 + LCM | |
| `flux2_klein_fast` | FLUX.2 Klein | 4B | 4 steps, img2img |
| `flux2_klein_quality` | FLUX.2 Klein | 4B | 6 steps, frame_batch=2 |
| `flux2_klein_9b` | FLUX.2 Klein | 9B | Higher quality |
| `passthrough` | none | — | NDI loopback |

## CLI Options

```bash
sdtd-bridge --help
```

### Core

| Flag | Default | Description |
|---|---|---|
| `--preset` | `sd_turbo_fast` | Model preset (see table) |
| `--width` / `--height` | 512 | Inference resolution |
| `--prompt` | `""` | Initial prompt |
| `--negative-prompt` | `""` | Negative prompt |
| `--guidance-scale` | 1.1 | CFG scale |
| `--delta` | 1.0 | RCFG delta |
| `--seed` | 2 | Seed |
| `--acceleration` | `tensorrt` | `none`, `xformers`, `tensorrt` |
| `--engine-dir` | `engines` | TensorRT cache dir |
| `--passthrough-test` | off | Skip model load |

### Stream batch

| Flag | Default | Description |
|---|---|---|
| `--frame-buffer-size` | preset | StreamDiffusion `frame_buffer_size` / stream batch depth |

Higher values increase temporal smoothing and batch throughput on SD presets.
Reloads the model when changed live.

### FLUX.2 Klein

| Flag | Default | Description |
|---|---|---|
| `--flux-transformer-engine` | on | Blackwell: `torch.compile` transformer + bfloat16 |
| `--no-flux-transformer-engine` | — | Use float16 eager mode instead |

### Upscale

| Flag | Default | Description |
|---|---|---|
| `--upscale` | off | Enable post-inference upscale |
| `--upscale-factor` | 2 | 1, 2, or 4 |
| `--upscale-method` | `maxine-vsr` | `maxine-vsr`, `realesrgan`, `bicubic` |
| `--upscale-maxine-quality` | `medium` | `low`…`ultra`, `highbitrate_*` |
| `--upscale-half` / `--no-upscale-half` | on | Real-ESRGAN fp16 |

### NDI / API

| Flag | Default | Description |
|---|---|---|
| `--input-name` | `td_streamdiffusion_in` | NDI source |
| `--output-name` | `streamdiffusion_out` | NDI sender |
| `--stream-id` | `remote-1` | REST stream id |
| `--daydream-port` | 8780 | REST API port |
| `--control-port` | 8765 | WebSocket control |
| `--video-backend` | `ndi` | `ndi` or `mock` |

## Launch Scripts (hal)

```bash
# Instance A (default)
./scripts/run_bridge_screen.sh
./scripts/run_bridge_screen.sh "prompt"

# Instance A + B
./scripts/run_bridge_screen.sh --dual "prompt A" "prompt B"

# Env overrides
SDTD_PRESET=flux2_klein_fast \
SDTD_WIDTH=768 SDTD_HEIGHT=768 \
SDTD_UPSCALE=1 \
SDTD_UPSCALE_METHOD=maxine-vsr \
SDTD_UPSCALE_MAXINE_QUALITY=high \
SDTD_UPSCALE_FACTOR=2 \
SDTD_FRAME_BUFFER_SIZE=2 \
./scripts/run_bridge_screen.sh
```

| Env var | Description |
|---|---|
| `SDTD_PRESET` | Preset name |
| `SDTD_WIDTH` / `SDTD_HEIGHT` | Inference res |
| `SDTD_UPSCALE` | `1` = enable upscale |
| `SDTD_UPSCALE_FACTOR` | 2 or 4 |
| `SDTD_UPSCALE_METHOD` | `maxine-vsr`, `realesrgan`, `bicubic` |
| `SDTD_UPSCALE_MAXINE_QUALITY` | Maxine quality preset |
| `SDTD_UPSCALE_HALF` | `0` = disable Real-ESRGAN fp16 |
| `SDTD_FRAME_BUFFER_SIZE` | Stream batch depth (reloads model) |
| `SDTD_FLUX_TRANSFORMER_ENGINE` | `0` = float16 eager FLUX path |
| `SDTD_DUAL` | `1` = launch A+B |
| `SDTD_PROMPT_B` | Instance B prompt |

Instance B also supports `SDTD_PRESET_B`, `SDTD_WIDTH_B`, `SDTD_HEIGHT_B`.

## REST API (Daydream-compatible)

Default: `http://hal:8780/v1/streams/remote-1`

```bash
# Status
curl http://hal:8780/v1/streams/remote-1

# Hot-update params
curl -X PATCH http://hal:8780/v1/streams/remote-1 \
  -H 'Content-Type: application/json' \
  -d '{"params": {
    "preset": "flux2_klein_fast",
    "prompt": "liquid chrome orchid",
    "width": 768,
    "height": 768,
    "frame_buffer_size": 2,
    "flux_transformer_engine": true,
    "upscale_enabled": true,
    "upscale_method": "maxine-vsr",
    "upscale_factor": 2,
    "upscale_maxine_quality": "medium",
    "acceleration": "none"
  }}'
```

Key REST params:

| Param | Description |
|---|---|
| `preset` | Preset name |
| `prompt` / `prompts` | Text prompt(s) |
| `negative_prompt` | Negative prompt |
| `t_index_list` | Denoise steps |
| `width` / `height` | Resolution |
| `guidance_scale`, `delta`, `seed` | Quality controls |
| `acceleration` | `none` / `xformers` / `tensorrt` |
| `frame_buffer_size` | Stream batch depth |
| `flux_transformer_engine` | FLUX Blackwell compile on/off |
| `upscale_enabled` | Toggle upscale |
| `upscale_method` | Upscale backend |
| `upscale_factor` | 2 or 4 |
| `upscale_maxine_quality` | Maxine preset |
| `paused` | `true` = passthrough |
| `loras` | LoRA list (SD presets) |

## WebSocket Control

```text
ws://<linux-box>:8765/control
```

```json
{ "type": "set_prompt", "prompt": "liquid chrome orchid" }
{ "type": "set_frame_buffer", "frame_buffer_size": 2 }
{ "type": "set_flux_transformer_engine", "enabled": false }
{ "type": "load_model", "preset": "flux2_klein_fast", "width": 768, "height": 768 }
{ "type": "set_upscale", "enabled": true, "method": "maxine-vsr", "factor": 2 }
```

Status snapshots (1 Hz) include `extra.pipeline`, `extra.frame_buffer_size`,
`extra.flux_transformer_engine`, `extra.upscale_runtime`, `extra.output_width`.

## TouchDesigner

See `touchdesigner/README.md` and `touchdesigner/DUAL_INSTANCES.md`.

Build HAL control UI:

```python
exec(open(".../touchdesigner/build_hal_control.py", encoding="utf-8").read())
```

HAL control exposes:

- Preset (including FLUX.2 Klein)
- Width / Height
- Frame Batch Count
- FLUX Blackwell Transformer Engine toggle
- Denoise steps, guidance, seed, acceleration
- Prompt / negative prompt

Helper:

```bash
sdtd-touchdesigner-helper
```

## Dual Instances

| | Instance A | Instance B |
|---|---|---|
| NDI in | `td_streamdiffusion_in` | `td_streamdiffusion_in_b` |
| NDI out | `streamdiffusion_out` | `streamdiffusion_out_b` |
| REST | `:8780/remote-1` | `:8781/remote-2` |
| screen | `sdtd-bridge` | `sdtd-bridge-b` |

```bash
./scripts/run_bridge_screen.sh --dual
```

## Latency Probe

```bash
sdtd-latency-probe --video-backend mock --preset passthrough --seconds 5
sdtd-latency-probe --video-backend ndi --preset passthrough --seconds 10
```

## Troubleshooting

| Issue | Fix |
|---|---|
| `No module named 'diffusers'` | `./scripts/fix_inference_deps.sh` |
| `sm_120` missing | `./scripts/install_pytorch_cu132.sh` |
| FLUX import error | `./scripts/install_flux2_klein_deps.sh` |
| SD broken after FLUX install | `./scripts/fix_inference_deps.sh` |
| NDI not found | `avahi-daemon`, check VLAN multicast |
| Slow upscale | `SDTD_UPSCALE_METHOD=maxine-vsr` or infer at 512 |
| TensorRT engine build fails | Check `engines/` cache; try `--acceleration xformers` temporarily |

## Project Layout

```
streamdiffusion_td_bridge/   # bridge service
  vendor/wrapper.py          # StreamDiffusion wrapper
  vendor/flux_klein_wrapper.py  # optional FLUX.2 Klein path
  maxine_upscaler.py         # NVIDIA Maxine VSR
  upscaler.py                # upscale factory
scripts/                     # setup + launch scripts
touchdesigner/               # TD builders + HAL remote sync
engines/                     # TensorRT + Real-ESRGAN weights
```
