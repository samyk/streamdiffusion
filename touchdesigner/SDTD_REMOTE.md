# StreamDiffusionTD Remote Backend

Use the **native StreamDiffusionTD Settings UI** with inference on your Linux box.

## Architecture

```
StreamDiffusionTD Settings (prompt, t_index, guidance, ...)
        │  immediate PATCH
        ▼
http://192.168.0.90:8780/v1/streams/remote-1   (Daydream-compatible API)
        │
        ▼
sdtd-bridge on hal (same process as WebSocket :8765)
        │
        ▼
NDI streamdiffusion_out → TouchDesigner
```

Video stays on **NDI** (`td_streamdiffusion_in` / `streamdiffusion_out`).
Control uses **Daydream-style REST** (not WebRTC).

## Setup

### 1. Linux (hal)

```bash
cd ~/c/samysd
source .venv/bin/activate
source scripts/env_cuda.sh
sdtd-bridge --acceleration none --preset sd_turbo_fast \
  --input-name td_streamdiffusion_in \
  --output-name streamdiffusion_out \
  --daydream-port 8780
```

Or: `./scripts/run_bridge_screen.sh`

### 2. TouchDesigner

1. Load `StreamDiffusionTD.tox` at `/project1/StreamDiffusionTD`
2. Run:

```python
exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_sdtd_remote_backend.py", encoding="utf-8").read())
```

3. Wire video (if not already):

```
webcam_flip → ndiout1 (td_streamdiffusion_in)
ndiin1 (streamdiffusion_out) → out1
```

4. On **Install** page: do **not** use Daydream cloud or local PC backend while Remote is active.
5. Change any Setting — it PATCHes hal immediately.

## Parameter mapping

| StreamDiffusionTD | Daydream / bridge |
|---|---|
| Promptdict | `prompts` → `set_prompts` |
| Tindexblock | `t_index_list` → `set_denoise` |
| Guidancescale | `guidance_scale` |
| Delta | `delta` |
| Seeddict | `seed` |
| Modelid | `model_id` → `load_model` preset |
| Sdmode | `mode` |
| Acceleration | `acceleration` |

Denoise is **integer 1–49** (T-index), not 0–1.

## Manual API test

```bash
curl -X PATCH http://192.168.0.90:8780/v1/streams/remote-1 \
  -H 'Content-Type: application/json' \
  -d '{"params":{"prompt":"liquid chrome orchid","t_index_list":[40],"guidance_scale":1.2,"delta":1.0,"seed":42,"enable_similar_image_filter":false}}'
```
