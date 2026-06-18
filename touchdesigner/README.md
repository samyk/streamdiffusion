# TouchDesigner Control

## Control UI

```python
exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_streamdiffusion_ui.py", encoding="utf-8").read())
```

Creates `/project1/streamdiffusion_ui` — a floating **parameterCOMP** panel wired to `streamdiffusion_bridge`.
Drag it into a pane as **Panel**, or right-click → **View** to control prompt, denoise, guidance, seed, preset, etc.

## Quick start (recommended)

Run in TD textport or a Text DAT:

```python
exec(open("/Users/samy/c/touch/samysd/touchdesigner/build_streamdiffusion_bridge_comp.py").read())
```

This creates `/project1/streamdiffusion_bridge` with:

| Parameter | Bridge command |
| --- | --- |
| Prompt / Prompt2 + weight | `set_prompt` / `set_prompts` |
| Negative Prompt | `set_negative_prompt` |
| Denoise, Step2–4 | `set_denoise` (t_index 1–49, StreamDiffusionTD-style) |
| Guidance / Delta | `set_guidance_scale` / `set_delta` |
| Seed | `set_seed` |
| Preset / SD Mode / Acceleration | `load_model` |
| Filter threshold / skip | `set_filter` |
| Bridge Host / Port | WebSocket target (`hal:8765`) |
| Push All | sends full state once |

**Video path (project level):** `webcam_in` → `webcam_flip` → `ndiout1` → hal → `ndiin1` → `out1`

The bridge COMP is **control-only** (WebSocket). NDI lives at `/project1` level to avoid duplicate senders.

1. Wire your source TOP into `webcam_in` (or replace webcam)
2. Set `Bridgehost` to your Linux box IP/hostname
3. Start bridge on hal: `sdtd-bridge --acceleration tensorrt --preset sd_turbo_fast`
4. Pulse **Push All** on first connect

## Manual patch

If you prefer wiring yourself:

1. Source TOP → resize to `512x512`
2. NDI Out: `td_streamdiffusion_in`
3. NDI In: `streamdiffusion_out`
4. WebSocket DAT → `netaddress` = hal IP, `port` = `8765`
5. Text DAT `callbacks` ← paste `touchdesigner/sdtd_ws_callbacks.py`
6. Parameter Execute DAT watching your control COMP

## Example calls

```python
cb = op("/project1/streamdiffusion_bridge/callbacks").module
cb.send_prompt("liquid chrome orchid")
cb.send_denoise([35, 20, 10])
cb.send_guidance(1.1)
cb.send_seed(42)
cb.load_preset("sd_turbo_fast", mode="img2img", acceleration="tensorrt")
cb.push_all()
```

Throttle slider/param sends to ~20–30 Hz max.
