# StreamDiffusion TouchDesigner Bridge

Python bridge that receives video from TouchDesigner over NDI, processes it with
StreamDiffusion on a Linux NVIDIA Blackwell machine, sends the processed frames back
over NDI, and accepts structured controls from TouchDesigner over WebSocket.

## Linux Setup

On the RTX 6000 Blackwell Linux machine:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev avahi-daemon
./scripts/setup_blackwell_linux.sh
```

Blackwell needs a CUDA 12.8+ PyTorch build with `sm_120` support. The setup script
installs matched PyTorch nightly `cu128` wheels (`torch`, `torchvision`, `torchaudio`
together), StreamDiffusion, and this bridge.

If pip reports a `torch` / `torchvision` conflict, your venv has mismatched wheels.
Repair with:

```bash
./scripts/install_pytorch_cu128.sh
```

Verify:

```bash
sdtd-verify-gpu
sdtd-verify-inference
```

You want to see the RTX 6000, device capability near `(12, 0)`, and `sm_120` in
`arch_list`.

If `run_streamdiffusion.sh` fails with `No module named 'diffusers'`, the inference
stack was not installed into the venv. Fix it with:

```bash
./scripts/fix_inference_deps.sh
```

On Python 3.13, do not use `streamdiffusion[tensorrt]` from pip. That extra pins
`onnxruntime==1.16.3`, which is unavailable for 3.13. The setup scripts install
the base package plus compatible `onnxruntime>=1.20.0` instead.

On Blackwell (`sm_120`), both TensorRT and xformers are currently unsupported. Use:

```bash
source scripts/env_cuda.sh
sdtd-bridge --acceleration none --preset sd_turbo_fast
```

The bridge will also auto-fallback from `xformers`/`tensorrt` to `none` on capability 12.x GPUs.

## First Network Test

Start with passthrough. This validates NDI both directions without loading any model:

```bash
./scripts/run_passthrough.sh
```

In TouchDesigner, send an `NDI Out TOP` named `td_streamdiffusion_in` and receive the
bridge output with an `NDI In TOP` source named `streamdiffusion_out`.

If NDI discovery fails on Linux, make sure `avahi-daemon` is running and that multicast
is allowed on the LAN/VLAN.

## Run StreamDiffusion

```bash
./scripts/run_streamdiffusion.sh "cybernetic botanical glass sculpture"
```

Or directly:

```bash
sdtd-bridge \
  --preset sd_turbo_fast \
  --input-name td_streamdiffusion_in \
  --output-name streamdiffusion_out \
  --width 512 \
  --height 512 \
  --prompt "cybernetic botanical glass sculpture"
```

The first TensorRT run can take a while because engines are built under `engines/`.
Subsequent runs with the same model/resolution/batch config reuse cached engines.

## Control API

TouchDesigner connects to:

```text
ws://<linux-box-ip>:8765/control
```

Example messages:

```json
{ "type": "set_prompt", "prompt": "liquid chrome orchid" }
{ "type": "set_negative_prompt", "negative_prompt": "blurry, low detail" }
{ "type": "set_guidance_scale", "value": 1.1 }
{ "type": "set_delta", "value": 1.0 }
{ "type": "set_strength", "value": 0.55 }
{ "type": "set_filter", "threshold": 0.98, "max_skip_frame": 10 }
{ "type": "load_model", "preset": "sd_turbo_fast" }
{ "type": "load_model", "preset": "sd_turbo_quality" }
{ "type": "load_model", "preset": "lcm_lora_style" }
```

Status snapshots are sent once per second:

```json
{
  "type": "status",
  "status": "running",
  "preset": "sd_turbo_fast",
  "fps_in": 30.0,
  "fps_out": 28.4,
  "latency_ms": 48.2,
  "loading": false,
  "last_error": null
}
```

## TouchDesigner

See `touchdesigner/README.md`.

Print the callback DAT code with:

```bash
sdtd-touchdesigner-helper
```

## Useful CLI Flags

```bash
sdtd-bridge --help
```

Important flags:

- `--passthrough-test`: no model load, just NDI in -> NDI out.
- `--video-backend mock`: generate synthetic input and discard output. Useful for
  testing WebSocket/status on a machine without NDI.
- `--width` / `--height`: keep these fixed during a performance. TensorRT engines are
  resolution-specific.
- `--engine-dir`: TensorRT cache location.

## Latency Probe

Local smoke test:

```bash
sdtd-latency-probe --video-backend mock --preset passthrough --seconds 5
```

On the Linux/TouchDesigner network, run the same probe with NDI passthrough:

```bash
sdtd-latency-probe --video-backend ndi --preset passthrough --seconds 10
```

Then run the full bridge with StreamDiffusion and compare `fps_out` and `latency_ms`.

