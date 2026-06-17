#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "Installing NVIDIA Maxine VSR (nvidia-vfx)..."
pip install nvidia-vfx

echo "Installing Real-ESRGAN fallback (spandrel)..."
pip install spandrel

# torch 2.10 pins cuda-bindings==12.9.4; cuda-python 12.9.7+ conflicts with that.
pip install "cuda-bindings==12.9.4" "cuda-python==12.9.4"

MODEL_DIR="${ROOT}/engines/models"
mkdir -p "${MODEL_DIR}"

download_model() {
  local name="$1"
  local url="$2"
  local target="${MODEL_DIR}/${name}"
  if [[ ! -f "${target}" ]]; then
    echo "Downloading ${name}..."
    curl -L "${url}" -o "${target}"
  fi
}

download_model "RealESRGAN_x2plus.pth" \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
download_model "RealESRGAN_x4plus.pth" \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

echo "Verifying Maxine VSR import..."
python - <<'PY'
import torch
from nvvfx import VideoSuperRes

if not torch.cuda.is_available():
    raise SystemExit("CUDA required for Maxine VSR")

frame = torch.rand(3, 256, 256, device="cuda", dtype=torch.float32)
with VideoSuperRes(quality=VideoSuperRes.QualityLevel.MEDIUM) as vsr:
    vsr.output_width = 512
    vsr.output_height = 512
    vsr.load()
    out = torch.from_dlpack(vsr.run(frame).image).clone()
print(f"Maxine VSR ok: {tuple(frame.shape)} -> {tuple(out.shape)}")
PY

echo "Upscaler deps ready (maxine-vsr + realesrgan fallback)"
