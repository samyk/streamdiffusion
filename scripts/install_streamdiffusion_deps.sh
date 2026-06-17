#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

STREAMDIFFUSION_URL="git+https://github.com/cumulo-autumn/StreamDiffusion.git@main"

echo "Installing StreamDiffusion base package..."
"${PYTHON}" -m pip install "streamdiffusion @ ${STREAMDIFFUSION_URL}"

echo "Pinning huggingface_hub/transformers for diffusers 0.24 compatibility..."
"${PYTHON}" -m pip install "huggingface_hub>=0.19.4,<0.26" "transformers>=4.40,<5"

echo "Installing TensorRT companion packages with Python 3.13-compatible pins..."
# torch 2.10 pins cuda-bindings==12.9.4 on Linux; keep cuda-python on the same line.
"${PYTHON}" -m pip install "protobuf>=4.25.1,<5" "cuda-bindings==12.9.4" "cuda-python==12.9.4" colored
"${PYTHON}" -m pip install "onnx>=1.16.0" "onnxruntime>=1.20.0"

echo "Installing xformers fallback acceleration..."
if ! "${PYTHON}" -m pip install xformers; then
  echo "WARNING: xformers install failed. You can still try tensorrt or acceleration=none." >&2
fi

echo "Installing NVIDIA TensorRT tooling..."
if ! "${PYTHON}" -m streamdiffusion.tools.install-tensorrt; then
  echo "WARNING: StreamDiffusion TensorRT installer failed." >&2
  echo "Try running with: sdtd-bridge --acceleration xformers" >&2
fi

echo "Restoring cuDNN 9 required by PyTorch cu128..."
"${PYTHON}" -m pip install --force-reinstall "nvidia-cudnn-cu12==9.20.0.48"
