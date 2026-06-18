#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=pytorch_cu132_env.sh
source "${ROOT}/scripts/pytorch_cu132_env.sh"

STREAMDIFFUSION_URL="git+https://github.com/cumulo-autumn/StreamDiffusion.git@main"
CONSTRAINT_ARGS=()
if [[ -f "${CU132_CONSTRAINTS}" ]]; then
  CONSTRAINT_ARGS=(-c "${CU132_CONSTRAINTS}")
fi

echo "Installing StreamDiffusion base package (without changing torch stack)..."
"${PYTHON}" -m pip install "streamdiffusion @ ${STREAMDIFFUSION_URL}" --no-deps "${CONSTRAINT_ARGS[@]}"

echo "Pinning huggingface_hub/transformers for diffusers 0.24 compatibility..."
"${PYTHON}" -m pip install "huggingface_hub>=0.19.4,<0.26" "transformers==4.48.3"

echo "Installing TensorRT companion packages..."
"${PYTHON}" -m pip install "protobuf>=4.25.1,<5" colored
"${PYTHON}" -m pip install "onnx>=1.16.0" "onnxruntime>=1.20.0" "onnxscript>=0.1.0"

echo "Installing xformers fallback acceleration..."
if ! "${PYTHON}" -m pip install xformers --no-deps; then
  echo "WARNING: xformers install failed. tensorrt still works." >&2
else
  if "${PYTHON}" - <<'PY' 2>/dev/null; then
import torch
from xformers.ops import memory_efficient_attention
q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
memory_efficient_attention(q, q, q)
print("xformers fp16 OK")
PY
    echo "xformers verified"
  fi
fi

sdtd_reassert_cu132_torch

echo "Installing NVIDIA TensorRT tooling (CUDA 13.x / cu132)..."
if ! "${ROOT}/scripts/install_tensorrt_deps.sh"; then
  echo "WARNING: TensorRT dependency install failed." >&2
  echo "Run ./scripts/install_tensorrt_deps.sh manually for details." >&2
fi

sdtd_reassert_cu132_torch
