#!/usr/bin/env bash
# TensorRT + polygraphy for StreamDiffusion (cu132 / CUDA 13.x).
#
# StreamDiffusion's `python -m streamdiffusion.tools.install-tensorrt` only handles
# CUDA 11/12 and exits early on torch 13.x — use this script instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=pytorch_cu132_env.sh
source "${ROOT}/scripts/pytorch_cu132_env.sh"

NGC_INDEX="https://pypi.ngc.nvidia.com"

CUDA_MAJOR="$("${PYTHON}" - <<'PY'
import torch

print(torch.version.cuda.split(".")[0])
PY
)"

CUDA_PY_VER="$("${PYTHON}" - <<'PY'
import torch

v = torch.version.cuda or ""
parts = v.split(".")
if len(parts) < 2:
    raise SystemExit("torch.version.cuda is missing or invalid")
print(f"{parts[0]}.{parts[1]}.0")
PY
)"

echo "Installing TensorRT stack for CUDA ${CUDA_MAJOR}.x (torch $(python -c 'import torch; print(torch.__version__)'))..."

echo "  cuda-python==${CUDA_PY_VER} (StreamDiffusion TRT utilities: from cuda import cudart)"
if ! "${PYTHON}" -m pip install "cuda-python==${CUDA_PY_VER}"; then
  echo "  cuda-python meta-package failed; trying cuda-bindings==${CUDA_PY_VER}" >&2
  "${PYTHON}" -m pip install "cuda-bindings==${CUDA_PY_VER}"
fi

echo "  cuda import shim (cuda-python 13.x moved cudart to cuda.bindings.runtime)"
"${PYTHON}" -m streamdiffusion_td_bridge.cuda_compat

if [[ "${CUDA_MAJOR}" -ge 13 ]]; then
  echo "  tensorrt-cu13 (PyPI)"
  "${PYTHON}" -m pip install --upgrade tensorrt-cu13
elif [[ "${CUDA_MAJOR}" == 12 ]]; then
  echo "  tensorrt-cu12 (PyPI)"
  "${PYTHON}" -m pip install --upgrade tensorrt-cu12
else
  echo "Unsupported CUDA major ${CUDA_MAJOR}. Need CUDA 12+ / cu132 torch." >&2
  exit 1
fi

echo "  polygraphy + onnx-graphsurgeon + onnxscript"
if ! "${PYTHON}" -m pip install "polygraphy>=0.47.1" "onnx-graphsurgeon>=0.3.26" "onnxscript>=0.1.0" \
  --extra-index-url "${NGC_INDEX}"; then
  "${PYTHON}" -m pip install "polygraphy>=0.47.1" "onnx-graphsurgeon>=0.3.26" "onnxscript>=0.1.0"
fi

sdtd_reassert_cu132_torch

echo "Verifying TensorRT imports..."
"${PYTHON}" -m streamdiffusion_td_bridge.verify_tensorrt
