#!/usr/bin/env bash
# Shared cu132 torch/torchvision/torchaudio pins + install helpers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu132"
CU132_CONSTRAINTS="${ROOT}/.venv/cu132-constraints.txt"

sdtd_resolve_cu132_versions() {
  mapfile -t _PYTORCH_VERSIONS < <("${PYTHON}" "${ROOT}/scripts/resolve_pytorch_cu132_versions.py")
  TORCH_VERSION="${_PYTORCH_VERSIONS[0]:-}"
  TV_VERSION="${_PYTORCH_VERSIONS[1]:-}"
  TA_VERSION="${_PYTORCH_VERSIONS[2]:-}"
  if [[ -z "${TORCH_VERSION}" || -z "${TV_VERSION}" || -z "${TA_VERSION}" ]]; then
    echo "Failed to resolve cu132 torch/torchvision/torchaudio versions." >&2
    return 1
  fi
  mkdir -p "$(dirname "${CU132_CONSTRAINTS}")"
  cat >"${CU132_CONSTRAINTS}" <<EOF
torch==${TORCH_VERSION}
torchvision==${TV_VERSION}
torchaudio==${TA_VERSION}
EOF
  export TORCH_VERSION TV_VERSION TA_VERSION CU132_CONSTRAINTS
}

sdtd_cu132_pip_args() {
  if [[ -f "${CU132_CONSTRAINTS}" ]]; then
    echo -c "${CU132_CONSTRAINTS}"
  fi
}

sdtd_fix_cuda_pathfinder() {
  "${PYTHON}" -m pip install --force-reinstall "cuda-pathfinder>=1.4.2"
}

sdtd_install_cu132_torch() {
  # shellcheck source=cleanup_legacy_cuda_python.sh
  source "${ROOT}/scripts/cleanup_legacy_cuda_python.sh"

  sdtd_resolve_cu132_versions

  echo "Installing matched PyTorch cu132 nightly wheels..."
  echo "  torch=${TORCH_VERSION}"
  echo "  torchvision=${TV_VERSION}"
  echo "  torchaudio=${TA_VERSION}"

  "${PYTHON}" -m pip install "setuptools<82"
  "${PYTHON}" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

  # One transaction: torchvision present before pip checks spandrel.
  # --no-deps avoids cuda-pathfinder~=1.1 downgrade via cuda-bindings.
  "${PYTHON}" -m pip install --pre --no-cache-dir --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TV_VERSION}" \
    "torchaudio==${TA_VERSION}" \
    --no-deps \
    --index-url "${PYTORCH_INDEX}"

  # Lightweight Python deps for torch/torchvision (no torch wheel from PyPI).
  "${PYTHON}" -m pip install \
    sympy networkx jinja2 fsspec filelock typing-extensions mpmath MarkupSafe \
    "numpy>=1.26,<3" pillow 2>/dev/null || true

  sdtd_fix_cuda_pathfinder
  sdtd_verify_cu132_torch
}

sdtd_verify_cu132_torch() {
  # shellcheck source=env_cuda.sh
  source "${ROOT}/scripts/env_cuda.sh"
  "${PYTHON}" - <<'PY'
import importlib.metadata
import sys
import torch

print(f"torch={torch.__version__}")
try:
    tv = importlib.metadata.version("torchvision")
except importlib.metadata.PackageNotFoundError:
    print("torchvision is not installed", file=sys.stderr)
    sys.exit(1)
print(f"torchvision={tv}")
print(f"torchaudio={importlib.metadata.version('torchaudio')}")
print(f"cuda_runtime={torch.version.cuda}")

if "+cu132" not in torch.__version__:
    print("Expected torch+cu132 wheel.", file=sys.stderr)
    sys.exit(1)
if "+cu132" not in tv:
    print("Expected torchvision+cu132 wheel.", file=sys.stderr)
    sys.exit(1)
if not torch.version.cuda or not torch.version.cuda.startswith("13.2"):
    print("Expected CUDA 13.2 runtime from cu132 wheels.", file=sys.stderr)
    sys.exit(1)

# Catch torch/torchvision ABI skew (torchvision::nms missing).
from torchvision.transforms import InterpolationMode  # noqa: F401

print(f"torchvision import ok ({InterpolationMode.BILINEAR})")
PY
}

sdtd_reassert_cu132_torch() {
  if ! sdtd_verify_cu132_torch 2>/dev/null; then
    echo "cu132 torch stack drifted; reinstalling..." >&2
    sdtd_install_cu132_torch
  fi
}

sdtd_pin_inference_stack() {
  echo "Pinning StreamDiffusion Python stack..."
  "${PYTHON}" -m pip install --force-reinstall \
    "diffusers==0.24.0" \
    "huggingface_hub>=0.19.4,<0.26" \
    "transformers==4.48.3" \
    fire omegaconf

  # accelerate depends on torch>=2.0 — pip would install PyPI torch 2.12.1 over cu132.
  "${PYTHON}" -m pip install --force-reinstall "accelerate==1.14.0" --no-deps
  "${PYTHON}" -m pip install psutil packaging 2>/dev/null || true

  sdtd_reassert_cu132_torch
}

sdtd_finalize_venv() {
  sdtd_fix_cuda_pathfinder
  if "${PYTHON}" -m pip show spandrel >/dev/null 2>&1; then
    "${PYTHON}" -m pip install --force-reinstall spandrel --no-deps 2>/dev/null || true
  fi
  sdtd_reassert_cu132_torch
}
