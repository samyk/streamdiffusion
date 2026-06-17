#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"

echo "Installing matched PyTorch cu128 nightly wheels..."
"${PYTHON}" -m pip install "setuptools<82"
"${PYTHON}" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

mapfile -t _PYTORCH_VERSIONS < <("${PYTHON}" "${ROOT}/scripts/resolve_pytorch_cu128_versions.py")
TORCH_VERSION="${_PYTORCH_VERSIONS[0]:-}"
TV_VERSION="${_PYTORCH_VERSIONS[1]:-}"
TA_VERSION="${_PYTORCH_VERSIONS[2]:-}"

if [[ -z "${TORCH_VERSION}" || -z "${TV_VERSION}" || -z "${TA_VERSION}" ]]; then
  echo "Failed to resolve cu128 torch/torchvision/torchaudio versions." >&2
  exit 1
fi

echo "Selected torch=${TORCH_VERSION}"
echo "Selected torchvision=${TV_VERSION}"
echo "Selected torchaudio=${TA_VERSION}"

install_with_deps() {
  "${PYTHON}" -m pip install --pre --no-cache-dir --force-reinstall \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TV_VERSION}" \
    "torchaudio==${TA_VERSION}" \
    --index-url "${PYTORCH_INDEX}"
}

install_without_vision_deps() {
  echo "Nightly wheels are out of sync on the cu128 index; installing torch with pinned vision/audio (--no-deps)." >&2
  "${PYTHON}" -m pip install --pre --no-cache-dir --force-reinstall \
    "torch==${TORCH_VERSION}" \
    --index-url "${PYTORCH_INDEX}"
  "${PYTHON}" -m pip install --pre --no-cache-dir --force-reinstall \
    "torchvision==${TV_VERSION}" \
    "torchaudio==${TA_VERSION}" \
    --no-deps \
    --index-url "${PYTORCH_INDEX}"
}

if ! install_with_deps; then
  "${PYTHON}" -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  install_without_vision_deps
fi

echo "Verifying PyTorch CUDA build..."
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"
"${PYTHON}" - <<'PY'
import importlib.metadata
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torchvision={importlib.metadata.version('torchvision')}")
print(f"torchaudio={importlib.metadata.version('torchaudio')}")
print(f"cuda_runtime={torch.version.cuda}")
if not torch.version.cuda or not torch.version.cuda.startswith("12.8"):
    print(
        "Expected a CUDA 12.8 PyTorch build for Blackwell. "
        "Got mismatched torch/torchvision wheels.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
