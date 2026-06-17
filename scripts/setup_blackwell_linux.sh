#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

"${PYTHON}" -m pip install --upgrade pip "setuptools<82" wheel
source "${ROOT}/scripts/install_pytorch_cu128.sh"
"${PYTHON}" -m pip install diffusers==0.24.0 transformers accelerate fire omegaconf
source "${ROOT}/scripts/install_streamdiffusion_deps.sh"
"${PYTHON}" -m pip install -e ".[ndi]"
# Re-assert matched cu128 wheels if other packages pulled a different torch build.
source "${ROOT}/scripts/install_pytorch_cu128.sh"
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"
"${PYTHON}" -m streamdiffusion_td_bridge.verify_gpu
"${PYTHON}" -m streamdiffusion_td_bridge.verify_inference

