#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=pytorch_cu132_env.sh
source "${ROOT}/scripts/pytorch_cu132_env.sh"

"${PYTHON}" -m pip install --upgrade pip "setuptools<82" wheel
sdtd_install_cu132_torch
sdtd_pin_inference_stack
source "${ROOT}/scripts/install_streamdiffusion_deps.sh"
"${PYTHON}" -m pip install -e ".[ndi]"
sdtd_reassert_cu132_torch
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"
"${PYTHON}" -m streamdiffusion_td_bridge.verify_gpu
"${PYTHON}" -m streamdiffusion_td_bridge.verify_inference
