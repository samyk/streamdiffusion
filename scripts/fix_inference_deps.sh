#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .venv/bin/activate ]]; then
  echo "No .venv found. Run ./scripts/setup_blackwell_linux.sh first." >&2
  exit 1
fi

source .venv/bin/activate
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

"${PYTHON}" -m pip install --upgrade pip "setuptools<82" wheel
"${PYTHON}" -m pip install diffusers==0.24.0 transformers accelerate fire omegaconf
source "${ROOT}/scripts/install_streamdiffusion_deps.sh"
"${PYTHON}" -m pip install -e ".[ndi]"
source "${ROOT}/scripts/install_pytorch_cu128.sh"
"${PYTHON}" -m streamdiffusion_td_bridge.verify_inference
