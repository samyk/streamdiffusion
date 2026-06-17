#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"

"${PYTHON}" -m streamdiffusion_td_bridge.verify_inference
sdtd-bridge \
  --acceleration none \
  --preset sd_turbo_fast \
  --input-name td_streamdiffusion_in \
  --output-name streamdiffusion_out \
  --width 512 \
  --height 512 \
  --prompt "${1:-cybernetic botanical glass sculpture}"

