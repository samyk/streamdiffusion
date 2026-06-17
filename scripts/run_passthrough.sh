#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"
sdtd-bridge \
  --passthrough-test \
  --input-name td_streamdiffusion_in \
  --output-name streamdiffusion_out \
  --width 512 \
  --height 512

