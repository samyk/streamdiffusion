#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"
# shellcheck source=bridge_settings.sh
source "${ROOT}/scripts/bridge_settings.sh"

PROMPT="${1:-}"
sdtd_resolve_instance a "${PROMPT}"
PROMPT="${SDTD_RESOLVED_PROMPT}"

"${PYTHON}" -m streamdiffusion_td_bridge.verify_inference

ARGS=(
  --acceleration "${SDTD_RESOLVED_ACCELERATION}"
  --preset "${SDTD_RESOLVED_PRESET}"
  --input-name "${SDTD_RESOLVED_INPUT_NAME}"
  --output-name "${SDTD_RESOLVED_OUTPUT_NAME}"
  --width "${SDTD_RESOLVED_WIDTH}"
  --height "${SDTD_RESOLVED_HEIGHT}"
  --prompt "${PROMPT}"
)
if [[ -n "${SDTD_RESOLVED_FRAME_BUFFER_SIZE}" ]]; then
  ARGS+=(--frame-buffer-size "${SDTD_RESOLVED_FRAME_BUFFER_SIZE}")
fi
if [[ "${SDTD_RESOLVED_FLUX_TRANSFORMER_ENGINE}" == "0" ]]; then
  ARGS+=(--no-flux-transformer-engine)
fi
if [[ "${SDTD_RESOLVED_UPSCALE}" == "1" ]]; then
  ARGS+=(
    --upscale
    --upscale-factor "${SDTD_RESOLVED_UPSCALE_FACTOR}"
    --upscale-method "${SDTD_RESOLVED_UPSCALE_METHOD}"
    --upscale-maxine-quality "${SDTD_RESOLVED_UPSCALE_MAXINE_QUALITY}"
  )
  if [[ "${SDTD_RESOLVED_UPSCALE_HALF}" == "0" ]]; then
    ARGS+=(--no-upscale-half)
  fi
fi

sdtd-bridge "${ARGS[@]}"
