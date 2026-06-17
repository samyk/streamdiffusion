#!/usr/bin/env bash
set -euo pipefail

# Launch one bridge instance on hal.
#
#   ./scripts/run_bridge_instance.sh a "my prompt"
#   ./scripts/run_bridge_instance.sh b "second prompt"
#
# Env overrides: SDTD_PRESET, SDTD_WIDTH, SDTD_HEIGHT, SDTD_HAL_ROOT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bridge_settings.sh
source "${SCRIPT_DIR}/bridge_settings.sh"

INSTANCE="${1:-a}"
PROMPT="${2:-}"
ROOT="$(sdtd_bridge_root)"

sdtd_resolve_instance "${INSTANCE}" "${PROMPT}"
PROMPT="${SDTD_RESOLVED_PROMPT}"

SESSION="${SDTD_RESOLVED_SESSION}"
INPUT_NAME="${SDTD_RESOLVED_INPUT_NAME}"
OUTPUT_NAME="${SDTD_RESOLVED_OUTPUT_NAME}"
STREAM_ID="${SDTD_RESOLVED_STREAM_ID}"
DAYDREAM_PORT="${SDTD_RESOLVED_DAYDREAM_PORT}"
CONTROL_PORT="${SDTD_RESOLVED_CONTROL_PORT}"
PRESET="${SDTD_RESOLVED_PRESET}"
WIDTH="${SDTD_RESOLVED_WIDTH}"
HEIGHT="${SDTD_RESOLVED_HEIGHT}"
UPSCALE="${SDTD_RESOLVED_UPSCALE}"
UPSCALE_FACTOR="${SDTD_RESOLVED_UPSCALE_FACTOR}"
UPSCALE_METHOD="${SDTD_RESOLVED_UPSCALE_METHOD}"
UPSCALE_HALF="${SDTD_RESOLVED_UPSCALE_HALF}"
UPSCALE_MAXINE_QUALITY="${SDTD_RESOLVED_UPSCALE_MAXINE_QUALITY}"
FRAME_BUFFER_SIZE="${SDTD_RESOLVED_FRAME_BUFFER_SIZE}"
FLUX_TRANSFORMER_ENGINE="${SDTD_RESOLVED_FLUX_TRANSFORMER_ENGINE}"

if [[ "${INSTANCE}" == "a" || "${INSTANCE}" == "A" ]]; then
  systemctl --user stop sdtd-bridge.service 2>/dev/null || true
fi

screen -S "${SESSION}" -X quit 2>/dev/null || true
pkill -f "sdtd-bridge.*--input-name ${INPUT_NAME} --" 2>/dev/null || true
sleep 1

sdtd_print_settings "${INSTANCE}" "${PROMPT}"

RUN_CMD="cd '${ROOT}' && source .venv/bin/activate && source scripts/env_cuda.sh && exec sdtd-bridge \
  --acceleration none \
  --preset ${PRESET} \
  --input-name ${INPUT_NAME} \
  --output-name ${OUTPUT_NAME} \
  --stream-id ${STREAM_ID} \
  --width ${WIDTH} \
  --height ${HEIGHT} \
  --daydream-port ${DAYDREAM_PORT} \
  --control-port ${CONTROL_PORT} \
  --prompt '${PROMPT}'"

if [[ -n "${FRAME_BUFFER_SIZE}" ]]; then
  RUN_CMD="${RUN_CMD} --frame-buffer-size ${FRAME_BUFFER_SIZE}"
fi
if [[ "${FLUX_TRANSFORMER_ENGINE}" == "0" ]]; then
  RUN_CMD="${RUN_CMD} --no-flux-transformer-engine"
fi

if [[ "${UPSCALE}" == "1" ]]; then
  RUN_CMD="${RUN_CMD} --upscale --upscale-factor ${UPSCALE_FACTOR} --upscale-method ${UPSCALE_METHOD} --upscale-maxine-quality ${UPSCALE_MAXINE_QUALITY}"
  if [[ "${UPSCALE_HALF}" == "0" ]]; then
    RUN_CMD="${RUN_CMD} --no-upscale-half"
  fi
fi

screen -dmS "${SESSION}" bash -lc "${RUN_CMD}"
sleep 2
if screen -list | grep -q "\.${SESSION}\s"; then
  echo "Started instance ${INSTANCE} in screen '${SESSION}'"
  echo "Attach: ssh samy@hal -t 'screen -r ${SESSION}'"
else
  echo "Failed to start screen session '${SESSION}'" >&2
  echo "Running bridge once to capture the error..." >&2
  bash -lc "${RUN_CMD}" 2>&1 | tail -40 >&2 || true
  exit 1
fi
