#!/usr/bin/env bash
set -euo pipefail

# Launch one bridge instance on hal.
#
#   ./scripts/run_bridge_instance.sh a "my prompt"
#   ./scripts/run_bridge_instance.sh b "second prompt"
#
# Env overrides: SDTD_PRESET, SDTD_WIDTH, SDTD_HEIGHT, SDTD_HAL_ROOT

INSTANCE="${1:-a}"
PROMPT="${2:-cybernetic botanical glass sculpture}"
ROOT="${SDTD_HAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

case "${INSTANCE}" in
  a|A)
    SESSION="${SDTD_SCREEN_SESSION:-sdtd-bridge}"
    INPUT_NAME="${SDTD_INPUT_NAME:-td_streamdiffusion_in}"
    OUTPUT_NAME="${SDTD_OUTPUT_NAME:-streamdiffusion_out}"
    STREAM_ID="${SDTD_STREAM_ID:-remote-1}"
    DAYDREAM_PORT="${SDTD_DAYDREAM_PORT:-8780}"
    CONTROL_PORT="${SDTD_CONTROL_PORT:-8765}"
    ;;
  b|B)
    SESSION="${SDTD_SCREEN_SESSION:-sdtd-bridge-b}"
    INPUT_NAME="${SDTD_INPUT_NAME:-td_streamdiffusion_in_b}"
    OUTPUT_NAME="${SDTD_OUTPUT_NAME:-streamdiffusion_out_b}"
    STREAM_ID="${SDTD_STREAM_ID:-remote-2}"
    DAYDREAM_PORT="${SDTD_DAYDREAM_PORT:-8781}"
    CONTROL_PORT="${SDTD_CONTROL_PORT:-8766}"
    ;;
  *)
    echo "Unknown instance ${INSTANCE}. Use a or b." >&2
    exit 1
    ;;
esac

PRESET="${SDTD_PRESET:-sdxl_turbo_fast}"
WIDTH="${SDTD_WIDTH:-768}"
HEIGHT="${SDTD_HEIGHT:-768}"
UPSCALE="${SDTD_UPSCALE:-0}"
UPSCALE_FACTOR="${SDTD_UPSCALE_FACTOR:-2}"
UPSCALE_METHOD="${SDTD_UPSCALE_METHOD:-maxine-vsr}"
UPSCALE_HALF="${SDTD_UPSCALE_HALF:-1}"
UPSCALE_MAXINE_QUALITY="${SDTD_UPSCALE_MAXINE_QUALITY:-medium}"

if [[ "${INSTANCE}" == "a" || "${INSTANCE}" == "A" ]]; then
  systemctl --user stop sdtd-bridge.service 2>/dev/null || true
fi

screen -S "${SESSION}" -X quit 2>/dev/null || true
pkill -f "sdtd-bridge.*--input-name ${INPUT_NAME} --" 2>/dev/null || true
sleep 1

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
  echo "  NDI in:  ${INPUT_NAME}"
  echo "  NDI out: ${OUTPUT_NAME}"
  if [[ "${UPSCALE}" == "1" ]]; then
    echo "  Upscale: x${UPSCALE_FACTOR} (${UPSCALE_METHOD}$([[ "${UPSCALE_METHOD}" == "maxine-vsr" ]] && echo ", quality ${UPSCALE_MAXINE_QUALITY}")$([[ "${UPSCALE_HALF}" == "1" && "${UPSCALE_METHOD}" == "realesrgan" ]] && echo ', fp16'))"
    if [[ "${WIDTH}" -ge 768 && "${UPSCALE_METHOD}" == "realesrgan" ]]; then
      echo "  Tip: SDTD_UPSCALE_METHOD=maxine-vsr or SDTD_WIDTH=512 for better fps"
    fi
  fi
  echo "  API:     :${DAYDREAM_PORT}/v1/streams/${STREAM_ID}"
  echo "Attach: ssh samy@hal -t 'screen -r ${SESSION}'"
else
  echo "Failed to start screen session '${SESSION}'" >&2
  echo "Running bridge once to capture the error..." >&2
  bash -lc "${RUN_CMD}" 2>&1 | tail -40 >&2 || true
  exit 1
fi
