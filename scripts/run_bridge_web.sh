#!/usr/bin/env bash
set -euo pipefail

# Phone/browser instance: camera -> StreamDiffusion -> fullscreen JPEG back.
#
#   ./scripts/run_bridge_web.sh
#   ./scripts/run_bridge_web.sh "optional prompt override"
#
# Config: config/web.toml (override with SDTD_WEB_CONFIG)
# Certs:  ./scripts/sync_web_certs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bridge_settings.sh
source "${SCRIPT_DIR}/bridge_settings.sh"
# shellcheck source=load_web_config.sh
source "${SCRIPT_DIR}/load_web_config.sh"

PROMPT="${1:-}"
ROOT="$(sdtd_bridge_root)"
sdtd_load_web_config
sdtd_load_python_defaults
PROMPT_ARG="${PROMPT}"

WEB_HOST="${SDTD_WEB_HOST:-0.0.0.0}"
WEB_PORT="${SDTD_WEB_PORT:-8790}"
WEB_PUBLIC_HOST="${SDTD_WEB_PUBLIC_HOST:-stream.sa.my}"
WEB_PUBLIC_PORT="${SDTD_WEB_PUBLIC_PORT:-443}"
WEB_PUBLIC_URL="${SDTD_WEB_PUBLIC_URL:-https://${WEB_PUBLIC_HOST}/}"
STREAM_ID="${SDTD_STREAM_ID:-phone-1}"
DAYDREAM_PORT="${SDTD_DAYDREAM_PORT:-8782}"
CONTROL_PORT="${SDTD_CONTROL_PORT:-8767}"
SESSION="${SDTD_SCREEN_SESSION:-sdtd-bridge-web}"
PRESET="${SDTD_PRESET:-${SDTD_DEFAULT_PRESET:-sd_turbo_fast}}"
WIDTH="${SDTD_WIDTH:-${SDTD_DEFAULT_WIDTH:-960}}"
HEIGHT="${SDTD_HEIGHT:-${SDTD_DEFAULT_HEIGHT:-536}}"
PROMPT_STATE_FILE="${SDTD_WEB_PROMPT_STATE_FILE:-${ROOT}/.state/web_prompt.txt}"
if [[ -z "${PROMPT}" && -f "${PROMPT_STATE_FILE}" ]]; then
  PROMPT="$(python3 - "${PROMPT_STATE_FILE}" <<'PY'
import sys
from pathlib import Path
print(Path(sys.argv[1]).read_text(encoding="utf-8").strip())
PY
  )"
fi
PROMPT="${PROMPT:-${SDTD_DEFAULT_PROMPT:-paper comic halftone hero, Ben-Day dots, speech bubble pop-art}}"
PROMPTS_FILE="${SDTD_PROMPTS_FILE:-${ROOT}/.prompts.txt}"
WEB_TLS_CERT="${SDTD_WEB_TLS_CERT:-}"
WEB_TLS_KEY="${SDTD_WEB_TLS_KEY:-}"
WEB_JPEG_QUALITY="${SDTD_WEB_JPEG_QUALITY:-72}"

UPSCALE="${SDTD_UPSCALE:-${SDTD_DEFAULT_UPSCALE:-1}}"
UPSCALE_FACTOR="${SDTD_UPSCALE_FACTOR:-${SDTD_DEFAULT_UPSCALE_FACTOR:-2}}"
UPSCALE_METHOD="${SDTD_UPSCALE_METHOD:-${SDTD_DEFAULT_UPSCALE_METHOD:-maxine-vsr}}"
UPSCALE_HALF="${SDTD_UPSCALE_HALF:-${SDTD_DEFAULT_UPSCALE_HALF:-1}}"
UPSCALE_MAXINE_QUALITY="${SDTD_UPSCALE_MAXINE_QUALITY:-${SDTD_DEFAULT_UPSCALE_MAXINE_QUALITY:-high}}"
FRAME_BUFFER_SIZE="${SDTD_FRAME_BUFFER_SIZE:-${SDTD_DEFAULT_FRAME_BUFFER_SIZE:-}}"
FLUX_TRANSFORMER_ENGINE="${SDTD_FLUX_TRANSFORMER_ENGINE:-${SDTD_DEFAULT_FLUX_TRANSFORMER_ENGINE:-1}}"
ATTENTION_BACKEND="${SDTD_ATTENTION_BACKEND:-${SDTD_DEFAULT_ATTENTION_BACKEND:-auto}}"

if [[ -n "${SDTD_ACCELERATION:-}" ]]; then
  ACCELERATION="${SDTD_ACCELERATION}"
else
  ACCELERATION="$(
    cd "${ROOT}" && source .venv/bin/activate && PRESET="${PRESET}" python - <<'PY'
import os
from streamdiffusion_td_bridge.config import PRESETS

print(PRESETS[os.environ["PRESET"]].acceleration)
PY
  )"
fi

if [[ -n "${WEB_TLS_CERT}" && ! -f "${WEB_TLS_CERT}" ]]; then
  echo "TLS cert missing: ${WEB_TLS_CERT} (run ./scripts/sync_web_certs.sh)" >&2
  exit 1
fi
if [[ -n "${WEB_TLS_KEY}" && ! -f "${WEB_TLS_KEY}" ]]; then
  echo "TLS key missing: ${WEB_TLS_KEY} (run ./scripts/sync_web_certs.sh)" >&2
  exit 1
fi

screen -S "${SESSION}" -X quit 2>/dev/null || true
pkill -f "sdtd-bridge.*--stream-id ${STREAM_ID} --" 2>/dev/null || true
sleep 2

echo ""
echo "=== sdtd-bridge web (phone) ==="
echo "  config:              ${SDTD_WEB_CONFIG:-${ROOT}/config/web.toml}"
echo "  screen:              ${SESSION}"
echo "  preset:              ${PRESET}"
echo "  prompt:              ${PROMPT}"
echo "  infer resolution:    ${WIDTH} x ${HEIGHT}"
echo "  stream id:           ${STREAM_ID}"
echo "  public URL:          ${WEB_PUBLIC_URL}"
echo "  bind:                ${WEB_HOST}:${WEB_PORT}"
echo "  TLS cert:            ${WEB_TLS_CERT:-(none)}"
echo "  prompts file:        ${PROMPTS_FILE}"
if [[ -z "${PROMPT_ARG}" && -f "${PROMPT_STATE_FILE}" ]]; then
  echo "  saved custom prompt: ${PROMPT_STATE_FILE}"
fi
echo ""

RUN_CMD="cd '${ROOT}' && source .venv/bin/activate && source scripts/env_cuda.sh && exec sdtd-bridge \
  --acceleration ${ACCELERATION} \
  --attention-backend ${ATTENTION_BACKEND} \
  --preset ${PRESET} \
  --video-backend web \
  --stream-id ${STREAM_ID} \
  --width ${WIDTH} \
  --height ${HEIGHT} \
  --daydream-port ${DAYDREAM_PORT} \
  --control-port ${CONTROL_PORT} \
  --web-host ${WEB_HOST} \
  --web-port ${WEB_PORT} \
  --web-public-host '${WEB_PUBLIC_HOST}' \
  --web-public-port ${WEB_PUBLIC_PORT} \
  --web-jpeg-quality ${WEB_JPEG_QUALITY} \
  --prompts-file '${PROMPTS_FILE}' \
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
else
  RUN_CMD="${RUN_CMD} --no-upscale"
fi
if [[ -n "${WEB_TLS_CERT}" && -n "${WEB_TLS_KEY}" ]]; then
  RUN_CMD="${RUN_CMD} --web-tls-cert '${WEB_TLS_CERT}' --web-tls-key '${WEB_TLS_KEY}'"
fi

screen -dmS "${SESSION}" bash -lc "${RUN_CMD}"
sleep 2
if screen -list | grep -q "\.${SESSION}\s"; then
  echo "Started web bridge in screen '${SESSION}'"
  echo "Open: ${WEB_PUBLIC_URL}"
  echo "Attach: ssh samy@hal -t 'screen -r ${SESSION}'"
else
  echo "Failed to start screen session '${SESSION}'" >&2
  bash -lc "${RUN_CMD}" 2>&1 | tail -40 >&2 || true
  exit 1
fi
