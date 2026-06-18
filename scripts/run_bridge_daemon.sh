#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"
# shellcheck source=bridge_settings.sh
source "${ROOT}/scripts/bridge_settings.sh"

LOG="${SDTD_LOG:-/tmp/sdtd-bridge.log}"
PIDFILE="${SDTD_PIDFILE:-/tmp/sdtd-bridge.pid}"
PROMPT="${1:-}"

sdtd_resolve_instance a "${PROMPT}"
PROMPT="${SDTD_RESOLVED_PROMPT}"

stop_bridge() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
  pkill -f "sdtd-bridge.*td_streamdiffusion_in" 2>/dev/null || true
}

bridge_args() {
  echo \
    --acceleration "${SDTD_RESOLVED_ACCELERATION}" \
    --attention-backend "${SDTD_RESOLVED_ATTENTION_BACKEND}" \
    --preset "${SDTD_RESOLVED_PRESET}" \
    --input-name "${SDTD_RESOLVED_INPUT_NAME}" \
    --output-name "${SDTD_RESOLVED_OUTPUT_NAME}" \
    --width "${SDTD_RESOLVED_WIDTH}" \
    --height "${SDTD_RESOLVED_HEIGHT}" \
    --prompt "${PROMPT}"
  if [[ -n "${SDTD_RESOLVED_FRAME_BUFFER_SIZE}" ]]; then
    echo --frame-buffer-size "${SDTD_RESOLVED_FRAME_BUFFER_SIZE}"
  fi
  if [[ "${SDTD_RESOLVED_FLUX_TRANSFORMER_ENGINE}" == "0" ]]; then
    echo --no-flux-transformer-engine
  fi
  if [[ "${SDTD_RESOLVED_UPSCALE}" == "1" ]]; then
    echo \
      --upscale \
      --upscale-factor "${SDTD_RESOLVED_UPSCALE_FACTOR}" \
      --upscale-method "${SDTD_RESOLVED_UPSCALE_METHOD}" \
      --upscale-maxine-quality "${SDTD_RESOLVED_UPSCALE_MAXINE_QUALITY}"
    if [[ "${SDTD_RESOLVED_UPSCALE_HALF}" == "0" ]]; then
      echo --no-upscale-half
    fi
  fi
}

run_bridge() {
  # shellcheck disable=SC2046
  exec sdtd-bridge $(bridge_args)
}

start_bridge() {
  stop_bridge
  # shellcheck disable=SC2046
  nohup sdtd-bridge $(bridge_args) >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "sdtd-bridge started pid=$(cat "$PIDFILE") log=$LOG"
}

case "${SDTD_ACTION:-start}" in
  stop) stop_bridge ;;
  foreground) run_bridge ;;
  restart) start_bridge ;;
  *) start_bridge ;;
esac
