#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"

LOG="${SDTD_LOG:-/tmp/sdtd-bridge.log}"
PIDFILE="${SDTD_PIDFILE:-/tmp/sdtd-bridge.pid}"
PROMPT="${1:-cybernetic botanical glass sculpture}"

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

run_bridge() {
  exec sdtd-bridge \
    --acceleration none \
    --preset sd_turbo_fast \
    --input-name td_streamdiffusion_in \
    --output-name streamdiffusion_out \
    --width 512 \
    --height 512 \
    --prompt "$PROMPT"
}

start_bridge() {
  stop_bridge
  nohup sdtd-bridge \
    --acceleration none \
    --preset sd_turbo_fast \
    --input-name td_streamdiffusion_in \
    --output-name streamdiffusion_out \
    --width 512 \
    --height 512 \
    --prompt "$PROMPT" \
    >>"$LOG" 2>&1 &
  echo $! >"$PIDFILE"
  echo "sdtd-bridge started pid=$(cat "$PIDFILE") log=$LOG"
}

case "${SDTD_ACTION:-start}" in
  stop) stop_bridge ;;
  foreground) run_bridge ;;
  restart) start_bridge ;;
  *) start_bridge ;;
esac
