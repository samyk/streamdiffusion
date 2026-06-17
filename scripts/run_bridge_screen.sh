#!/usr/bin/env bash
set -euo pipefail

# Start StreamDiffusion bridge(s) in screen session(s).
#
#   ./scripts/run_bridge_screen.sh                    # instance A only (default)
#   ./scripts/run_bridge_screen.sh "prompt A"
#   ./scripts/run_bridge_screen.sh --dual             # A + B
#   ./scripts/run_bridge_screen.sh --dual "prompt A" "prompt B"
#
# Flags: --dual | -d | --both
# Env:   SDTD_DUAL=1 also enables both (for scripts)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUAL="${SDTD_DUAL:-0}"
PROMPT_A=""
PROMPT_B=""
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dual | -d | --both)
      DUAL=1
      shift
      ;;
    -h | --help)
      sed -n '3,14p' "$0"
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

PROMPT_A="${ARGS[0]:-cybernetic botanical glass sculpture}"
PROMPT_B="${ARGS[1]:-${SDTD_PROMPT_B:-coral reef person covered in anemones and fish}}"

systemctl --user stop sdtd-bridge.service 2>/dev/null || true

"${ROOT}/scripts/run_bridge_instance.sh" a "${PROMPT_A}"

if [[ "${DUAL}" == "1" ]]; then
  SDTD_PRESET="${SDTD_PRESET_B:-${SDTD_PRESET:-sdxl_turbo_fast}}" \
  SDTD_WIDTH="${SDTD_WIDTH_B:-${SDTD_WIDTH:-768}}" \
  SDTD_HEIGHT="${SDTD_HEIGHT_B:-${SDTD_HEIGHT:-768}}" \
    "${ROOT}/scripts/run_bridge_instance.sh" b "${PROMPT_B}"
fi

echo ""
if [[ "${DUAL}" == "1" ]]; then
  echo "Both bridges running:"
else
  echo "Bridge running (instance A only):"
fi
echo "  A: screen -r sdtd-bridge     NDI td_streamdiffusion_in / streamdiffusion_out     API :8780/remote-1"
if [[ "${DUAL}" == "1" ]]; then
  echo "  B: screen -r sdtd-bridge-b   NDI td_streamdiffusion_in_b / streamdiffusion_out_b API :8781/remote-2"
fi
echo "List: screen -ls"
