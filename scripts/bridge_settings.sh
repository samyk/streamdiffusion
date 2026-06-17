#!/usr/bin/env bash

# Shared bridge settings resolution + printing for launch scripts.
# Launch defaults: streamdiffusion_td_bridge/defaults.py (HAL_BRIDGE_LAUNCH_DEFAULTS)

sdtd_bridge_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}")/.." && pwd)"
  echo "${SDTD_HAL_ROOT:-${script_dir}}"
}

sdtd_load_python_defaults() {
  local root
  root="$(sdtd_bridge_root)"
  SDTD_DEFAULT_PROMPT=""
  SDTD_DEFAULT_NEGATIVE_PROMPT=""
  SDTD_DEFAULT_PRESET=""
  SDTD_DEFAULT_WIDTH=""
  SDTD_DEFAULT_HEIGHT=""
  SDTD_DEFAULT_UPSCALE=""
  SDTD_DEFAULT_UPSCALE_FACTOR=""
  SDTD_DEFAULT_UPSCALE_METHOD=""
  SDTD_DEFAULT_UPSCALE_HALF=""
  SDTD_DEFAULT_UPSCALE_MAXINE_QUALITY=""
  SDTD_DEFAULT_FRAME_BUFFER_SIZE=""
  SDTD_DEFAULT_FLUX_TRANSFORMER_ENGINE=""
  SDTD_DEFAULT_ACCELERATION=""

  if [[ ! -f "${root}/.venv/bin/activate" ]]; then
    return 0
  fi

  local line key val
  while IFS='=' read -r key val; do
    case "${key}" in
      PROMPT) SDTD_DEFAULT_PROMPT="${val}" ;;
      NEGATIVE_PROMPT) SDTD_DEFAULT_NEGATIVE_PROMPT="${val}" ;;
      PRESET) SDTD_DEFAULT_PRESET="${val}" ;;
      WIDTH) SDTD_DEFAULT_WIDTH="${val}" ;;
      HEIGHT) SDTD_DEFAULT_HEIGHT="${val}" ;;
      UPSCALE) SDTD_DEFAULT_UPSCALE="${val}" ;;
      UPSCALE_FACTOR) SDTD_DEFAULT_UPSCALE_FACTOR="${val}" ;;
      UPSCALE_METHOD) SDTD_DEFAULT_UPSCALE_METHOD="${val}" ;;
      UPSCALE_HALF) SDTD_DEFAULT_UPSCALE_HALF="${val}" ;;
      UPSCALE_MAXINE_QUALITY) SDTD_DEFAULT_UPSCALE_MAXINE_QUALITY="${val}" ;;
      FRAME_BUFFER_SIZE) SDTD_DEFAULT_FRAME_BUFFER_SIZE="${val}" ;;
      FLUX_TRANSFORMER_ENGINE) SDTD_DEFAULT_FLUX_TRANSFORMER_ENGINE="${val}" ;;
      ACCELERATION) SDTD_DEFAULT_ACCELERATION="${val}" ;;
    esac
  done < <(
    cd "${root}" && source .venv/bin/activate && python - <<'PY'
from streamdiffusion_td_bridge.defaults import HAL_BRIDGE_LAUNCH_DEFAULTS as d

print(f"PROMPT={d['prompt']}")
print(f"NEGATIVE_PROMPT={d['negative_prompt']}")
print(f"PRESET={d['preset']}")
print(f"WIDTH={d['width']}")
print(f"HEIGHT={d['height']}")
print(f"UPSCALE={1 if d['upscale_enabled'] else 0}")
print(f"UPSCALE_FACTOR={d['upscale_factor']}")
print(f"UPSCALE_METHOD={d['upscale_method']}")
print(f"UPSCALE_HALF={1 if d['upscale_half'] else 0}")
print(f"UPSCALE_MAXINE_QUALITY={d['upscale_maxine_quality']}")
print(f"FRAME_BUFFER_SIZE={d['frame_buffer_size']}")
print(f"FLUX_TRANSFORMER_ENGINE={1 if d['flux_transformer_engine'] else 0}")
print(f"ACCELERATION={d['acceleration']}")
PY
  )
}

sdtd_resolve_instance() {
  local instance="${1:-a}"
  local prompt="${2:-}"
  sdtd_load_python_defaults
  case "${instance}" in
    a|A)
      SDTD_RESOLVED_INSTANCE="A"
      SDTD_RESOLVED_SESSION="${SDTD_SCREEN_SESSION:-sdtd-bridge}"
      SDTD_RESOLVED_INPUT_NAME="${SDTD_INPUT_NAME:-td_streamdiffusion_in}"
      SDTD_RESOLVED_OUTPUT_NAME="${SDTD_OUTPUT_NAME:-streamdiffusion_out}"
      SDTD_RESOLVED_STREAM_ID="${SDTD_STREAM_ID:-remote-1}"
      SDTD_RESOLVED_DAYDREAM_PORT="${SDTD_DAYDREAM_PORT:-8780}"
      SDTD_RESOLVED_CONTROL_PORT="${SDTD_CONTROL_PORT:-8765}"
      SDTD_RESOLVED_PRESET="${SDTD_PRESET:-${SDTD_DEFAULT_PRESET:-sd_turbo_fast}}"
      SDTD_RESOLVED_WIDTH="${SDTD_WIDTH:-${SDTD_DEFAULT_WIDTH:-960}}"
      SDTD_RESOLVED_HEIGHT="${SDTD_HEIGHT:-${SDTD_DEFAULT_HEIGHT:-540}}"
      ;;
    b|B)
      SDTD_RESOLVED_INSTANCE="B"
      SDTD_RESOLVED_SESSION="${SDTD_SCREEN_SESSION:-sdtd-bridge-b}"
      SDTD_RESOLVED_INPUT_NAME="${SDTD_INPUT_NAME:-td_streamdiffusion_in_b}"
      SDTD_RESOLVED_OUTPUT_NAME="${SDTD_OUTPUT_NAME:-streamdiffusion_out_b}"
      SDTD_RESOLVED_STREAM_ID="${SDTD_STREAM_ID:-remote-2}"
      SDTD_RESOLVED_DAYDREAM_PORT="${SDTD_DAYDREAM_PORT:-8781}"
      SDTD_RESOLVED_CONTROL_PORT="${SDTD_CONTROL_PORT:-8766}"
      SDTD_RESOLVED_PRESET="${SDTD_PRESET_B:-${SDTD_PRESET:-${SDTD_DEFAULT_PRESET:-sdxl_turbo_fast}}}"
      SDTD_RESOLVED_WIDTH="${SDTD_WIDTH_B:-${SDTD_WIDTH:-${SDTD_DEFAULT_WIDTH:-768}}}"
      SDTD_RESOLVED_HEIGHT="${SDTD_HEIGHT_B:-${SDTD_HEIGHT:-${SDTD_DEFAULT_HEIGHT:-768}}}"
      ;;
    *)
      echo "Unknown instance ${instance}. Use a or b." >&2
      return 1
      ;;
  esac

  SDTD_RESOLVED_PROMPT="${prompt:-${SDTD_DEFAULT_PROMPT:-paper comic halftone hero, Ben-Day dots, speech bubble pop-art}}"
  SDTD_RESOLVED_UPSCALE="${SDTD_UPSCALE:-${SDTD_DEFAULT_UPSCALE:-1}}"
  SDTD_RESOLVED_UPSCALE_FACTOR="${SDTD_UPSCALE_FACTOR:-${SDTD_DEFAULT_UPSCALE_FACTOR:-2}}"
  SDTD_RESOLVED_UPSCALE_METHOD="${SDTD_UPSCALE_METHOD:-${SDTD_DEFAULT_UPSCALE_METHOD:-maxine-vsr}}"
  SDTD_RESOLVED_UPSCALE_HALF="${SDTD_UPSCALE_HALF:-${SDTD_DEFAULT_UPSCALE_HALF:-1}}"
  SDTD_RESOLVED_UPSCALE_MAXINE_QUALITY="${SDTD_UPSCALE_MAXINE_QUALITY:-${SDTD_DEFAULT_UPSCALE_MAXINE_QUALITY:-high}}"
  SDTD_RESOLVED_FRAME_BUFFER_SIZE="${SDTD_FRAME_BUFFER_SIZE:-${SDTD_DEFAULT_FRAME_BUFFER_SIZE:-}}"
  SDTD_RESOLVED_FLUX_TRANSFORMER_ENGINE="${SDTD_FLUX_TRANSFORMER_ENGINE:-${SDTD_DEFAULT_FLUX_TRANSFORMER_ENGINE:-1}}"
  SDTD_RESOLVED_ACCELERATION="${SDTD_ACCELERATION:-${SDTD_DEFAULT_ACCELERATION:-none}}"
}

sdtd_load_preset_meta() {
  local root preset
  root="$(sdtd_bridge_root)"
  preset="${SDTD_RESOLVED_PRESET}"

  SDTD_RESOLVED_PIPELINE="unknown"
  SDTD_RESOLVED_MODEL="unknown"
  SDTD_RESOLVED_FRAME_BUFFER_DEFAULT="1"
  SDTD_RESOLVED_T_INDEX_LIST=""

  if [[ ! -f "${root}/.venv/bin/activate" ]]; then
    return 0
  fi

  local meta
  meta="$(
    cd "${root}" && source .venv/bin/activate && PRESET="${preset}" python - <<'PY'
import os
from streamdiffusion_td_bridge.config import PRESETS

preset = os.environ.get("PRESET", "")
entry = PRESETS.get(preset)
if entry is None:
    print("pipeline=unknown")
    print("model=unknown")
    print("frame_buffer_default=1")
else:
    print(f"pipeline={entry.pipeline}")
    print(f"model={entry.model_id_or_path}")
    print(f"frame_buffer_default={entry.frame_buffer_size}")
    print(f"t_index_list={entry.t_index_list}")
    print(f"mode={entry.mode}")
PY
  )" || meta=""

  SDTD_RESOLVED_PIPELINE="$(echo "${meta}" | awk -F= '/^pipeline=/{print $2}')"
  SDTD_RESOLVED_MODEL="$(echo "${meta}" | awk -F= '/^model=/{print $2}')"
  SDTD_RESOLVED_FRAME_BUFFER_DEFAULT="$(echo "${meta}" | awk -F= '/^frame_buffer_default=/{print $2}')"
  SDTD_RESOLVED_T_INDEX_LIST="$(echo "${meta}" | awk -F= '/^t_index_list=/{print $2}')"
  SDTD_RESOLVED_MODE="$(echo "${meta}" | awk -F= '/^mode=/{print $2}')"
}

sdtd_print_settings() {
  local instance="${1:-a}"
  local prompt="${2:-}"

  sdtd_resolve_instance "${instance}" "${prompt}" || return 1
  sdtd_load_preset_meta

  local effective_frame_buffer frame_buffer_note out_width out_height
  local flux_engine_label upscale_label

  effective_frame_buffer="${SDTD_RESOLVED_FRAME_BUFFER_SIZE:-${SDTD_RESOLVED_FRAME_BUFFER_DEFAULT:-1}}"
  if [[ -n "${SDTD_RESOLVED_FRAME_BUFFER_SIZE}" ]]; then
    frame_buffer_note="explicit"
  else
    frame_buffer_note="preset default"
  fi

  if [[ "${SDTD_RESOLVED_UPSCALE}" == "1" ]]; then
    out_width=$((SDTD_RESOLVED_WIDTH * SDTD_RESOLVED_UPSCALE_FACTOR))
    out_height=$((SDTD_RESOLVED_HEIGHT * SDTD_RESOLVED_UPSCALE_FACTOR))
    upscale_label="on x${SDTD_RESOLVED_UPSCALE_FACTOR} (${SDTD_RESOLVED_UPSCALE_METHOD}"
    if [[ "${SDTD_RESOLVED_UPSCALE_METHOD}" == "maxine-vsr" ]]; then
      upscale_label="${upscale_label}, quality ${SDTD_RESOLVED_UPSCALE_MAXINE_QUALITY}"
    elif [[ "${SDTD_RESOLVED_UPSCALE_METHOD}" == "realesrgan" && "${SDTD_RESOLVED_UPSCALE_HALF}" == "1" ]]; then
      upscale_label="${upscale_label}, fp16"
    fi
    upscale_label="${upscale_label})"
  else
    out_width="${SDTD_RESOLVED_WIDTH}"
    out_height="${SDTD_RESOLVED_HEIGHT}"
    upscale_label="off"
  fi

  if [[ "${SDTD_RESOLVED_FLUX_TRANSFORMER_ENGINE}" == "1" ]]; then
    flux_engine_label="on (Blackwell bfloat16 + torch.compile)"
  else
    flux_engine_label="off (float16 eager)"
  fi

  echo ""
  echo "=== sdtd-bridge instance ${SDTD_RESOLVED_INSTANCE} ==="
  echo "  screen:              ${SDTD_RESOLVED_SESSION}"
  echo "  preset:              ${SDTD_RESOLVED_PRESET}"
  echo "  pipeline:            ${SDTD_RESOLVED_PIPELINE:-unknown}"
  echo "  model:               ${SDTD_RESOLVED_MODEL:-unknown}"
  if [[ -n "${SDTD_RESOLVED_MODE:-}" ]]; then
    echo "  mode:                ${SDTD_RESOLVED_MODE}"
  fi
  if [[ -n "${SDTD_RESOLVED_T_INDEX_LIST:-}" ]]; then
    echo "  t_index_list:        ${SDTD_RESOLVED_T_INDEX_LIST}"
  fi
  echo "  prompt:              ${SDTD_RESOLVED_PROMPT}"
  echo "  infer resolution:    ${SDTD_RESOLVED_WIDTH} x ${SDTD_RESOLVED_HEIGHT}"
  echo "  output resolution:   ${out_width} x ${out_height}"
  echo "  frame_buffer_size:   ${effective_frame_buffer} (${frame_buffer_note})"
  echo "  flux_transformer:    ${flux_engine_label}"
  echo "  acceleration:        ${SDTD_RESOLVED_ACCELERATION}"
  echo "  upscale:             ${upscale_label}"
  echo "  NDI in:              ${SDTD_RESOLVED_INPUT_NAME}"
  echo "  NDI out:             ${SDTD_RESOLVED_OUTPUT_NAME}"
  echo "  stream id:           ${SDTD_RESOLVED_STREAM_ID}"
  echo "  REST API:            :${SDTD_RESOLVED_DAYDREAM_PORT}/v1/streams/${SDTD_RESOLVED_STREAM_ID}"
  echo "  WebSocket control:   :${SDTD_RESOLVED_CONTROL_PORT}/control"
  echo ""
}
