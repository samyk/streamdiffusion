#!/usr/bin/env bash
# shellcheck disable=SC2034
# Exports SDTD_* vars from config/web.toml (override path with SDTD_WEB_CONFIG).

sdtd_load_web_config() {
  local root config
  root="$(sdtd_bridge_root)"
  config="${SDTD_WEB_CONFIG:-${root}/config/web.toml}"

  if [[ ! -f "${config}" ]]; then
    echo "Web config not found: ${config}" >&2
    return 1
  fi

  eval "$(
    cd "${root}" && source .venv/bin/activate 2>/dev/null || true
    python - "${config}" "${root}" <<'PY'
import sys
from pathlib import Path

from streamdiffusion_td_bridge.web_config import load_web_config

cfg_path = Path(sys.argv[1])
root = Path(sys.argv[2])
cfg = load_web_config(cfg_path, root=root)

def emit(key: str, value) -> None:
    text = str(value).replace("'", "'\\''")
    print(f"{key}='{text}'")

emit("SDTD_WEB_PUBLIC_HOST", cfg.public_host)
emit("SDTD_WEB_PUBLIC_PORT", cfg.public_port)
emit("SDTD_WEB_PUBLIC_URL", cfg.public_url)
emit("SDTD_WEB_HOST", cfg.bind_host)
emit("SDTD_WEB_PORT", cfg.bind_port)
emit("SDTD_WEB_TLS_CERT", cfg.tls_cert)
emit("SDTD_WEB_TLS_KEY", cfg.tls_key)
emit("SDTD_STREAM_ID", cfg.stream_id)
emit("SDTD_DAYDREAM_PORT", cfg.daydream_port)
emit("SDTD_CONTROL_PORT", cfg.control_port)
emit("SDTD_SCREEN_SESSION", cfg.screen_session)
emit("SDTD_PROMPTS_FILE", cfg.prompts_file)
emit("SDTD_PRESET", cfg.preset)
emit("SDTD_WIDTH", cfg.width)
emit("SDTD_HEIGHT", cfg.height)
emit("SDTD_WEB_JPEG_QUALITY", cfg.jpeg_quality)
emit("SDTD_DEFAULT_PROMPT", cfg.prompt)
emit("SDTD_UPSCALE", 1 if cfg.upscale_enabled else 0)
emit("SDTD_UPSCALE_FACTOR", cfg.upscale_factor)
emit("SDTD_UPSCALE_METHOD", cfg.upscale_method)
emit("SDTD_UPSCALE_MAXINE_QUALITY", cfg.upscale_maxine_quality)
emit("SDTD_UPSCALE_HALF", 1 if cfg.upscale_half else 0)
emit("SDTD_ACCELERATION", cfg.acceleration)
emit("SDTD_ATTENTION_BACKEND", cfg.attention_backend)
emit("SDTD_FLUX_TRANSFORMER_ENGINE", 1 if cfg.flux_transformer_engine else 0)
if cfg.frame_buffer_size is not None:
    emit("SDTD_FRAME_BUFFER_SIZE", cfg.frame_buffer_size)
PY
  )"
}
