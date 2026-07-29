#!/usr/bin/env bash
set -euo pipefail

# Redirect TCP :443 -> :8790 on hal (LAN + local access to sdtd mobile web).
#
#   sudo ./scripts/setup_hal_web_port_forward.sh install
#   sudo ./scripts/setup_hal_web_port_forward.sh remove
#   ./scripts/setup_hal_web_port_forward.sh status
#
# Env:
#   SDTD_WEB_PUBLIC_PORT=443
#   SDTD_WEB_BIND_PORT=8790

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=load_web_config.sh disable=SC1091
source "${SCRIPT_DIR}/load_web_config.sh" 2>/dev/null || true

PUBLIC_PORT="${SDTD_WEB_PUBLIC_PORT:-443}"
BIND_PORT="${SDTD_WEB_BIND_PORT:-${SDTD_WEB_PORT:-8790}}"
NFT_TABLE="sdtd_web"

load_ports_from_config() {
  local root config
  root="$(cd "${SCRIPT_DIR}/.." && pwd)"
  config="${SDTD_WEB_CONFIG:-${root}/config/web.toml}"
  if [[ ! -f "${config}" ]]; then
    return 0
  fi
  eval "$(
    cd "${root}" && source .venv/bin/activate 2>/dev/null || true
    python - "${config}" <<'PY'
import sys
from pathlib import Path
from streamdiffusion_td_bridge.web_config import load_web_config

cfg = load_web_config(Path(sys.argv[1]), root=Path(sys.argv[1]).parent.parent)
print(f"SDTD_WEB_PUBLIC_PORT={cfg.public_port}")
print(f"SDTD_WEB_BIND_PORT={cfg.bind_port}")
PY
  )" || true
  PUBLIC_PORT="${SDTD_WEB_PUBLIC_PORT:-443}"
  BIND_PORT="${SDTD_WEB_BIND_PORT:-8790}"
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root: sudo $0 $*" >&2
    exit 1
  fi
}

install_rules() {
  require_root "${1:-install}"
  load_ports_from_config

  nft delete table ip "${NFT_TABLE}" 2>/dev/null || true
  nft -f - <<EOF
table ip ${NFT_TABLE} {
  chain prerouting {
    type nat hook prerouting priority dstnat - 10; policy accept;
    # Only hijack traffic destined for this host — not Docker FORWARD to the internet.
    fib daddr type local tcp dport ${PUBLIC_PORT} redirect to :${BIND_PORT}
  }

  chain output {
    type nat hook output priority dstnat - 10; policy accept;
    fib daddr type local tcp dport ${PUBLIC_PORT} redirect to :${BIND_PORT}
  }
}
EOF

  echo "Installed ${PUBLIC_PORT} -> ${BIND_PORT} redirect (table ip ${NFT_TABLE})"
}

remove_rules() {
  require_root "${1:-remove}"
  if nft delete table ip "${NFT_TABLE}" 2>/dev/null; then
    echo "Removed table ip ${NFT_TABLE}"
  else
    echo "Table ip ${NFT_TABLE} not present"
  fi
}

show_status() {
  load_ports_from_config
  echo "Configured redirect: ${PUBLIC_PORT} -> ${BIND_PORT}"
  if nft list table ip "${NFT_TABLE}" 2>/dev/null; then
    echo ""
    ss -tlnp | grep -E ":${PUBLIC_PORT}|:${BIND_PORT}" || true
  else
    echo "Table ip ${NFT_TABLE} not installed"
    exit 1
  fi
}

install_systemd() {
  require_root install-systemd
  local root unit_src unit_dest
  root="$(cd "${SCRIPT_DIR}/.." && pwd)"
  unit_src="${SCRIPT_DIR}/sdtd-web-port-forward.service"
  unit_dest="/etc/systemd/system/sdtd-web-port-forward.service"

  sed "s|@REPO_ROOT@|${root}|g" "${unit_src}" > /tmp/sdtd-web-port-forward.service
  install -m 644 /tmp/sdtd-web-port-forward.service "${unit_dest}"
  systemctl daemon-reload
  systemctl enable --now sdtd-web-port-forward.service
  echo "Enabled sdtd-web-port-forward.service"
}

case "${1:-status}" in
  install) install_rules install ;;
  remove) remove_rules remove ;;
  status) show_status ;;
  install-systemd) install_systemd ;;
  *)
    echo "Usage: $0 {install|remove|status|install-systemd}" >&2
    exit 1
    ;;
esac
