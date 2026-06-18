#!/usr/bin/env bash
# Remove CUDA 12.x pip packages that conflict with cu132 torch (cuda-bindings 13.x).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"

echo "Removing legacy CUDA 12.x Python packages..."
for pkg in cuda-python cuda-bindings; do
  if "${PYTHON}" -m pip show "${pkg}" >/dev/null 2>&1; then
    version="$("${PYTHON}" -m pip show "${pkg}" | awk -F': ' '/^Version:/{print $2}')"
    if [[ "${version}" == 12.* ]]; then
      echo "  uninstall ${pkg}==${version}"
      "${PYTHON}" -m pip uninstall -y "${pkg}" >/dev/null 2>&1 || true
    fi
  fi
done

# Unused here; pins cuda-pathfinder>=1.4.2 and conflicts with torch's cuda-bindings pin.
if "${PYTHON}" -m pip show cuda-core >/dev/null 2>&1; then
  echo "  uninstall cuda-core (not used by sdtd-bridge)"
  "${PYTHON}" -m pip uninstall -y cuda-core >/dev/null 2>&1 || true
fi
