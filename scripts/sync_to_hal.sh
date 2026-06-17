#!/usr/bin/env bash
set -euo pipefail

# Sync local repo to hal.
#
#   ./scripts/sync_to_hal.sh
#   ./scripts/sync_to_hal.sh scripts/ streamdiffusion_td_bridge/

HAL_HOST="${SDTD_HAL_HOST:-samy@hal}"
HAL_ROOT="${SDTD_HAL_ROOT_REMOTE:-/home/samy/c/samysd}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PATHS=("$@")
if [[ ${#PATHS[@]} -eq 0 ]]; then
  PATHS=(
    scripts/
    streamdiffusion_td_bridge/
    touchdesigner/
    README.md
    pyproject.toml
  )
fi

RSYNC_ARGS=(
  -avz
  --delete
  --exclude '.venv/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.git/'
  --exclude 'engines/'
  --exclude 'node_modules/'
)

for path in "${PATHS[@]}"; do
  src="${LOCAL_ROOT}/${path}"
  if [[ ! -e "${src}" ]]; then
    echo "Missing ${src}" >&2
    exit 1
  fi
  dest="${HAL_HOST}:${HAL_ROOT}/${path}"
  if [[ "${path}" == */ ]]; then
    rsync "${RSYNC_ARGS[@]}" "${src}" "${dest}"
  else
    rsync "${RSYNC_ARGS[@]}" "${src}" "${HAL_HOST}:${HAL_ROOT}/$(dirname "${path}")/"
  fi
done

echo "Synced to ${HAL_HOST}:${HAL_ROOT}"
