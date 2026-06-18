#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=pytorch_cu132_env.sh
source "${ROOT}/scripts/pytorch_cu132_env.sh"

sdtd_install_cu132_torch
