#!/usr/bin/env bash
# Deprecated alias — CUDA 13.2 (cu132) is the supported stack now.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "install_pytorch_cu128.sh is deprecated; using install_pytorch_cu132.sh" >&2
# shellcheck source=install_pytorch_cu132.sh
source "${ROOT}/scripts/install_pytorch_cu132.sh"
