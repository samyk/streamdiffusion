#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "Installing NVIDIA Model Optimizer (optional DiT quant/export)..."
pip install "nvidia-modelopt[all]" || pip install "nvidia-modelopt"

python - <<'PY'
try:
    import modelopt.torch.quantization as mtq
    print("modelopt OK", mtq)
except Exception as exc:
    raise SystemExit(f"modelopt import failed: {exc}")
PY

cat <<'EOF'

ModelOpt installed.

Use with a pre-quantized checkpoint:
  sdtd-bridge --preset sd35_medium_fast --modelopt --modelopt-checkpoint /path/to/quantized.pt

To quantize/export from upstream examples:
  https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/diffusers

EOF
