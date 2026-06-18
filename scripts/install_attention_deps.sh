#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.sh
source "${ROOT}/scripts/common.sh"
# shellcheck source=pytorch_cu132_env.sh
source "${ROOT}/scripts/pytorch_cu132_env.sh"

source .venv/bin/activate

echo "Installing optional attention kernels for Blackwell / CUDA 13.x ..."
echo "Torch: $("${PYTHON}" -c 'import torch; print(torch.__version__, torch.version.cuda)')"

ATTN_OK=0

echo ""
echo "==> xformers (Blackwell: uses XFormersAttnProcessor, skips broken fp32 self-test)"
if "${PYTHON}" -m pip install xformers --no-deps; then
  if "${PYTHON}" - <<'PY'; then
import torch
from xformers.ops import memory_efficient_attention

q = torch.randn(1, 64, 8, 64, device="cuda", dtype=torch.float16)
_ = memory_efficient_attention(q, q, q)
print("xformers fp16 smoke test OK")
PY
    echo "xformers verified on GPU"
  else
    echo "WARNING: xformers installed but fp16 smoke test failed." >&2
  fi
else
  echo "WARNING: xformers wheel install failed." >&2
fi

echo ""
echo "==> flash-attn (FlashAttention 2/3)"
export FLASH_ATTENTION_FORCE_BUILD="${FLASH_ATTENTION_FORCE_BUILD:-0}"
export MAX_JOBS="${MAX_JOBS:-8}"
if "${PYTHON}" -m pip install flash-attn --no-build-isolation; then
  ATTN_OK=1
else
  echo "WARNING: flash-attn build/install failed. Try FLASH_ATTENTION_FORCE_BUILD=1 on hal." >&2
fi

echo ""
echo "==> SageAttention (Blackwell sm_120; build from source on Linux)"
if "${PYTHON}" -m pip install "git+https://github.com/thu-ml/SageAttention.git" --no-build-isolation; then
  ATTN_OK=1
else
  echo "WARNING: SageAttention build failed. See https://github.com/thu-ml/SageAttention" >&2
fi

sdtd_reassert_cu132_torch

echo ""
"${PYTHON}" -m streamdiffusion_td_bridge.verify_attention || true

cat <<'EOF'

Attention deps attempted.

Notes:
- TensorRT SD Turbo path replaces the UNet entirely; attention kernels apply to:
  * SD eager/xformers fallback paths
  * FLUX.2 Klein / SD3.5 DiT transformer paths
- auto backend order: flash -> sage -> xformers -> sdpa
- Verify at runtime: curl http://hal:8780/v1/streams/remote-1 | jq .runtime.extra

EOF
