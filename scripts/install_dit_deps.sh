#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "Installing DiT / SD3.5 stack (diffusers main + updated transformers)..."
pip install "git+https://github.com/huggingface/diffusers.git"
pip install "transformers>=4.51" "accelerate>=1.4" "safetensors" "sentencepiece" "protobuf"
pip install "huggingface-hub>=1.0"

echo "Verifying SD3.5 pipeline import..."
python - <<'PY'
from diffusers import StableDiffusion3Img2ImgPipeline
import diffusers

print(f"diffusers={diffusers.__version__}")
print(f"StableDiffusion3Img2ImgPipeline={StableDiffusion3Img2ImgPipeline}")
PY

cat <<'EOF'

DiT / SD3.5 deps installed.

Presets:
  sd35_medium_fast
  sd35_medium_quality
  sd35_large_fast

Launch example:
  sdtd-bridge --preset sd35_medium_fast --attention-backend auto --width 960 --height 544

Also run:
  ./scripts/install_attention_deps.sh

EOF
