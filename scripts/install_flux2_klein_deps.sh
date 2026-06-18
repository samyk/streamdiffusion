#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

echo "Installing FLUX.2 Klein stack (diffusers main + updated transformers)..."
pip install "git+https://github.com/huggingface/diffusers.git"
pip install "transformers>=4.51" "accelerate>=1.4" "safetensors" "sentencepiece" "protobuf"
pip install "huggingface-hub>=1.0"

echo "Verifying Flux2KleinPipeline import..."
python - <<'PY'
from diffusers import Flux2KleinPipeline
import diffusers

print(f"diffusers={diffusers.__version__}")
print(f"Flux2KleinPipeline={Flux2KleinPipeline}")
PY

cat <<'EOF'

FLUX.2 Klein deps installed.

Notes:
- This upgrades diffusers beyond the StreamDiffusion 0.24 pin.
- SD Turbo presets may still work, but if anything breaks run:
    ./scripts/fix_inference_deps.sh
- First FLUX run downloads model weights from Hugging Face (~8B params total).
  Accept the model license on HF for black-forest-labs/FLUX.2-klein-4B first.

Launch example:
  sdtd-bridge --preset flux2_klein_fast --width 768 --height 768 --acceleration none

EOF
