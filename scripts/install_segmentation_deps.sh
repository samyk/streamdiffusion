#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
# shellcheck source=env_cuda.sh
source "${ROOT}/scripts/env_cuda.sh"

echo "Verifying CUDA person segmentation (torchvision DeepLabV3)..."
python - <<'PY'
import numpy as np
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA required for person segmentation")

from streamdiffusion_td_bridge.person_segmentation import create_person_segmenter

segmenter = create_person_segmenter(enabled=True, feather=3.0, backend="cuda")
method = segmenter.method
frame = np.random.randint(0, 255, (536, 960, 3), dtype=np.uint8)
mask = segmenter.segment_mask(frame)
segmenter.close()
assert mask.shape == (536, 960), mask.shape
assert mask.dtype == np.float32
print(f"Person segmentation ok: backend={method} mask={mask.shape} range=({mask.min():.3f},{mask.max():.3f})")
PY

echo "Segmentation deps ready (CUDA DeepLab; Maxine AIGS auto when nvvfx adds it)"
