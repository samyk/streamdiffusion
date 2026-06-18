from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F

PERSON_CLASS_ID = 15  # COCO "person"


def _feather_mask_tensor(mask: torch.Tensor, feather: float) -> torch.Tensor:
    if feather <= 0.0:
        return mask
    kernel = max(3, int(feather * 2) | 1)
    pad = kernel // 2
    blurred = mask.unsqueeze(0).unsqueeze(0)
    blurred = F.pad(blurred, (pad, pad, pad, pad), mode="replicate")
    blurred = F.avg_pool2d(blurred, kernel_size=kernel, stride=1)
    return blurred.squeeze(0).squeeze(0)


class PersonSegmenter(Protocol):
    method: str

    def segment_mask(self, rgb: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...


class _MaxineGreenScreenSegmenter:
    """Optional Maxine AI Green Screen when nvvfx exposes it."""

    _SELECTOR = "AIGreenScreen"

    def __init__(self, *, device: int = 0, feather: float = 3.0) -> None:
        from nvvfx.effects.base import Effect

        class GreenScreen(Effect):
            _SELECTOR = _MaxineGreenScreenSegmenter._SELECTOR

            def run(self, input_array: Any, **kwargs: Any) -> Any:
                return self._effect.run(input_array)

        self.feather = max(0.0, float(feather))
        self._effect = GreenScreen(device=device)
        self._effect.load()
        self._input: torch.Tensor | None = None
        self._input_shape: tuple[int, int] | None = None
        self._method = "maxine-aigs"
        print(f"[segmentation] Maxine AI Green Screen ready feather={self.feather}")

    @property
    def method(self) -> str:
        return self._method

    def close(self) -> None:
        if self._effect is not None:
            self._effect.close()
            self._effect = None

    def _ensure_input(self, rgb: np.ndarray) -> torch.Tensor:
        height, width = rgb.shape[:2]
        shape = (height, width)
        if self._input is None or self._input_shape != shape:
            self._input = torch.empty(3, height, width, device="cuda", dtype=torch.float32)
            self._input_shape = shape
        cpu = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0)
        self._input.copy_(cpu, non_blocking=True)
        return self._input

    def segment_mask(self, rgb: np.ndarray) -> np.ndarray:
        assert self._effect is not None
        tensor = self._ensure_input(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8))
        result = self._effect.run(tensor)
        mask_u8 = torch.from_dlpack(result.image).clone()
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[0]
        mask = mask_u8.float().div(255.0)
        mask = _feather_mask_tensor(mask, self.feather)
        return np.ascontiguousarray(mask.cpu().numpy(), dtype=np.float32)


class CudaPersonSegmenter:
    """CUDA person mask via DeepLabV3 (Blackwell fp16 + optional torch.compile)."""

    def __init__(self, *, device: int = 0, feather: float = 3.0, compile_model: bool = True) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA person segmentation requires a CUDA GPU")

        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        self.device = torch.device(f"cuda:{device}")
        self.feather = max(0.0, float(feather))
        self._model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        ).to(self.device)
        self._model.eval()
        self._model.half()
        if compile_model:
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
            except Exception:
                pass
        self._input: torch.Tensor | None = None
        self._input_shape: tuple[int, int] | None = None
        self._method = "cuda-deeplab"
        print(f"[segmentation] CUDA DeepLab person mask ready feather={self.feather}")

    @property
    def method(self) -> str:
        return self._method

    def close(self) -> None:
        self._model = None
        self._input = None

    def _ensure_input(self, rgb: np.ndarray) -> torch.Tensor:
        height, width = rgb.shape[:2]
        shape = (height, width)
        if self._input is None or self._input_shape != shape:
            self._input = torch.empty(3, height, width, device=self.device, dtype=torch.float16)
            self._input_shape = shape
        cpu = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0).half()
        self._input.copy_(cpu, non_blocking=True)
        return self._input

    def segment_mask(self, rgb: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("segmenter is closed")
        source = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        tensor = self._ensure_input(source)
        with torch.inference_mode():
            logits = self._model(tensor.unsqueeze(0))["out"]
            probs = logits.softmax(dim=1)[0, PERSON_CLASS_ID]
        mask = probs.float()
        mask = _feather_mask_tensor(mask, self.feather)
        return np.ascontiguousarray(mask.cpu().numpy(), dtype=np.float32)


def _try_maxine_segmenter(*, device: int, feather: float) -> PersonSegmenter | None:
    try:
        return _MaxineGreenScreenSegmenter(device=device, feather=feather)
    except Exception as exc:
        print(f"[segmentation] Maxine AI Green Screen unavailable ({exc}); using CUDA DeepLab")
        return None


def create_person_segmenter(
    *,
    enabled: bool,
    feather: float = 3.0,
    device: int = 0,
    backend: str = "auto",
) -> PersonSegmenter | None:
    if not enabled:
        return None
    backend = str(backend or "auto").strip().lower()
    if backend in ("maxine", "maxine-aigs", "aigs"):
        segmenter = _try_maxine_segmenter(device=device, feather=feather)
        if segmenter is not None:
            return segmenter
    if backend == "maxine":
        raise RuntimeError("Maxine AI Green Screen is not available in this nvidia-vfx build")
    return CudaPersonSegmenter(device=device, feather=feather, compile_model=True)
