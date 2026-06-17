from __future__ import annotations

from typing import Any

import numpy as np
import torch

MAXINE_QUALITY_ALIASES = {
    "bicubic": "BICUBIC",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "ultra": "ULTRA",
    "highbitrate_low": "HIGHBITRATE_LOW",
    "highbitrate_medium": "HIGHBITRATE_MEDIUM",
    "highbitrate_high": "HIGHBITRATE_HIGH",
    "highbitrate_ultra": "HIGHBITRATE_ULTRA",
}


def normalize_maxine_quality(quality: str) -> str:
    key = str(quality).strip().lower().replace("-", "_")
    return MAXINE_QUALITY_ALIASES.get(key, "MEDIUM")


class MaxineVsrUpscaler:
    """NVIDIA Maxine Video Super Resolution via the nvidia-vfx Python bindings."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        factor: int = 2,
        quality: str = "medium",
        device: int = 0,
    ) -> None:
        self.enabled = enabled
        self.factor = max(1, int(factor))
        self.quality_name = normalize_maxine_quality(quality)
        self.device = device
        self._vsr: Any | None = None
        self._configured: tuple[int, int] | None = None
        self._input: torch.Tensor | None = None
        self._input_shape: tuple[int, int] | None = None
        self._method_label = "maxine-vsr"

        if not self.enabled or self.factor <= 1:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("Maxine VSR requires CUDA")

        try:
            from nvvfx import VideoSuperRes
        except ImportError as exc:
            raise RuntimeError(
                "nvidia-vfx is not installed. Run: ./scripts/install_upscaler_deps.sh"
            ) from exc

        quality_level = getattr(VideoSuperRes.QualityLevel, self.quality_name)
        self._vsr = VideoSuperRes(quality=quality_level, device=device)
        self._method_label = f"maxine-vsr-{self.quality_name.lower()}"
        self._warmup()
        print(f"[upscaler] Maxine VSR ready ({self._method_label}, output x{self.factor})")

    @property
    def method(self) -> str:
        if not self.enabled or self.factor <= 1:
            return "off"
        return self._method_label

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        if not self.enabled or self.factor <= 1:
            return width, height
        return width * self.factor, height * self.factor

    def close(self) -> None:
        if self._vsr is not None:
            self._vsr.close()
            self._vsr = None

    def _configure(self, height: int, width: int) -> None:
        assert self._vsr is not None
        out_width = width * self.factor
        out_height = height * self.factor
        target = (out_width, out_height)
        if self._configured == target:
            return
        self._vsr.output_width = out_width
        self._vsr.output_height = out_height
        self._vsr.load()
        self._configured = target

    def _ensure_input(self, rgb: np.ndarray) -> torch.Tensor:
        height, width = rgb.shape[:2]
        shape = (height, width)
        if self._input is None or self._input_shape != shape:
            self._input = torch.empty(
                3,
                height,
                width,
                device="cuda",
                dtype=torch.float32,
            )
            self._input_shape = shape
        cpu = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255.0)
        self._input.copy_(cpu, non_blocking=True)
        return self._input

    def _warmup(self) -> None:
        assert self._vsr is not None
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        _ = self.upscale_rgb(dummy)

    def upscale_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if not self.enabled or self.factor <= 1:
            return np.ascontiguousarray(rgb[:, :, :3])
        assert self._vsr is not None

        source = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
        height, width = source.shape[:2]
        self._configure(height, width)
        tensor = self._ensure_input(source)
        result = self._vsr.run(tensor)
        output = torch.from_dlpack(result.image).clone()
        rgb_out = output.permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
        return np.ascontiguousarray(rgb_out.cpu().numpy())
