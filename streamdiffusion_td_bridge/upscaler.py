from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn.functional as F

from .maxine_upscaler import MaxineVsrUpscaler, normalize_maxine_quality

UpscaleMethod = Literal["bicubic", "realesrgan", "maxine-vsr"]


class Upscaler(Protocol):
    enabled: bool
    factor: int

    @property
    def method(self) -> str: ...

    def output_size(self, width: int, height: int) -> tuple[int, int]: ...

    def upscale_rgb(self, rgb: np.ndarray) -> np.ndarray: ...

MODEL_URLS = {
    2: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    4: "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def normalize_upscale_factor(factor: int) -> int:
    value = int(factor)
    if value <= 1:
        return 1
    if value <= 2:
        return 2
    return 4


def normalize_upscale_method(method: str) -> UpscaleMethod:
    value = str(method).strip().lower().replace("_", "-")
    if value in {"bicubic", "fast", "gpu"}:
        return "bicubic"
    if value in {"maxine-vsr", "maxine", "vsr", "nvidia-vsr", "nvidia", "maxinevsr"}:
        return "maxine-vsr"
    return "realesrgan"


def default_model_path(engine_dir: str, factor: int) -> Path:
    normalized = normalize_upscale_factor(factor)
    return Path(engine_dir) / "models" / f"RealESRGAN_x{normalized}plus.pth"


def ensure_model(path: Path, factor: int) -> Path:
    if path.exists():
        return path
    normalized = normalize_upscale_factor(factor)
    url = MODEL_URLS[normalized]
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[upscaler] downloading RealESRGAN x{normalized} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


class GpuUpscaler:
    """GPU upscaler: Real-ESRGAN (fp16) or fast bicubic on CUDA."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        factor: int = 2,
        method: UpscaleMethod | str = "realesrgan",
        use_half: bool = True,
        model_path: str | None = None,
        engine_dir: str = "engines",
        device: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.factor = normalize_upscale_factor(factor)
        self.requested_method = normalize_upscale_method(method)
        self.use_half = use_half
        self.engine_dir = engine_dir
        self.model_path = Path(model_path) if model_path else default_model_path(engine_dir, self.factor)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Any | None = None
        self._dtype = torch.float32
        self._input: torch.Tensor | None = None
        self._input_shape: tuple[int, int] | None = None
        self._method_label = "bicubic"

        if not self.enabled or self.factor <= 1:
            return

        if self.requested_method == "bicubic" or self.device.type != "cuda":
            if self.device.type != "cuda":
                print("[upscaler] CUDA unavailable; using bicubic upscale on CPU")
            else:
                print(f"[upscaler] using GPU bicubic x{self.factor}")
            self._method_label = "bicubic"
            self._warmup_bicubic()
            return

        try:
            from spandrel import ImageModelDescriptor, ModelLoader
        except ImportError:
            print(
                "[upscaler] spandrel not installed; using bicubic GPU upscale. "
                "Run: pip install spandrel"
            )
            self.requested_method = "bicubic"
            self._method_label = "bicubic"
            self._warmup_bicubic()
            return

        ensure_model(self.model_path, self.factor)
        loaded = ModelLoader().load_from_file(str(self.model_path))
        if not isinstance(loaded, ImageModelDescriptor):
            raise TypeError(f"Unsupported upscale model type: {type(loaded)!r}")

        self._model = loaded.to(self.device).eval()
        if use_half and getattr(self._model, "supports_half", False):
            self._model = self._model.half()
            self._dtype = torch.float16
            self._method_label = "realesrgan-fp16"
        else:
            self._method_label = "realesrgan-fp32"

        self._warmup_model()
        model_scale = int(getattr(self._model, "scale", self.factor))
        print(
            f"[upscaler] loaded {self.model_path.name} on {self.device} "
            f"({self._method_label}, model x{model_scale}, output x{self.factor})"
        )

    @property
    def method(self) -> str:
        if not self.enabled or self.factor <= 1:
            return "off"
        return self._method_label

    def output_size(self, width: int, height: int) -> tuple[int, int]:
        if not self.enabled or self.factor <= 1:
            return width, height
        return width * self.factor, height * self.factor

    def _run_model(self, tensor: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        return self._model.model(tensor)

    def _ensure_input(self, height: int, width: int) -> torch.Tensor:
        shape = (height, width)
        if self._input is None or self._input_shape != shape:
            self._input = torch.empty(
                1,
                3,
                height,
                width,
                device=self.device,
                dtype=self._dtype,
            )
            self._input_shape = shape
        return self._input

    def _copy_rgb_to_input(self, rgb: np.ndarray) -> torch.Tensor:
        height, width = rgb.shape[:2]
        tensor = self._ensure_input(height, width)
        cpu = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        if self._dtype == torch.float16:
            cpu = cpu.half()
        cpu = cpu.div(255.0).unsqueeze(0)
        tensor.copy_(cpu, non_blocking=True)
        return tensor

    def _tensor_to_rgb(self, output: torch.Tensor) -> np.ndarray:
        rgb = output.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).mul(255.0).to(torch.uint8)
        return np.ascontiguousarray(rgb.cpu().numpy())

    def _warmup_bicubic(self) -> None:
        dummy = torch.zeros(1, 3, 64, 64, device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            _ = F.interpolate(
                dummy,
                scale_factor=float(self.factor),
                mode="bicubic",
                align_corners=False,
            )

    def _warmup_model(self) -> None:
        dummy = torch.zeros(1, 3, 64, 64, device=self.device, dtype=self._dtype)
        with torch.inference_mode():
            _ = self._run_model(dummy)

    def upscale_rgb(self, rgb: np.ndarray) -> np.ndarray:
        if not self.enabled or self.factor <= 1:
            return np.ascontiguousarray(rgb[:, :, :3])

        source = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)

        with torch.inference_mode():
            if self._model is not None:
                tensor = self._copy_rgb_to_input(source)
                output = self._run_model(tensor)
                return self._tensor_to_rgb(output)

            tensor = self._copy_rgb_to_input(source).float()
            output = F.interpolate(
                tensor,
                scale_factor=float(self.factor),
                mode="bicubic",
                align_corners=False,
            )
            return self._tensor_to_rgb(output)


def create_upscaler(
    *,
    enabled: bool,
    factor: int,
    method: UpscaleMethod | str,
    use_half: bool,
    model_path: str | None,
    engine_dir: str,
    maxine_quality: str = "medium",
) -> Upscaler | None:
    if not enabled or normalize_upscale_factor(factor) <= 1:
        return None
    normalized = normalize_upscale_method(method)
    if normalized == "maxine-vsr":
        return MaxineVsrUpscaler(
            enabled=True,
            factor=factor,
            quality=maxine_quality,
        )
    return GpuUpscaler(
        enabled=True,
        factor=factor,
        method=normalized,
        use_half=use_half,
        model_path=model_path,
        engine_dir=engine_dir,
    )
