from __future__ import annotations

from .config import Acceleration


def gpu_capability() -> tuple[int, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def resolve_acceleration(requested: Acceleration | None, preset_default: Acceleration) -> Acceleration:
    accel: Acceleration = requested or preset_default
    # FLUX.2 Klein uses diffusers eager/compile; StreamDiffusion TRT is incompatible.
    if preset_default == "none" and accel == "tensorrt":
        return "none"
    return accel
