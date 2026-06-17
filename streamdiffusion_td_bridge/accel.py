from __future__ import annotations

import warnings

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
    acceleration = requested or preset_default
    if acceleration == "none":
        return "none"

    capability = gpu_capability()
    if capability is None:
        return acceleration

    major, _minor = capability
    if major >= 12 and acceleration in ("xformers", "tensorrt"):
        warnings.warn(
            f"Requested acceleration={acceleration!r} is not supported on "
            f"Blackwell/sm_{major}{_minor} yet; falling back to acceleration='none'.",
            stacklevel=2,
        )
        return "none"
    return acceleration
