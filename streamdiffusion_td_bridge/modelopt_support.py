from __future__ import annotations

from pathlib import Path
from typing import Any


def modelopt_available() -> bool:
    try:
        import modelopt.torch.quantization as _mtq  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def maybe_load_modelopt_checkpoint(module: Any, checkpoint: str | None) -> Any:
    """Load a ModelOpt-quantized module checkpoint when provided."""
    if not checkpoint:
        return module
    path = Path(checkpoint)
    if not path.exists():
        print(f"[modelopt] checkpoint not found: {path}; using base weights")
        return module
    try:
        import torch
    except ImportError:
        return module

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        module.load_state_dict(state["state_dict"], strict=False)
    elif isinstance(state, dict):
        module.load_state_dict(state, strict=False)
    print(f"[modelopt] loaded quantized checkpoint: {path}")
    return module


def describe_modelopt_status(*, enabled: bool, checkpoint: str | None) -> str:
    if not enabled:
        return "off"
    if not modelopt_available():
        return "requested (nvidia-modelopt not installed)"
    if checkpoint:
        return f"checkpoint:{checkpoint}"
    return "ready (runtime quant export via ./scripts/install_modelopt_deps.sh)"
