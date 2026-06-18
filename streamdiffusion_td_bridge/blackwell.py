from __future__ import annotations

import os


def gpu_capability() -> tuple[int, int] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability(0)


def is_blackwell() -> bool:
    cap = gpu_capability()
    return cap is not None and cap[0] >= 12


def preferred_compute_dtype():
    import torch

    if is_blackwell():
        return torch.bfloat16
    return torch.float16


def tune_cuda_for_inference() -> None:
    """Enable Blackwell-friendly CUDA math defaults once per process."""
    if getattr(tune_cuda_for_inference, "_done", False):
        return
    try:
        import torch
    except ImportError:
        return

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        # Allow cuDNN SDPA fusion on recent stacks when present.
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            try:
                torch.backends.cuda.enable_cudnn_sdp(True)
            except Exception:  # noqa: BLE001
                pass
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    tune_cuda_for_inference._done = True  # type: ignore[attr-defined]
