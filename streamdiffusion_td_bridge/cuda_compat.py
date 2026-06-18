from __future__ import annotations

import site
import sys
from pathlib import Path

_SHIM_MARKER = "sdtd cuda-python 13.x compat"
_SHIM_SOURCE = f'''\
"""Backward-compat shim ({_SHIM_MARKER})."""
from cuda.bindings import runtime as cudart

try:
    from cuda.bindings import driver as cuda
except ImportError:
    cuda = None

try:
    from cuda.bindings import nvrtc as nvrtc
except ImportError:
    nvrtc = None

__all__ = ["cuda", "cudart", "nvrtc"]
'''


def _site_packages() -> Path:
    paths = site.getsitepackages()
    if not paths:
        raise RuntimeError("Could not locate site-packages")
    return Path(paths[0])


def install_cuda_cudart_shim(*, force: bool = False) -> Path:
    """Write cuda/__init__.py so `from cuda import cudart` works on cuda-python 13.x."""
    init_path = _site_packages() / "cuda" / "__init__.py"
    if init_path.exists() and not force:
        existing = init_path.read_text(encoding="utf-8")
        if _SHIM_MARKER in existing:
            return init_path
        if existing.strip():
            raise RuntimeError(
                f"Refusing to overwrite existing {init_path}. "
                "Remove it manually or reinstall cuda-python."
            )

    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text(_SHIM_SOURCE, encoding="utf-8")

    # Drop cached namespace module so the new __init__ is picked up.
    sys.modules.pop("cuda", None)
    return init_path


def ensure_cuda_cudart_importable() -> None:
    try:
        from cuda import cudart  # noqa: F401
    except ImportError:
        install_cuda_cudart_shim()
        from cuda import cudart  # noqa: F401


def main() -> None:
    path = install_cuda_cudart_shim(force=True)
    ensure_cuda_cudart_importable()
    print(f"Installed cuda import shim: {path}")


if __name__ == "__main__":
    main()
