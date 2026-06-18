from __future__ import annotations

import importlib
import sys
import types

# StreamDiffusion 0.1.1 expects diffusers 0.24 module paths. Newer diffusers
# (required for FLUX.2 Klein) moved models under autoencoders/ and unets/.
_SHIMS: tuple[tuple[str, str], ...] = (
    ("diffusers.models.autoencoder_tiny", "diffusers.models.autoencoders.autoencoder_tiny"),
    ("diffusers.models.unet_2d_condition", "diffusers.models.unets.unet_2d_condition"),
    ("diffusers.models.vae", "diffusers.models.autoencoders.vae"),
)
_INSTALLED = False


def _clear_streamdiffusion_acceleration_modules() -> None:
    prefix = "streamdiffusion.acceleration"
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def ensure_diffusers_streamdiffusion_compat(*, force: bool = False) -> None:
    """Install legacy diffusers.models.* aliases for StreamDiffusion TensorRT."""
    global _INSTALLED
    if _INSTALLED and not force:
        return

    for legacy, modern in _SHIMS:
        if legacy in sys.modules and not force:
            continue
        target = importlib.import_module(modern)
        sys.modules[legacy] = target
        # Also expose as a package-like module when diffusers uses sub-imports.
        if not hasattr(target, "__path__"):
            pkg = types.ModuleType(legacy)
            pkg.__dict__.update(target.__dict__)
            pkg.__spec__ = getattr(target, "__spec__", None)
            sys.modules[legacy] = pkg

    _clear_streamdiffusion_acceleration_modules()
    _INSTALLED = True


def main() -> None:
    ensure_diffusers_streamdiffusion_compat(force=True)
    import streamdiffusion.acceleration.tensorrt.utilities as utilities

    print(f"diffusers compat OK; tensorrt.utilities={utilities}")


if __name__ == "__main__":
    main()
