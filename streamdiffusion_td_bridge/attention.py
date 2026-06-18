from __future__ import annotations

import importlib
from typing import Any, Literal

AttentionBackend = Literal["auto", "flash", "sage", "xformers", "sdpa", "none"]
AttentionKind = Literal["unet", "transformer"]

_AUTO_ORDER: tuple[AttentionBackend, ...] = ("flash", "sage", "xformers", "sdpa")


def normalize_attention_backend(value: str | None) -> AttentionBackend:
    if value is None:
        return "auto"
    normalized = str(value).strip().lower().replace("_", "-")
    aliases = {
        "flash-attn": "flash",
        "flash-attention": "flash",
        "flashattention": "flash",
        "flash2": "flash",
        "flash3": "flash",
        "sage-attn": "sage",
        "sageattention": "sage",
        "xformer": "xformers",
        "pytorch": "sdpa",
        "pytorch-sdpa": "sdpa",
        "scaled-dot-product": "sdpa",
        "eager": "none",
        "off": "none",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in ("auto", "flash", "sage", "xformers", "sdpa", "none"):
        return normalized  # type: ignore[return-value]
    return "auto"


def _flash_attn_available() -> bool:
    try:
        importlib.import_module("flash_attn")
        return True
    except Exception:  # noqa: BLE001
        return False


def _sage_attn_available() -> bool:
    try:
        importlib.import_module("sageattention")
        return True
    except Exception:  # noqa: BLE001
        return False


def _xformers_available() -> bool:
    try:
        import xformers  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def detect_available_backends() -> dict[str, bool]:
    return {
        "flash": _flash_attn_available(),
        "sage": _sage_attn_available(),
        "xformers": _xformers_available(),
        "sdpa": True,
    }


def resolve_attention_backend(
    requested: str | AttentionBackend | None,
    *,
    for_tensorrt: bool = False,
) -> AttentionBackend:
    if for_tensorrt:
        return "none"
    backend = normalize_attention_backend(requested)
    if backend != "auto":
        return backend
    available = detect_available_backends()
    for candidate in _AUTO_ORDER:
        if available.get(candidate, False):
            return candidate
    return "sdpa"


def _set_attn_processor(module: Any, processor: Any) -> None:
    if hasattr(module, "set_attn_processor"):
        module.set_attn_processor(processor)
        return
    if hasattr(module, "attn_processors"):
        processors = {name: processor for name in module.attn_processors}
        module.set_attn_processor(processors)


def _apply_sdpa(module: Any) -> None:
    from diffusers.models.attention_processor import AttnProcessor2_0

    _set_attn_processor(module, AttnProcessor2_0())


def _apply_flash(module: Any, *, kind: AttentionKind) -> str:
    if hasattr(module, "set_attention_backend"):
        for name in ("flash", "_flash_3", "_flash_2"):
            try:
                module.set_attention_backend(name)
                return f"flash:{name.lstrip('_')}"
            except Exception:  # noqa: BLE001
                continue

    if _flash_attn_available():
        try:
            from diffusers.models.attention_processor import (
                FusedAttnProcessor2_0,
            )

            _set_attn_processor(module, FusedAttnProcessor2_0())
            return "flash:fused"
        except Exception:  # noqa: BLE001
            pass

        if kind == "transformer":
            try:
                from diffusers.models.attention_processor import (
                    FluxAttnProcessor2_0,
                )

                _set_attn_processor(module, FluxAttnProcessor2_0())
                return "flash:flux-processor"
            except Exception:  # noqa: BLE001
                pass

    _apply_sdpa(module)
    return "sdpa"


def _apply_sage(module: Any) -> str:
    if not _sage_attn_available():
        raise RuntimeError("sageattention is not installed")

    try:
        from sageattention import sageattn  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("sageattention import failed") from exc

    if hasattr(module, "set_attention_backend"):
        for name in ("sage", "_sage"):
            try:
                module.set_attention_backend(name)
                return "sage"
            except Exception:  # noqa: BLE001
                continue

    # Fallback: keep module on SDPA but mark sage requested; full graph patching is model-specific.
    _apply_sdpa(module)
    return "sage:sdpa-fallback"


def _apply_xformers_module(module: Any) -> bool:
    """Apply xFormers to one diffusers module (Flux2 backend API or UNet processor)."""
    if hasattr(module, "set_attention_backend"):
        for name in ("xformers", "_xformers"):
            try:
                module.set_attention_backend(name)
                return True
            except Exception:  # noqa: BLE001
                continue

    from diffusers.models.attention_processor import XFormersAttnProcessor

    processor = XFormersAttnProcessor()
    if hasattr(module, "set_attn_processor"):
        module.set_attn_processor(processor)
        return True
    if hasattr(module, "unet") and module.unet is not None:
        module.unet.set_attn_processor(processor)
        return True
    return False


def _apply_xformers(modules: list[Any]) -> None:
    """Apply xFormers without diffusers' fp32 self-test (breaks on Blackwell sm_120)."""
    if not any(_apply_xformers_module(module) for module in modules):
        raise RuntimeError("no module accepted xformers backend")


def _target_modules(pipe: Any, kind: AttentionKind) -> list[Any]:
    modules: list[Any] = []
    if kind == "unet" and hasattr(pipe, "unet") and pipe.unet is not None:
        modules.append(pipe.unet)
        return modules

    for attr in ("transformer", "transformer_2", "text_encoder", "text_encoder_2"):
        mod = getattr(pipe, attr, None)
        if mod is not None:
            modules.append(mod)
    return modules


def apply_attention(
    pipe: Any,
    backend: str | AttentionBackend | None,
    *,
    kind: AttentionKind = "transformer",
    for_tensorrt: bool = False,
) -> str:
    """Apply an attention backend to a diffusers pipeline module. Returns active backend label."""
    resolved = resolve_attention_backend(backend, for_tensorrt=for_tensorrt)
    if resolved == "none":
        return "none"

    modules = _target_modules(pipe, kind)
    if not modules:
        return "none"

    if resolved == "xformers":
        if not _xformers_available():
            raise RuntimeError("xformers is not installed")
        _apply_xformers(modules)
        return "xformers"

    active = resolved
    for module in modules:
        if resolved == "sdpa":
            _apply_sdpa(module)
            active = "sdpa"
        elif resolved == "flash":
            active = _apply_flash(module, kind=kind)
        elif resolved == "sage":
            active = _apply_sage(module)
        else:
            _apply_sdpa(module)
            active = "sdpa"

    print(f"[attention] {kind} backend active: {active}")
    return active
