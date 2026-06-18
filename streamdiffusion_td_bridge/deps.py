from __future__ import annotations

import importlib

from .config import ModelPreset, PRESETS, is_dit_preset, is_flux_preset, is_transformer_preset

INFERENCE_MODULES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "streamdiffusion",
)

SETUP_HINT = """\
Inference dependencies are missing. From the project root with the venv active, run:

  ./scripts/fix_inference_deps.sh

Or manually:

  source .venv/bin/activate
  python3 -m pip install diffusers==0.24.0 transformers accelerate fire omegaconf
  ./scripts/install_streamdiffusion_deps.sh
  ./scripts/install_tensorrt_deps.sh
  python3 -m streamdiffusion_td_bridge.verify_tensorrt
"""

FLUX_SETUP_HINT = """\
FLUX.2 Klein dependencies are missing. From the project root with the venv active, run:

  ./scripts/install_flux2_klein_deps.sh

This upgrades diffusers/transformers for Flux2KleinPipeline. To return to the SD stack:

  ./scripts/fix_inference_deps.sh
"""


def missing_inference_modules() -> list[str]:
    missing: list[str] = []
    for module_name in INFERENCE_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    return missing


def ensure_inference_deps() -> None:
    missing = missing_inference_modules()
    if not missing:
        return
    raise RuntimeError(
        "Missing inference packages: "
        + ", ".join(missing)
        + ".\n\n"
        + SETUP_HINT
    )


def ensure_flux_deps() -> None:
    ensure_inference_modules = missing_inference_modules()
    if "torch" in ensure_inference_modules:
        raise RuntimeError("PyTorch is required for FLUX.2 Klein.\n\n" + FLUX_SETUP_HINT)
    try:
        importlib.import_module("diffusers")
        from diffusers import Flux2KleinPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"FLUX.2 Klein pipeline unavailable ({type(exc).__name__}: {exc}).\n\n" + FLUX_SETUP_HINT
        ) from exc


def _resolve_preset(preset: str | ModelPreset | None) -> ModelPreset | None:
    if preset is None:
        return None
    if isinstance(preset, ModelPreset):
        return preset
    return PRESETS.get(preset)


DIT_SETUP_HINT = """\
DiT / SD3.5 dependencies are missing. From the project root with the venv active, run:

  ./scripts/install_dit_deps.sh
  ./scripts/install_attention_deps.sh

This upgrades diffusers for StableDiffusion3 pipelines. To return to the SD stack:

  ./scripts/fix_inference_deps.sh
"""


def ensure_dit_deps() -> None:
    missing = missing_inference_modules()
    if "torch" in missing:
        raise RuntimeError("PyTorch is required for DiT / SD3.5.\n\n" + DIT_SETUP_HINT)
    try:
        from diffusers import StableDiffusion3Img2ImgPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"SD3.5 pipeline unavailable ({type(exc).__name__}: {exc}).\n\n" + DIT_SETUP_HINT
        ) from exc


def load_wrapper_class(preset: str | ModelPreset | None = None):
    active = _resolve_preset(preset)
    if active is not None and is_flux_preset(pipeline=active.pipeline, name=active.name):
        ensure_flux_deps()
        from streamdiffusion_td_bridge.vendor.flux_klein_wrapper import FluxKleinWrapper

        return FluxKleinWrapper

    if active is not None and is_dit_preset(pipeline=active.pipeline, name=active.name):
        ensure_dit_deps()
        from streamdiffusion_td_bridge.vendor.dit_wrapper import DitWrapper

        return DitWrapper

    ensure_inference_deps()

    errors: list[str] = []
    for import_path in (
        "streamdiffusion_td_bridge.vendor.wrapper",
        "utils.wrapper",
    ):
        try:
            module = importlib.import_module(import_path)
            return module.StreamDiffusionWrapper
        except Exception as exc:  # noqa: BLE001 - surfaced in error message
            errors.append(f"{import_path}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Could not import StreamDiffusionWrapper after inference dependencies were present. "
        "Tried: " + "; ".join(errors)
    )
