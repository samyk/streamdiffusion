from __future__ import annotations

import importlib

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
  python3 -m streamdiffusion_td_bridge.verify_inference
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


def load_wrapper_class():
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
