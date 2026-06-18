from __future__ import annotations

import importlib
import json
import sys

from streamdiffusion_td_bridge.cuda_compat import ensure_cuda_cudart_importable
from streamdiffusion_td_bridge.diffusers_compat import ensure_diffusers_streamdiffusion_compat


def verify_tensorrt_imports() -> dict:
    ensure_cuda_cudart_importable()
    ensure_diffusers_streamdiffusion_compat()
    report: dict[str, str | None] = {}
    errors: list[str] = []

    for name in ("tensorrt", "polygraphy", "onnx_graphsurgeon", "onnxscript"):
        try:
            mod = importlib.import_module(name)
            report[name] = getattr(mod, "__version__", "ok")
        except Exception as exc:  # noqa: BLE001
            report[name] = None
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    try:
        from polygraphy import cuda  # noqa: F401

        report["polygraphy.cuda"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["polygraphy.cuda"] = None
        errors.append(f"polygraphy.cuda: {type(exc).__name__}: {exc}")

    try:
        from cuda import cudart  # noqa: F401

        report["cuda.cudart"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["cuda.cudart"] = None
        errors.append(f"cuda.cudart: {type(exc).__name__}: {exc}")

    for symbol in (
        "streamdiffusion.acceleration.tensorrt",
        "streamdiffusion.acceleration.tensorrt.engine",
        "streamdiffusion.acceleration.tensorrt.models",
    ):
        try:
            importlib.import_module(symbol)
            report[symbol] = "ok"
        except Exception as exc:  # noqa: BLE001
            report[symbol] = None
            errors.append(f"{symbol}: {type(exc).__name__}: {exc}")

    report["errors"] = errors
    return report


def main() -> None:
    report = verify_tensorrt_imports()
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2))

    errors = report.get("errors") or []
    if errors:
        print("TensorRT import failures:", file=sys.stderr)
        for line in errors:
            print(f"  - {line}", file=sys.stderr)
        print("Run: ./scripts/install_tensorrt_deps.sh", file=sys.stderr)
        sys.exit(1)

    print("TensorRT stack looks usable.")


if __name__ == "__main__":
    main()
