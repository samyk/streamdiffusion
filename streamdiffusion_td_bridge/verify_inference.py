from __future__ import annotations

import importlib
import json
import sys

from .deps import INFERENCE_MODULES, SETUP_HINT, missing_inference_modules


def main() -> None:
    report = {"modules": {}, "missing": missing_inference_modules()}
    for module_name in INFERENCE_MODULES:
        try:
            module = importlib.import_module(module_name)
            report["modules"][module_name] = getattr(module, "__version__", "ok")
        except ImportError:
            report["modules"][module_name] = None

    print(json.dumps(report, indent=2))

    if report["missing"]:
        print(SETUP_HINT, file=sys.stderr)
        sys.exit(1)

    try:
        from .deps import load_wrapper_class

        wrapper_cls = load_wrapper_class()
        report["wrapper"] = wrapper_cls.__module__
        print(json.dumps({"wrapper": report["wrapper"]}, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"Wrapper import failed: {exc}", file=sys.stderr)
        sys.exit(2)

    print("Inference stack looks usable.")


if __name__ == "__main__":
    main()
