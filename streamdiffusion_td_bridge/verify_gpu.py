from __future__ import annotations

import json
import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        print("torch is not installed")
        sys.exit(1)

    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "devices": [],
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            report["devices"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
            )

    print(json.dumps(report, indent=2))

    if not report["cuda_available"]:
        print("CUDA is not available to PyTorch", file=sys.stderr)
        sys.exit(2)

    if "+cu132" not in torch.__version__:
        print(
            f"Expected torch+cu132 nightly; got {torch.__version__!r}. "
            "Run ./scripts/install_pytorch_cu132.sh",
            file=sys.stderr,
        )
        sys.exit(3)

    cuda_runtime = report["cuda_runtime"] or ""
    if not cuda_runtime.startswith("13.2"):
        print(
            f"Expected CUDA 13.2 runtime from cu132 wheels; got {cuda_runtime!r}. "
            "Run ./scripts/fix_inference_deps.sh",
            file=sys.stderr,
        )
        sys.exit(3)

    arch_list = set(report["arch_list"])
    capabilities = {tuple(device["capability"]) for device in report["devices"]}
    if (12, 0) in capabilities and "sm_120" not in arch_list:
        print(
            "Detected Blackwell capability (12, 0), but this PyTorch build lacks sm_120. "
            "Install cu132 nightly wheels: ./scripts/install_pytorch_cu132.sh",
            file=sys.stderr,
        )
        sys.exit(4)

    try:
        from torchvision.transforms import InterpolationMode  # noqa: F401
    except RuntimeError as exc:
        print(f"torchvision import failed (torch/torchvision skew): {exc}", file=sys.stderr)
        print("Run ./scripts/fix_inference_deps.sh", file=sys.stderr)
        sys.exit(5)

    print("GPU stack looks usable.")


if __name__ == "__main__":
    main()
