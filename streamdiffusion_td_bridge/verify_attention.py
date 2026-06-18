from __future__ import annotations

import json
import sys

from streamdiffusion_td_bridge.attention import detect_available_backends, resolve_attention_backend


def verify_attention_stack() -> dict:
    available = detect_available_backends()
    resolved = resolve_attention_backend("auto")
    return {
        "available": available,
        "auto_resolves_to": resolved,
        "recommended_blackwell_order": ["flash", "sage", "xformers", "sdpa"],
    }


def main() -> None:
    report = verify_attention_stack()
    print(json.dumps(report, indent=2))
    if report["auto_resolves_to"] == "sdpa" and not any(report["available"].values()):
        print(
            "No optional attention kernels found. Install with ./scripts/install_attention_deps.sh",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
