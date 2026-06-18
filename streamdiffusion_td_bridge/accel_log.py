from __future__ import annotations


def normalize_accel_label(active: str | None, requested: str | None = None) -> str:
    """Map runtime backend to one of: tensorrt, xformers, none."""
    active_l = str(active or "").lower()
    requested_l = str(requested or "").lower()
    if "tensorrt" in active_l or active_l == "trt":
        return "tensorrt"
    if "xformers" in active_l:
        return "xformers"
    # Pre-load / TD change: requested xformers before wrapper reports active backend.
    if requested_l == "xformers" and active_l in ("", "xformers", "none"):
        return "xformers"
    return "none"


def log_acceleration(active: str | None, *, requested: str | None = None, detail: str = "") -> str:
    """Print a single clear acceleration line to screen/log."""
    label = normalize_accel_label(active, requested)
    line = f"[sdtd] acceleration: {label}"
    if label == "none" and detail:
        line += f" ({detail})"
    print(line, flush=True)
    return label
