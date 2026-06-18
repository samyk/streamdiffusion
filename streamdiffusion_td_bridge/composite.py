from __future__ import annotations

import re

import numpy as np

_HEX_COLOR = re.compile(r"^#?([0-9a-fA-F]{6})$")


def parse_background_color(value: str | tuple[int, int, int] | list[int] | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float32)
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    text = str(value).strip()
    match = _HEX_COLOR.match(text)
    if match:
        hex_rgb = match.group(1)
        return np.array(
            [int(hex_rgb[0:2], 16), int(hex_rgb[2:4], 16), int(hex_rgb[4:6], 16)],
            dtype=np.float32,
        )
    parts = [part.strip() for part in text.split(",")]
    if len(parts) >= 3:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def apply_segmentation_composite(
    source_rgb: np.ndarray,
    styled_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    person_only: bool,
    cut_background: bool,
    background_color: np.ndarray,
) -> np.ndarray:
    """Composite styled output with optional person-only and background cutout."""
    if not person_only and not cut_background:
        return np.ascontiguousarray(styled_rgb[:, :, :3], dtype=np.uint8)

    source = source_rgb[:, :, :3].astype(np.float32)
    styled = styled_rgb[:, :, :3].astype(np.float32)
    alpha = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    alpha3 = alpha[..., None]

    if person_only and not cut_background:
        background = source
    else:
        background = background_color.reshape(1, 1, 3)

    blended = alpha3 * styled + (1.0 - alpha3) * background
    return np.ascontiguousarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))
