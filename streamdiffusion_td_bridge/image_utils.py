from __future__ import annotations

import numpy as np
from PIL import Image


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width and frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    image = Image.fromarray(frame[:, :, :3], "RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.array(image, dtype=np.uint8))
