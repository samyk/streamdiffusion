from __future__ import annotations

import threading
from typing import Literal

import numpy as np

SceneIdleMode = Literal["off", "respect_controls", "strict"]

SCENE_IDLE_MODES: tuple[SceneIdleMode, ...] = ("off", "respect_controls", "strict")

_DOWNSAMPLE = (64, 64)


def normalize_scene_idle_mode(value: str | None) -> SceneIdleMode:
    raw = str(value or "off").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "off": "off",
        "respect_controls": "respect_controls",
        "respectcontrols": "respect_controls",
        "prompt_wake": "respect_controls",
        "promptwake": "respect_controls",
        "strict": "strict",
        "ignore_controls": "strict",
    }
    return aliases.get(raw, "off")  # type: ignore[return-value]


def downsample_rgb(rgb: np.ndarray) -> np.ndarray:
    data = np.ascontiguousarray(rgb[:, :, :3], dtype=np.float32)
    h, w = data.shape[:2]
    th, tw = _DOWNSAMPLE
    if h != th or w != tw:
        ys = (np.linspace(0, h - 1, th)).astype(np.int32)
        xs = (np.linspace(0, w - 1, tw)).astype(np.int32)
        data = data[ys][:, xs]
    return _box_blur(data)


def _box_blur(data: np.ndarray) -> np.ndarray:
    out = np.empty_like(data)
    for c in range(data.shape[2]):
        plane = data[:, :, c]
        padded = np.pad(plane, 1, mode="edge")
        out[:, :, c] = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out


def compute_frame_change(current: np.ndarray, reference: np.ndarray) -> float:
    """Mean absolute channel diff on a blurred 64x64 downsample, normalized 0-1."""
    cur = downsample_rgb(current)
    ref = downsample_rgb(reference) if reference.shape != cur.shape else reference
    return float(np.mean(np.abs(cur - ref)) / 255.0)


class SceneIdleGate:
    """Skip frame processing when the input scene is static."""

    def __init__(
        self,
        mode: SceneIdleMode | str = "off",
        threshold: float = 0.015,
        *,
        wake_frames: int = 2,
    ) -> None:
        self._lock = threading.Lock()
        self.mode: SceneIdleMode = normalize_scene_idle_mode(mode)
        self.threshold = max(0.0, min(1.0, float(threshold)))
        self.wake_frames = max(1, int(wake_frames))
        self._reference: np.ndarray | None = None
        self._force_once = False
        self._idle = False
        self._wake_streak = 0

    def configure(
        self,
        *,
        mode: SceneIdleMode | str | None = None,
        threshold: float | None = None,
        wake_frames: int | None = None,
    ) -> None:
        with self._lock:
            if mode is not None:
                self.mode = normalize_scene_idle_mode(mode)
            if threshold is not None:
                self.threshold = max(0.0, min(1.0, float(threshold)))
            if wake_frames is not None:
                self.wake_frames = max(1, int(wake_frames))
            if self.mode == "off":
                self._idle = False
                self._force_once = False
                self._wake_streak = 0

    def reset_reference(self) -> None:
        with self._lock:
            self._reference = None
            self._idle = False
            self._wake_streak = 0

    def request_force_process(self) -> None:
        with self._lock:
            self._force_once = True

    def should_accept_frame(self, frame_data: np.ndarray) -> bool:
        with self._lock:
            if self.mode == "off":
                self._idle = False
                self._wake_streak = 0
                return True

            rgb = np.ascontiguousarray(frame_data[:, :, :3])
            if self._reference is None:
                self._reference = downsample_rgb(rgb)
                self._idle = False
                self._wake_streak = 0
                return True

            change = compute_frame_change(rgb, self._reference)

            if self._force_once and self.mode == "respect_controls":
                self._force_once = False
                self._reference = downsample_rgb(rgb)
                self._idle = False
                self._wake_streak = 0
                return True

            if change >= self.threshold:
                self._wake_streak += 1
            else:
                self._wake_streak = 0

            if self._wake_streak >= self.wake_frames:
                self._reference = downsample_rgb(rgb)
                self._idle = False
                return True

            self._idle = True
            return False

    @property
    def idle(self) -> bool:
        with self._lock:
            return self._idle and self.mode != "off"
