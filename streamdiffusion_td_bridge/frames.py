from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class VideoFrame:
    data: np.ndarray
    timestamp: float
    sequence: int = 0

    @classmethod
    def now(cls, data: np.ndarray, sequence: int = 0) -> "VideoFrame":
        return cls(data=data, timestamp=time.perf_counter(), sequence=sequence)


class LatestFrameQueue:
    """One-slot queue: keeps latency bounded by dropping stale frames."""

    def __init__(self) -> None:
        self._queue: queue.Queue[VideoFrame] = queue.Queue(maxsize=1)

    def put(self, frame: VideoFrame) -> None:
        try:
            self._queue.put_nowait(frame)
            return
        except queue.Full:
            pass

        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(frame)

    def get(self, timeout: float = 0.1) -> VideoFrame | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class SharedState:
    def __init__(self, value) -> None:  # noqa: ANN001 - generic dataclass holder
        self._value = value
        self._lock = threading.Lock()

    def update(self, **values) -> None:  # noqa: ANN003 - mirrors dataclass fields
        with self._lock:
            for key, value in values.items():
                setattr(self._value, key, value)

    def mutate(self, fn):  # noqa: ANN001, ANN201 - callback helper
        with self._lock:
            return fn(self._value)

    def snapshot(self) -> dict:
        with self._lock:
            return self._value.snapshot()

