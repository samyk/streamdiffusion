from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .frames import LatestFrameQueue, VideoFrame
from .image_utils import resize_rgb


@dataclass
class WebFrameHub:
    """Shared JPEG/RGB frame exchange between browser WebSockets and bridge threads."""

    incoming: LatestFrameQueue = field(default_factory=LatestFrameQueue)
    outgoing: LatestFrameQueue = field(default_factory=LatestFrameQueue)
    latest_output_jpeg: bytes | None = None
    jpeg_quality: int = 72
    _last_input_at: float = 0.0

    def ingest_jpeg(self, payload: bytes, width: int, height: int) -> None:
        if not payload:
            return
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except Exception:  # noqa: BLE001
            return
        rgb = np.ascontiguousarray(np.array(image, dtype=np.uint8))
        rgb = resize_rgb(rgb, width, height)
        self._last_input_at = time.monotonic()
        self.incoming.put(VideoFrame.now(rgb))

    def publish_rgb(self, frame: VideoFrame) -> None:
        self.outgoing.put(frame)
        rgb = np.ascontiguousarray(frame.data[:, :, :3])
        buffer = io.BytesIO()
        Image.fromarray(rgb, "RGB").save(
            buffer,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        self.latest_output_jpeg = buffer.getvalue()


@dataclass
class WebVideoInput:
    hub: WebFrameHub
    width: int
    height: int
    sequence: int = 0
    _last_frame_at: float = 0.0

    def set_resolution(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def read(self, timeout_ms: int = 1000) -> VideoFrame | None:
        frame = self.hub.incoming.get(timeout=timeout_ms / 1000.0)
        if frame is None:
            return None
        self.sequence += 1
        self._last_frame_at = time.monotonic()
        return VideoFrame.now(frame.data, self.sequence)

    def close(self) -> None:
        return None


@dataclass
class WebVideoOutput:
    hub: WebFrameHub
    name: str = "web"

    def write(self, frame: VideoFrame) -> None:
        self.hub.publish_rgb(frame)

    def close(self) -> None:
        return None
