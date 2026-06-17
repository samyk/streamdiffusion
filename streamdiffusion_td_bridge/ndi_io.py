from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from .frames import VideoFrame


class VideoInput(Protocol):
    def read(self, timeout_ms: int = 1000) -> VideoFrame | None: ...
    def close(self) -> None: ...


class VideoOutput(Protocol):
    def write(self, frame: VideoFrame) -> None: ...
    def close(self) -> None: ...


def resize_rgb(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width and frame.shape[2] == 3:
        return np.ascontiguousarray(frame)
    image = Image.fromarray(frame[:, :, :3], "RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.array(image, dtype=np.uint8))


@dataclass
class MockVideoInput:
    width: int
    height: int
    fps: float = 30.0
    sequence: int = 0

    def set_resolution(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def read(self, timeout_ms: int = 1000) -> VideoFrame:
        del timeout_ms
        time.sleep(1.0 / self.fps)
        self.sequence += 1
        x = np.linspace(0, 255, self.width, dtype=np.uint8)
        y = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
        phase = int((math.sin(self.sequence / 20.0) + 1.0) * 127)
        frame = np.dstack(
            [
                np.broadcast_to(x, (self.height, self.width)),
                np.broadcast_to(y, (self.height, self.width)),
                np.full((self.height, self.width), phase, dtype=np.uint8),
            ]
        )
        return VideoFrame.now(np.ascontiguousarray(frame), self.sequence)

    def close(self) -> None:
        return None


@dataclass
class MockVideoOutput:
    name: str
    frames: int = 0

    def write(self, frame: VideoFrame) -> None:
        del frame
        self.frames += 1

    def close(self) -> None:
        return None


class NdiError(RuntimeError):
    pass


def _resolve_stream_label(ndi_name: str) -> str:
    """Extract the stream label from an NDI source name."""
    if "(" in ndi_name and ndi_name.endswith(")"):
        return ndi_name.rsplit("(", 1)[-1][:-1].strip()
    return ndi_name.strip()


def _source_matches(requested: str, ndi_name: str) -> bool:
    """Exact stream label match — avoids streamdiffusion_out matching _out_b."""
    return _resolve_stream_label(ndi_name) == requested


class NdiVideoInput:
    def __init__(self, source_name: str | None, width: int, height: int) -> None:
        self.source_name = source_name
        self.width = width
        self.height = height
        self.sequence = 0
        self._stale_after_s = float(
            __import__("os").environ.get("SDTD_NDI_STALE_SEC", "2.0")
        )
        self._last_frame_at = 0.0
        self._last_reconnect_at = 0.0
        self._connected_ndi_name: str | None = None
        self._source_rotate = 0
        self.ndi = _load_ndi()
        if not self.ndi.initialize():
            raise NdiError("NDI initialization failed")

        self.finder = self.ndi.find_create_v2()
        if self.finder is None:
            raise NdiError("NDI finder creation failed")

        source = self._wait_for_source()
        self._connect_source(source, initial=True)

    def _find_matching_sources(self) -> list:
        self.ndi.find_wait_for_sources(self.finder, 250)
        sources = self.ndi.find_get_current_sources(self.finder) or []
        if self.source_name is None:
            return list(sources)
        return [source for source in sources if _source_matches(self.source_name, source.ndi_name)]

    def _wait_for_source(self):
        deadline = time.monotonic() + 20
        last_seen: list[str] = []
        while time.monotonic() < deadline:
            matches = self._find_matching_sources()
            last_seen = [source.ndi_name for source in matches]
            if matches:
                return matches[0]
            if self.source_name is None:
                self.ndi.find_wait_for_sources(self.finder, 1000)
                sources = self.ndi.find_get_current_sources(self.finder)
                if sources:
                    return sources[0]
        raise NdiError(f"NDI source {self.source_name!r} not found. Seen: {last_seen}")

    def _connect_source(self, source, *, initial: bool = False) -> None:
        if initial:
            create = self.ndi.RecvCreateV3()
            create.source_to_connect_to = source
            create.color_format = self.ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            create.bandwidth = self.ndi.RECV_BANDWIDTH_HIGHEST
            self.receiver = self.ndi.recv_create_v3(create)
            if self.receiver is None:
                raise NdiError(f"Could not create NDI receiver for {source.ndi_name}")
        else:
            self.ndi.recv_connect(self.receiver, source)
        self._connected_ndi_name = source.ndi_name
        self._last_reconnect_at = time.monotonic()
        print(f"[ndi] input connected: {source.ndi_name}")

    def _maybe_reconnect(self) -> None:
        now = time.monotonic()
        if self._last_frame_at and now - self._last_frame_at < self._stale_after_s:
            return
        if now - self._last_reconnect_at < 1.0:
            return
        matches = self._find_matching_sources()
        if not matches:
            return
        start = self._source_rotate % len(matches)
        for offset in range(len(matches)):
            candidate = matches[(start + offset) % len(matches)]
            if candidate.ndi_name == self._connected_ndi_name:
                continue
            try:
                self._connect_source(candidate)
                self._source_rotate = (start + offset + 1) % len(matches)
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[ndi] reconnect failed for {candidate.ndi_name}: {exc}")
        # Same host still advertising but no frames — refresh the connection.
        try:
            self._connect_source(matches[0])
        except Exception as exc:  # noqa: BLE001
            print(f"[ndi] reconnect refresh failed for {matches[0].ndi_name}: {exc}")

    def set_resolution(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def read(self, timeout_ms: int = 1000) -> VideoFrame | None:
        frame_type, video, _audio, _metadata = self.ndi.recv_capture_v2(self.receiver, timeout_ms)
        if frame_type == self.ndi.FRAME_TYPE_NONE:
            self._maybe_reconnect()
            return None
        if frame_type != self.ndi.FRAME_TYPE_VIDEO:
            return None

        try:
            data = np.copy(video.data)
            if data.ndim != 3 or data.shape[2] < 3:
                raise NdiError(f"Unsupported NDI frame shape: {data.shape}")

            # NDI is requested as BGRX/BGRA; StreamDiffusion expects RGB.
            rgb = data[:, :, :3][:, :, ::-1]
            rgb = resize_rgb(rgb, self.width, self.height)
            self.sequence += 1
            self._last_frame_at = time.monotonic()
            return VideoFrame.now(rgb, self.sequence)
        finally:
            self.ndi.recv_free_video_v2(self.receiver, video)

    def close(self) -> None:
        self.ndi.recv_destroy(self.receiver)
        self.ndi.find_destroy(self.finder)
        self.ndi.destroy()


class NdiVideoOutput:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ndi = _load_ndi()
        if not self.ndi.initialize():
            raise NdiError("NDI initialization failed")

        create = self.ndi.SendCreate()
        create.ndi_name = name
        self.sender = self.ndi.send_create(create)
        if self.sender is None:
            raise NdiError(f"Could not create NDI sender {name!r}")

    def write(self, frame: VideoFrame) -> None:
        rgb = np.ascontiguousarray(frame.data[:, :, :3])
        bgrx = np.empty((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
        bgrx[:, :, :3] = rgb[:, :, ::-1]
        bgrx[:, :, 3] = 255

        video = self.ndi.VideoFrameV2()
        video.data = bgrx
        video.FourCC = self.ndi.FOURCC_VIDEO_TYPE_BGRX
        self.ndi.send_send_video_v2(self.sender, video)

    def close(self) -> None:
        self.ndi.send_destroy(self.sender)
        self.ndi.destroy()


def make_video_io(
    backend: str,
    input_name: str | None,
    output_name: str,
    width: int,
    height: int,
) -> tuple[VideoInput, VideoOutput]:
    if backend == "mock":
        return MockVideoInput(width, height), MockVideoOutput(output_name)
    if backend == "ndi":
        return NdiVideoInput(input_name, width, height), NdiVideoOutput(output_name)
    raise ValueError(f"Unsupported video backend: {backend}")


def _load_ndi():
    try:
        import NDIlib as ndi
    except ImportError as exc:
        raise NdiError(
            "NDI backend requires ndi-python. Install with `pip install .[ndi]` and make "
            "sure the NDI runtime/SDK and Avahi are available on Linux."
        ) from exc
    return ndi

