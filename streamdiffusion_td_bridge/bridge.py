from __future__ import annotations

import asyncio
import queue
import signal
import threading
import time
from pathlib import Path
from typing import Any

from .config import PRESETS, BridgeConfig, RuntimeState
from .control_server import ControlServer
from .daydream_api import DaydreamApiServer
from .frames import LatestFrameQueue, SharedState
from .ndi_io import make_video_io
from .scene_idle import SceneIdleGate
from .web_io import WebFrameHub
from .web_server import MobileWebServer, load_prompts
from .stream_worker import StreamWorker


class BridgeApp:
    def __init__(self, config: BridgeConfig) -> None:
        if config.preset not in PRESETS:
            raise ValueError(f"Unknown preset {config.preset!r}. Known presets: {sorted(PRESETS)}")
        self.config = config
        self.input_queue = LatestFrameQueue()
        self.output_queue = LatestFrameQueue()
        self.command_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.state = SharedState(
            RuntimeState(preset=config.preset, width=config.width, height=config.height)
        )
        self.scene_idle_gate = SceneIdleGate(
            mode=config.scene_idle_mode,
            threshold=config.scene_change_threshold,
        )
        self.frame_hub = (
            WebFrameHub(jpeg_quality=config.web_jpeg_quality)
            if config.video_backend == "web"
            else None
        )
        self.stop_event = threading.Event()
        self.worker = StreamWorker(
            config=config,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
            command_queue=self.command_queue,
            state=self.state,
            scene_idle_gate=self.scene_idle_gate,
        )
        self.video_input = None
        self.video_output = None
        self.threads: list[threading.Thread] = []

    async def run(self) -> None:
        self.config.print_startup_settings()
        self.video_input, self.video_output = make_video_io(
            self.config.video_backend,
            self.config.input_name,
            self.config.output_name,
            self.config.width,
            self.config.height,
            frame_hub=self.frame_hub,
        )
        self._start_thread("video-input", self._video_input_loop)
        self._start_thread("stream-worker", self.worker.run)
        self._start_thread("video-output", self._video_output_loop)

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop)
            except NotImplementedError:
                pass

        server = ControlServer(
            self.config.control_host,
            self.config.control_port,
            self.command_queue,
            self.state,
        )
        daydream = DaydreamApiServer(
            self.config.daydream_host,
            self.config.daydream_port,
            self.command_queue,
            self.state,
            stream_id=self.config.stream_id,
            input_name=self.config.input_name or "td_streamdiffusion_in",
            output_name=self.config.output_name,
        )
        daydream.start()
        web_server: MobileWebServer | None = None
        if self.config.web_port:
            prompts_path = self.config.prompts_file
            if prompts_path is None:
                repo_prompts = Path.cwd() / ".prompts.txt"
                prompts_path = str(repo_prompts) if repo_prompts.is_file() else None
            assert self.frame_hub is not None
            web_server = MobileWebServer(
                self.config.web_host,
                self.config.web_port,
                frame_hub=self.frame_hub,
                command_queue=self.command_queue,
                state=self.state,
                stream_id=self.config.stream_id,
                infer_width=self.config.width,
                infer_height=self.config.height,
                prompts=load_prompts(prompts_path),
                prompt_state_path=Path.cwd() / ".state" / "web_prompt.txt",
                public_host=self.config.web_public_host,
                public_port=self.config.web_public_port,
                tls_cert=self.config.web_tls_cert,
                tls_key=self.config.web_tls_key,
            )
            await web_server.start()
        server_task = asyncio.create_task(server.run())
        try:
            await self._wait_for_stop()
        finally:
            server_task.cancel()
            if web_server is not None:
                await web_server.stop()
            daydream.stop()
        self._close()

    def stop(self) -> None:
        self.stop_event.set()
        self.worker.stop()

    def _start_thread(self, name: str, target) -> None:  # noqa: ANN001
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self.threads.append(thread)

    def _video_input_loop(self) -> None:
        assert self.video_input is not None
        last_time = time.perf_counter()
        last_resolution = (self.config.width, self.config.height)
        while not self.stop_event.is_set():
            resolution = (self.config.width, self.config.height)
            if resolution != last_resolution and hasattr(self.video_input, "set_resolution"):
                self.video_input.set_resolution(*resolution)
                last_resolution = resolution
            frame = self.video_input.read(timeout_ms=1000)
            if frame is None:
                if time.monotonic() - getattr(self.video_input, "_last_frame_at", 0.0) > 2.0:
                    self.state.update(fps_in=0.0)
                continue
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                self.state.update(fps_in=1.0 / dt)
            if self.scene_idle_gate.should_accept_frame(frame.data):
                self.input_queue.put(frame)
            else:
                self.state.update(scene_idle=True, fps_out=0.0)

    def _video_output_loop(self) -> None:
        assert self.video_output is not None
        while not self.stop_event.is_set():
            frame = self.output_queue.get(timeout=0.1)
            if frame is None:
                continue
            self.video_output.write(frame)

    async def _wait_for_stop(self) -> None:
        while not self.stop_event.is_set():
            await asyncio.sleep(0.2)

    def _close(self) -> None:
        self.worker.stop()
        for io in (self.video_input, self.video_output):
            if io is None:
                continue
            try:
                io.close()
            except Exception:  # noqa: BLE001
                pass

