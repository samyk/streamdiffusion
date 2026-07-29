from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebBridgeSettings:
    public_host: str
    public_port: int
    bind_host: str
    bind_port: int
    tls_cert: str
    tls_key: str
    stream_id: str
    daydream_port: int
    control_port: int
    screen_session: str
    prompts_file: str
    preset: str
    width: int
    height: int
    jpeg_quality: int
    prompt: str
    upscale_enabled: bool
    upscale_factor: int
    upscale_method: str
    upscale_maxine_quality: str
    upscale_half: bool
    acceleration: str
    attention_backend: str
    flux_transformer_engine: bool
    frame_buffer_size: int | None

    @property
    def public_url(self) -> str:
        scheme = "https" if self.tls_cert else "http"
        port = self.public_port
        default_port = 443 if scheme == "https" else 80
        if port == default_port:
            return f"{scheme}://{self.public_host}/"
        return f"{scheme}://{self.public_host}:{port}/"


def _resolve_path(root: Path, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((root / path).resolve())


def load_web_config(path: Path, *, root: Path | None = None) -> WebBridgeSettings:
    repo_root = (root or path.parent.parent).resolve()
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    public = data.get("public", {})
    server = data.get("server", {})
    tls = data.get("tls", {})
    bridge = data.get("bridge", {})
    upscale = bridge.get("upscale", {})
    performance = bridge.get("performance", {})

    frame_buffer_size = performance.get("frame_buffer_size")
    if frame_buffer_size is not None:
        frame_buffer_size = int(frame_buffer_size)

    return WebBridgeSettings(
        public_host=str(public.get("host", "localhost")),
        public_port=int(public.get("port", 443)),
        bind_host=str(server.get("bind_host", "0.0.0.0")),
        bind_port=int(server.get("bind_port", 8790)),
        tls_cert=_resolve_path(repo_root, str(tls.get("cert", ""))),
        tls_key=_resolve_path(repo_root, str(tls.get("key", ""))),
        stream_id=str(bridge.get("stream_id", "phone-1")),
        daydream_port=int(bridge.get("daydream_port", 8782)),
        control_port=int(bridge.get("control_port", 8767)),
        screen_session=str(bridge.get("screen_session", "sdtd-bridge-web")),
        prompts_file=_resolve_path(repo_root, str(bridge.get("prompts_file", ".prompts.txt"))),
        preset=str(bridge.get("preset", "sd_turbo_fast")),
        width=int(bridge.get("width", 960)),
        height=int(bridge.get("height", 536)),
        jpeg_quality=int(bridge.get("jpeg_quality", 72)),
        prompt=str(bridge.get("prompt", "")),
        upscale_enabled=bool(upscale.get("enabled", True)),
        upscale_factor=int(upscale.get("factor", 2)),
        upscale_method=str(upscale.get("method", "maxine-vsr")),
        upscale_maxine_quality=str(upscale.get("maxine_quality", "high")),
        upscale_half=bool(upscale.get("half", True)),
        acceleration=str(performance.get("acceleration", "tensorrt")),
        attention_backend=str(performance.get("attention_backend", "auto")),
        flux_transformer_engine=bool(performance.get("flux_transformer_engine", True)),
        frame_buffer_size=frame_buffer_size,
    )
