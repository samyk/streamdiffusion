from __future__ import annotations

import asyncio
import json
import ssl
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

from .frames import SharedState
from .sdtd_mapper import daydream_params_to_commands, normalize_stream_params
from .web_io import WebFrameHub

_STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


def load_prompts(path: str | None) -> list[str]:
    if not path:
        return []
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        return []
    prompts: list[str] = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        text = line.strip().lstrip("\ufeff")
        if text and not text.startswith("#"):
            prompts.append(text)
    return prompts


class MobileWebServer:
    def __init__(
        self,
        host: str,
        port: int,
        *,
        frame_hub: WebFrameHub,
        command_queue,
        state: SharedState,
        stream_id: str,
        infer_width: int,
        infer_height: int,
        prompts: list[str],
        prompt_state_path: Path | None = None,
        public_host: str | None = None,
        public_port: int | None = None,
        tls_cert: str | None = None,
        tls_key: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_hub = frame_hub
        self.command_queue = command_queue
        self.state = state
        self.stream_id = stream_id
        self.infer_width = infer_width
        self.infer_height = infer_height
        self.prompts = prompts
        self.prompt_state_path = prompt_state_path
        self.public_host = public_host
        self.public_port = public_port
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self._params: dict[str, Any] = {}
        self._clients: set[web.WebSocketResponse] = set()
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._broadcast_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/", self._handle_index)
        app.router.add_get("/app.js", self._handle_app_js)
        app.router.add_get("/style.css", self._handle_style_css)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/api/config", self._handle_config)
        app.router.add_get("/api/prompts", self._handle_prompts)
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get(f"/v1/streams/{self.stream_id}", self._handle_stream_get)
        app.router.add_patch(f"/v1/streams/{self.stream_id}", self._handle_stream_patch)
        app.router.add_options("/{tail:.*}", self._handle_options)
        app.router.add_get("/ws", self._handle_ws)

        ssl_context = None
        if self.tls_cert and self.tls_key:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.tls_cert, self.tls_key)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port, ssl_context=ssl_context)
        await self._site.start()
        scheme = "https" if ssl_context else "http"
        print(f"[web] mobile UI on {scheme}://{self.host}:{self.port}/", flush=True)
        self._broadcast_task = asyncio.create_task(self._broadcast_output_loop())

    async def stop(self) -> None:
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
            self._broadcast_task = None
        for client in list(self._clients):
            await client.close()
        self._clients.clear()
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def _cors(self, response: web.StreamResponse) -> None:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, PATCH, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

    async def _json(self, payload: dict[str, Any], *, status: int = 200) -> web.Response:
        response = web.json_response(payload, status=status)
        self._cors(response)
        return response

    async def _handle_options(self, _request: web.Request) -> web.Response:
        response = web.Response(status=204)
        self._cors(response)
        return response

    async def _handle_index(self, _request: web.Request) -> web.Response:
        return self._static_file(_STATIC_DIR / "index.html")

    async def _handle_app_js(self, _request: web.Request) -> web.Response:
        return self._static_file(_STATIC_DIR / "app.js")

    async def _handle_style_css(self, _request: web.Request) -> web.Response:
        return self._static_file(_STATIC_DIR / "style.css")

    def _static_file(self, path: Path) -> web.FileResponse:
        response = web.FileResponse(path)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    async def _handle_health(self, _request: web.Request) -> web.Response:
        return await self._json({"ok": True})

    async def _handle_config(self, _request: web.Request) -> web.Response:
        snap = self.state.snapshot()
        scheme = "https" if self.tls_cert else "http"
        host = self.public_host or _request.host.split(":", 1)[0]
        port = self.public_port if self.public_port is not None else self.port
        default_port = 443 if scheme == "https" else 80
        if port == default_port:
            public_url = f"{scheme}://{host}/"
        else:
            public_url = f"{scheme}://{host}:{port}/"
        return await self._json(
            {
                "stream_id": self.stream_id,
                "infer_width": self.infer_width,
                "infer_height": self.infer_height,
                "output_width": snap.get("width", self.infer_width),
                "output_height": snap.get("height", self.infer_height),
                "prompt": snap.get("prompt"),
                "public_host": host,
                "public_port": port,
                "public_url": public_url,
            }
        )

    async def _handle_prompts(self, _request: web.Request) -> web.Response:
        return await self._json({"prompts": self.prompts})

    async def _handle_status(self, _request: web.Request) -> web.Response:
        return await self._json(self.state.snapshot())

    async def _handle_stream_get(self, _request: web.Request) -> web.Response:
        snap = self.state.snapshot()
        return await self._json(
            {
                "id": self.stream_id,
                "pipeline": "streamdiffusion",
                "status": snap.get("status", "running"),
                "params": self._params,
                "runtime": snap,
            }
        )

    async def _handle_stream_patch(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        params = payload.get("params", payload)
        if isinstance(params, dict):
            params = self._normalize_web_params(params)
            self._apply_params(params)
        return await self._json(
            {
                "success": True,
                "id": self.stream_id,
                "pipeline": "streamdiffusion",
                "params": self._params,
            }
        )

    def _apply_params(self, params: dict[str, Any]) -> None:
        if not params:
            return
        self._persist_custom_prompt(params)
        previous = dict(self._params)
        params = normalize_stream_params(params, previous)
        self._params.update(params)
        for command in daydream_params_to_commands(params, previous=previous):
            self.command_queue.put_nowait(command)

    def _persist_custom_prompt(self, params: dict[str, Any]) -> None:
        if self.prompt_state_path is None:
            return
        if params.get("clear_custom_prompt"):
            try:
                self.prompt_state_path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        if not params.get("custom_prompt"):
            return
        prompt = str(params.get("prompt", "")).strip()
        if not prompt:
            return
        try:
            self.prompt_state_path.parent.mkdir(parents=True, exist_ok=True)
            self.prompt_state_path.write_text(prompt + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"[web] could not persist custom prompt: {exc}", flush=True)

    def _normalize_web_params(self, params: dict[str, Any]) -> dict[str, Any]:
        params = dict(params)
        if "strength" in params:
            strength = max(0.0, min(1.0, float(params["strength"])))
            params.pop("strength", None)
            params["t_index_list"] = [round(strength * 60)]
        if "t_index_list" in params:
            params["t_index_list"] = [
                max(0, min(60, int(round(float(value)))))
                for value in (params.get("t_index_list") or [])
            ]
        return params

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        for client in list(self._clients):
            await client.close(code=1000, message=b"new client connected")
        self._clients.clear()

        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        self._clients.add(ws)
        latest = self.frame_hub.latest_output_jpeg
        if latest:
            await ws.send_bytes(latest)
        try:
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    self.frame_hub.ingest_jpeg(
                        message.data,
                        self.infer_width,
                        self.infer_height,
                    )
                elif message.type == WSMsgType.TEXT:
                    await self._handle_ws_text(message.data)
                elif message.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def _handle_ws_text(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        if payload.get("type") == "set_prompt":
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                self._apply_params({"prompt": prompt.strip()})

    async def _broadcast_output_loop(self) -> None:
        last_payload: bytes | None = None
        while True:
            await asyncio.sleep(0.016)
            payload = self.frame_hub.latest_output_jpeg
            if not payload or payload is last_payload:
                continue
            last_payload = payload
            dead: list[web.WebSocketResponse] = []
            for client in self._clients:
                try:
                    await client.send_bytes(payload)
                except ConnectionResetError:
                    dead.append(client)
            for client in dead:
                self._clients.discard(client)
