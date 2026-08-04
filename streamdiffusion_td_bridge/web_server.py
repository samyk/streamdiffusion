from __future__ import annotations

import asyncio
import base64
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


def _dedupe_prompts(prompts: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for prompt in prompts:
        text = str(prompt).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


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
        self.prompt_history_path = (
            prompt_state_path.with_name("web_prompt_history.json")
            if prompt_state_path is not None
            else None
        )
        self.public_host = public_host
        self.public_port = public_port
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self._params: dict[str, Any] = {}
        self._prompt_history = self._load_prompt_history(prompts)
        self._clients: set[web.WebSocketResponse] = set()
        self._client_roles: dict[web.WebSocketResponse, str] = {}
        self._producer: web.WebSocketResponse | None = None
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
        self._broadcast_task.add_done_callback(self._log_broadcast_task_done)

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
        self._client_roles.clear()
        if self._site is not None:
            await self._site.stop()
            self._site = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    def _log_broadcast_task_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            print(f"[web] broadcast task stopped: {exc}", flush=True)

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
                "producer_active": self._producer is not None and not self._producer.closed,
                "viewer_count": self._viewer_count(),
                "prompt_history": self._prompt_history,
            }
        )

    async def _handle_prompts(self, _request: web.Request) -> web.Response:
        return await self._json(
            {"prompts": self.prompts, "history": self._prompt_history}
        )

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
            params = await self._apply_and_broadcast_params(params)
        return await self._json(
            {
                "success": True,
                "id": self.stream_id,
                "pipeline": "streamdiffusion",
                "params": self._params,
            }
        )

    async def _apply_and_broadcast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        applied = self._apply_params(params)
        if applied:
            event: dict[str, Any] = {"type": "params_update", "params": applied}
            if "prompt" in applied:
                event["prompt_history"] = self._prompt_history
            await self._broadcast_event(event)
        return applied

    def _apply_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not params:
            return {}
        params = self._normalize_web_params(params)
        self._persist_custom_prompt(params)
        previous = dict(self._params)
        params = normalize_stream_params(params, previous)
        self._params.update(params)
        if prompt := str(params.get("prompt", "")).strip():
            self._remember_prompt(prompt)
        for command in daydream_params_to_commands(params, previous=previous):
            self.command_queue.put_nowait(command)
        return params

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

    def _load_prompt_history(self, prompts: list[str]) -> list[str]:
        history: list[str] = []
        if self.prompt_history_path is not None and self.prompt_history_path.is_file():
            try:
                payload = json.loads(self.prompt_history_path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    history.extend(str(item) for item in payload)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[web] could not load prompt history: {exc}", flush=True)
        history.extend(prompts)
        return _dedupe_prompts(history)[:100]

    def _remember_prompt(self, prompt: str) -> None:
        self._prompt_history = _dedupe_prompts([prompt, *self._prompt_history])[:100]
        if self.prompt_history_path is None:
            return
        try:
            self.prompt_history_path.parent.mkdir(parents=True, exist_ok=True)
            self.prompt_history_path.write_text(
                json.dumps(self._prompt_history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"[web] could not persist prompt history: {exc}", flush=True)

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
        role = request.query.get("role", "viewer")
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=20)
        await ws.prepare(request)
        if role == "producer":
            if self._producer is not None and not self._producer.closed:
                await self._producer.close(code=1000, message=b"new producer connected")
            self._producer = ws
        self._clients.add(ws)
        self._client_roles[ws] = role
        await ws.send_str(
            json.dumps(
                {
                    "type": "producer_status",
                    "active": self._producer is not None and not self._producer.closed,
                }
            )
        )
        await ws.send_str(
            json.dumps({"type": "viewer_count", "count": self._viewer_count()})
        )
        await self._broadcast_viewer_count()
        latest_input = self.frame_hub.latest_input_preview_jpeg
        if latest_input:
            await self._send_input_preview(ws, latest_input)
        if role == "producer":
            await self._broadcast_event({"type": "producer_status", "active": True})
        latest = self.frame_hub.latest_output_jpeg
        if latest:
            await ws.send_bytes(latest)
        try:
            async for message in ws:
                if message.type == WSMsgType.BINARY:
                    if ws is not self._producer:
                        continue
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
            self._client_roles.pop(ws, None)
            if ws is self._producer:
                self._producer = None
                await self._broadcast_event({"type": "producer_status", "active": False})
            await self._broadcast_viewer_count()
        return ws

    def _viewer_count(self) -> int:
        if self._producer is None or self._producer.closed:
            return 0
        return sum(1 for role in self._client_roles.values() if role != "producer")

    async def _broadcast_viewer_count(self) -> None:
        await self._broadcast_event({"type": "viewer_count", "count": self._viewer_count()})

    async def _broadcast_event(self, payload: dict[str, Any]) -> None:
        message = json.dumps(payload)
        dead: list[web.WebSocketResponse] = []
        for client in list(self._clients):
            try:
                await asyncio.wait_for(client.send_str(message), timeout=0.25)
            except Exception as exc:  # noqa: BLE001
                print(f"[web] dropping client after text send failed: {exc}", flush=True)
                dead.append(client)
        for client in dead:
            self._drop_client(client)

    async def _send_input_preview(self, client: web.WebSocketResponse, payload: bytes) -> None:
        await asyncio.wait_for(
            client.send_str(
                json.dumps(
                    {
                        "type": "input_preview",
                        "jpeg": "data:image/jpeg;base64,"
                        + base64.b64encode(payload).decode("ascii"),
                    }
                )
            ),
            timeout=0.25,
        )

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
                await self._apply_and_broadcast_params({"prompt": prompt.strip()})

    async def _broadcast_output_loop(self) -> None:
        last_output_sequence = 0
        last_input_preview_sequence = 0
        last_input_preview_at = 0.0
        while True:
            await asyncio.sleep(0.016)
            input_preview = self.frame_hub.latest_input_preview_jpeg
            input_preview_sequence = self.frame_hub.latest_input_preview_sequence
            now = asyncio.get_running_loop().time()
            if (
                input_preview
                and input_preview_sequence != last_input_preview_sequence
                and now - last_input_preview_at >= 0.25
            ):
                last_input_preview_sequence = input_preview_sequence
                last_input_preview_at = now
                dead: list[web.WebSocketResponse] = []
                for client in list(self._clients):
                    try:
                        await self._send_input_preview(client, input_preview)
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[web] dropping client after preview send failed: {exc}",
                            flush=True,
                        )
                        dead.append(client)
                for client in dead:
                    self._drop_client(client)
            payload = self.frame_hub.latest_output_jpeg
            output_sequence = self.frame_hub.latest_output_sequence
            if not payload or output_sequence == last_output_sequence:
                continue
            last_output_sequence = output_sequence
            dead: list[web.WebSocketResponse] = []
            for client in list(self._clients):
                try:
                    await asyncio.wait_for(client.send_bytes(payload), timeout=0.25)
                except Exception as exc:  # noqa: BLE001
                    print(f"[web] dropping client after frame send failed: {exc}", flush=True)
                    dead.append(client)
            for client in dead:
                self._drop_client(client)

    def _drop_client(self, client: web.WebSocketResponse) -> None:
        self._clients.discard(client)
        self._client_roles.pop(client, None)
        if client is self._producer:
            self._producer = None
        asyncio.create_task(client.close())
