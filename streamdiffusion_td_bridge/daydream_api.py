from __future__ import annotations

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .frames import SharedState
from .sdtd_mapper import daydream_params_to_commands, normalize_stream_params


class DaydreamApiServer:
    """Minimal Daydream Streams API for StreamDiffusionTD Remote backend.

    Video transport remains NDI; this server only handles parameter hot-updates
    compatible with Daydream's PATCH /v1/streams/{id} payload shape.
    """

    def __init__(
        self,
        host: str,
        port: int,
        command_queue: queue.Queue[dict[str, Any]],
        state: SharedState,
        stream_id: str = "remote-1",
        input_name: str = "td_streamdiffusion_in",
        output_name: str = "streamdiffusion_out",
    ) -> None:
        self.host = host
        self.port = port
        self.command_queue = command_queue
        self.state = state
        self.stream_id = stream_id
        self.input_name = input_name
        self.output_name = output_name
        self._thread: threading.Thread | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._params: dict[str, Any] = {}

    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/health":
                    self._json(200, {"ok": True})
                    return
                if path == f"/v1/streams/{server.stream_id}":
                    snap = server.state.snapshot()
                    self._json(
                        200,
                        {
                            "id": server.stream_id,
                            "pipeline": "streamdiffusion",
                            "status": snap.get("status", "running"),
                            "params": server._params,
                            "runtime": snap,
                        },
                    )
                    return
                self._json(404, {"success": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path != "/v1/streams":
                    self._json(404, {"success": False, "error": "not found"})
                    return
                payload = self._read_json()
                params = payload.get("params", payload)
                server._apply_params(params)
                self._json(
                    200,
                    {
                        "id": server.stream_id,
                        "pipeline": "streamdiffusion",
                        "whip_url": f"ndi://{server.input_name}",
                        "whep_url": f"ndi://{server.output_name}",
                        "params": server._params,
                    },
                )

            def do_PATCH(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if not path.startswith("/v1/streams/"):
                    self._json(404, {"success": False, "error": "not found"})
                    return
                stream_id = path.rsplit("/", 1)[-1]
                if stream_id != server.stream_id:
                    self._json(404, {"success": False, "error": f"unknown stream {stream_id}"})
                    return
                payload = self._read_json()
                params = payload.get("params", payload)
                server._apply_params(params)
                self._json(
                    200,
                    {
                        "success": True,
                        "id": server.stream_id,
                        "pipeline": "streamdiffusion",
                        "params": server._params,
                    },
                )

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    return {}
                return data if isinstance(data, dict) else {}

        self._httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="daydream-api",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def _apply_params(self, params: dict[str, Any]) -> None:
        if not params:
            return
        previous = dict(self._params)
        params = normalize_stream_params(params, previous)
        self._params.update(params)
        for command in daydream_params_to_commands(params, previous=previous):
            self.command_queue.put_nowait(command)
