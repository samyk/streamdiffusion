from __future__ import annotations

import asyncio
import contextlib
import json
import queue
from typing import Any

from websockets.exceptions import ConnectionClosed

from websockets.asyncio.server import serve

from .frames import SharedState


class ControlServer:
    def __init__(
        self,
        host: str,
        port: int,
        command_queue: "queue.Queue[dict[str, Any]]",
        state: SharedState,
    ) -> None:
        self.host = host
        self.port = port
        self.command_queue = command_queue
        self.state = state

    async def run(self) -> None:
        # TouchDesigner's websocketDAT does not reliably answer server pings while
        # the UI thread is busy, so keepalive is disabled to avoid screen spam.
        async with serve(
            self._handler,
            self.host,
            self.port,
            ping_interval=None,
            ping_timeout=None,
        ):
            await asyncio.Future()

    async def _handler(self, websocket) -> None:  # noqa: ANN001 - websockets protocol type changes
        status_task = asyncio.create_task(self._send_status_loop(websocket))
        try:
            async for raw_message in websocket:
                response = self._handle_message(raw_message)
                await websocket.send(json.dumps(response))
        except Exception:
            return
        finally:
            status_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await status_task

    def _handle_message(self, raw_message: str | bytes) -> dict[str, Any]:
        try:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")
            message = json.loads(raw_message)
            if not isinstance(message, dict):
                raise ValueError("Control message must be a JSON object")
            self.command_queue.put_nowait(message)
            return {"type": "ack", "ok": True, "command": message.get("type")}
        except Exception as exc:  # noqa: BLE001 - sent back to controller
            return {"type": "ack", "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    async def _send_status_loop(self, websocket) -> None:  # noqa: ANN001
        while True:
            await asyncio.sleep(1.0)
            try:
                await websocket.send(json.dumps(self.state.snapshot()))
            except ConnectionClosed:
                return

