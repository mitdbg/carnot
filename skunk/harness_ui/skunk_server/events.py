"""WebSocket connection management and state broadcasting."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket


class EventHub:
    def __init__(self, snapshot: Callable[[str | None], dict[str, Any]]) -> None:
        self._snapshot = snapshot
        self._clients: dict[WebSocket, str] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, worker_id: str) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients[websocket] = worker_id
        await websocket.send_json(self._snapshot(worker_id))

    async def disconnect(self, websocket: WebSocket) -> str | None:
        async with self._lock:
            return self._clients.pop(websocket, None)

    async def broadcast(self) -> None:
        async with self._lock:
            clients = list(self._clients.items())
        dead: list[WebSocket] = []
        for websocket, worker_id in clients:
            try:
                await websocket.send_json(self._snapshot(worker_id))
            except Exception:
                dead.append(websocket)
        if dead:
            async with self._lock:
                for websocket in dead:
                    self._clients.pop(websocket, None)
