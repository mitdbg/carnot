"""Ephemeral human-worker identities and user-saved display names."""

from __future__ import annotations

import threading

from skunk_server.domain import HumanWorker, utc_now


class HumanWorkerRegistry:
    def __init__(self) -> None:
        self._workers: dict[str, HumanWorker] = {}
        self._lock = threading.RLock()

    def create_worker(self) -> HumanWorker:
        with self._lock:
            worker = HumanWorker()
            self._workers[worker.worker_id] = worker
            return worker

    def get(self, worker_id: str) -> HumanWorker | None:
        with self._lock:
            return self._workers.get(worker_id)

    def save_display_name(self, worker_id: str, display_name: str) -> HumanWorker:
        cleaned = display_name.strip()
        if len(cleaned) > 80:
            raise ValueError("display name must be at most 80 characters")
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None:
                raise KeyError(worker_id)
            worker.display_name = cleaned or None
            worker.last_seen_at = utc_now()
            return worker

    def disconnect(self, worker_id: str) -> None:
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is not None:
                worker.connected = False
                worker.last_seen_at = utc_now()

    def list_workers(self) -> list[HumanWorker]:
        with self._lock:
            return list(self._workers.values())
