"""Bounded task-ID queues for agent, ready, and failed work."""

from __future__ import annotations

import queue


class TaskQueues:
    def __init__(self, max_size: int = 200) -> None:
        self.agent: queue.Queue[str] = queue.Queue(maxsize=max_size)
        self.ready: queue.Queue[str] = queue.Queue(maxsize=max_size)
        self.failed: queue.Queue[str] = queue.Queue(maxsize=max_size)

    def enqueue_agent(self, task_id: str) -> None:
        self.agent.put(task_id)

    def enqueue_ready(self, task_id: str) -> None:
        self.ready.put(task_id)

    def enqueue_failed(self, task_id: str) -> None:
        self.failed.put(task_id)
