"""Server-owned worker threads that execute reasoning attempts."""

from __future__ import annotations

import asyncio
import inspect
import queue
import threading
import traceback
from collections.abc import Callable
from typing import Any

from cup_kit.agent_runtime import AgentAnswer
from cup_kit.protocol import MAX_SOURCE_DOC_REF_CHARS

from skunk_server.domain import AnswerCandidate, Attempt, FailureRecord
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskRegistry

Reasoner = Callable[[str], AgentAnswer | dict[str, Any] | Any]
CompletionCallback = Callable[[str, str], None]


class AgentWorkerPool:
    def __init__(
        self,
        registry: TaskRegistry,
        queues: TaskQueues,
        reasoner: Reasoner,
        max_workers: int,
    ) -> None:
        self._registry = registry
        self._queues = queues
        self._reasoner = reasoner
        self._max_workers = max(1, max_workers)
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._on_completion: CompletionCallback | None = None

    def start(
        self,
        loop: asyncio.AbstractEventLoop,
        on_completion: CompletionCallback,
    ) -> None:
        self._loop = loop
        self._on_completion = on_completion
        for index in range(self._max_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"skunk-agent-{index + 1}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        self._stop.set()
        for _ in self._threads:
            try:
                self._queues.agent.put_nowait("")
            except queue.Full:
                break
        for thread in self._threads:
            thread.join(timeout=5)
        self._threads.clear()

    def _worker_loop(self) -> None:
        worker_id = threading.current_thread().name
        while not self._stop.is_set():
            try:
                task_id = self._queues.agent.get(timeout=0.5)
            except queue.Empty:
                continue
            if not task_id:
                self._queues.agent.task_done()
                continue
            attempt = self._registry.begin_attempt(task_id, worker_id)
            if attempt is None:
                self._queues.agent.task_done()
                continue
            if self._loop is not None and self._on_completion is not None:
                self._loop.call_soon_threadsafe(
                    self._on_completion,
                    task_id,
                    "processing",
                )
            outcome = "failed"
            try:
                task = self._registry.get(task_id)
                if task is None:
                    continue
                prompt = self._reasoner_prompt(task.prompt, attempt)
                raw = self._reasoner(prompt)
                if inspect.isawaitable(raw):
                    raw = asyncio.run(raw)
                answer, reasoning, source_docs = self._normalize_answer(raw)
                candidate = AnswerCandidate(
                    attempt_id=attempt.attempt_id,
                    answer_text=answer,
                    reasoning=reasoning,
                    source_docs=source_docs,
                )
                if self._registry.complete_attempt(task_id, attempt.attempt_id, candidate):
                    self._queues.enqueue_ready(task_id)
                    outcome = "ready"
            except Exception as error:
                failure = FailureRecord(
                    attempt_id=attempt.attempt_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback="".join(traceback.format_exception(error))[-20_000:],
                )
                if self._registry.fail_attempt(task_id, attempt.attempt_id, failure):
                    self._queues.enqueue_failed(task_id)
            finally:
                self._queues.agent.task_done()
                if self._loop is not None and self._on_completion is not None:
                    self._loop.call_soon_threadsafe(self._on_completion, task_id, outcome)

    @staticmethod
    def _reasoner_prompt(prompt: str, attempt: Attempt) -> str:
        context: list[str] = []
        if attempt.feedback:
            context.append(f"Human retry feedback:\n{attempt.feedback}")
        if attempt.context_feedback:
            context.append("Competition feedback:\n" + "\n".join(attempt.context_feedback))
        if not context:
            return prompt
        return prompt + "\n\n" + "\n\n".join(context)

    @staticmethod
    def _normalize_answer(raw: Any) -> tuple[str, str, list[str]]:
        if isinstance(raw, dict):
            answer = raw.get("answer", "")
            reasoning = raw.get("reasoning", "")
            source_docs = raw.get("source_docs", [])
        else:
            answer = getattr(raw, "answer", "")
            reasoning = getattr(raw, "reasoning", "")
            source_docs = getattr(raw, "source_docs", [])
        cleaned_docs = [
            str(item).strip()[:MAX_SOURCE_DOC_REF_CHARS]
            for item in source_docs
            if str(item).strip()
        ][:64]
        return str(answer), str(reasoning), cleaned_docs
