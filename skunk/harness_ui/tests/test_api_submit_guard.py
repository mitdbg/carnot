from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI

from skunk_server.api import SubmitTaskBody, install_command_routes
from skunk_server.task_registry import TaskConflict


class _Coordinator:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[str] = []
        self.error = error

    async def submit_ready(self, task_id: str) -> None:
        self.calls.append(task_id)
        if self.error is not None:
            raise self.error


class _Broker:
    def __init__(self, holder: str | None = None, open_reviews: int = 0) -> None:
        self.holder = holder
        self.open_reviews = open_reviews

    def get_lock_holder(self, _task_id: str) -> str | None:
        return self.holder

    def count_open_reviews(self, _task_id: str) -> int:
        return self.open_reviews


class _MissingBroker(_Broker):
    def get_lock_holder(self, _task_id: str) -> str | None:
        raise KeyError(_task_id)


def _endpoint(coordinator: _Coordinator, broker):
    app = FastAPI()
    install_command_routes(app)
    app.state.coordinator = coordinator
    app.state.broker = broker
    for route in app.routes:
        if getattr(route, "path", "") == "/api/submit/{task_id:path}":
            return route.endpoint
    raise AssertionError("submit route not installed")


def _json(response):
    return response.status_code, json.loads(response.body)


def test_submit_body_requires_client_id() -> None:
    with pytest.raises(Exception):
        SubmitTaskBody.model_validate({})
    with pytest.raises(Exception):
        SubmitTaskBody.model_validate({"client_id": ""})


def test_submit_rejects_foreign_lock() -> None:
    coordinator = _Coordinator()
    endpoint = _endpoint(coordinator, _Broker(holder="other"))

    response = asyncio.run(endpoint("1:Q1", SubmitTaskBody(client_id="me")))

    assert _json(response) == (409, {"ok": False, "error": "Task is locked by another user"})
    assert coordinator.calls == []


def test_submit_rejects_open_reviews() -> None:
    coordinator = _Coordinator()
    endpoint = _endpoint(coordinator, _Broker(holder="me", open_reviews=1))

    status, body = _json(asyncio.run(endpoint("1:Q1", SubmitTaskBody(client_id="me"))))

    assert status == 409
    assert "open review" in body["error"]
    assert coordinator.calls == []


def test_submit_success_with_no_review_conflicts() -> None:
    coordinator = _Coordinator()
    endpoint = _endpoint(coordinator, _Broker())

    response = asyncio.run(endpoint("1:Q1", SubmitTaskBody(client_id="me")))

    assert _json(response) == (200, {"ok": True, "task_id": "1:Q1"})
    assert coordinator.calls == ["1:Q1"]


def test_submit_missing_task_from_guard_returns_404() -> None:
    coordinator = _Coordinator()
    endpoint = _endpoint(coordinator, _MissingBroker())

    response = asyncio.run(endpoint("1:Q1", SubmitTaskBody(client_id="me")))

    assert _json(response) == (
        404,
        {"ok": False, "error": "task not found or has no answer"},
    )
    assert coordinator.calls == []


def test_submit_preserves_existing_conflicts() -> None:
    coordinator = _Coordinator(error=TaskConflict("not ready"))
    endpoint = _endpoint(coordinator, _Broker())

    response = asyncio.run(endpoint("1:Q1", SubmitTaskBody(client_id="me")))

    assert _json(response) == (409, {"ok": False, "error": "not ready"})
