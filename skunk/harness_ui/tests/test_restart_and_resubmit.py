from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI

from skunk_server.api import RestartTaskBody, install_command_routes
from skunk_server.domain import AnswerCandidate, FailureRecord, TaskStatus
from skunk_server.task_queues import TaskQueues
from skunk_server.task_registry import TaskConflict, TaskRegistry


def _command_endpoint(
    registry: TaskRegistry, queues: TaskQueues, published: list[bool], path: str
):
    app = FastAPI()
    install_command_routes(app)
    app.state.registry = registry
    app.state.queues = queues
    app.state.coordinator = None
    app.state.broker = None
    app.state.publish_status = lambda: published.append(True)
    for route in app.routes:
        if getattr(route, "path", "") == path:
            return route.endpoint
    raise AssertionError(f"{path} route not installed")


def _json(response):
    return response.status_code, json.loads(response.body)


def _failed_no_answer_task(registry: TaskRegistry) -> str:
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "worker")
    assert attempt is not None
    ok = registry.fail_attempt(
        task.task_id,
        attempt.attempt_id,
        FailureRecord(
            attempt_id=attempt.attempt_id,
            error_type="RuntimeError",
            error_message="planning failed",
        ),
    )
    assert ok
    return task.task_id


def test_restart_failed_no_answer_requeues_with_feedback() -> None:
    registry = TaskRegistry()
    queues = TaskQueues()
    published: list[bool] = []
    task_id = _failed_no_answer_task(registry)
    endpoint = _command_endpoint(
        registry, queues, published, "/api/restart/{task_id:path}"
    )

    response = asyncio.run(
        endpoint(task_id, RestartTaskBody(client_id="me", feedback="try a simpler plan"))
    )

    assert _json(response) == (200, {"ok": True, "task_id": task_id})
    task = registry.get(task_id)
    assert task is not None
    assert task.status == TaskStatus.QUEUED
    assert task.cup_feedback == ["Operator restart feedback: try a simpler plan"]
    assert queues.agent.get_nowait() == task_id
    assert published == [True]


def test_restart_rejects_failed_task_that_already_has_answer() -> None:
    registry = TaskRegistry()
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "worker")
    assert attempt is not None
    assert registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(
            attempt_id=attempt.attempt_id,
            answer_text="bad",
            reasoning="reasoning",
        ),
    )
    task.status = TaskStatus.FAILED

    try:
        registry.restart_failed_no_answer(task.task_id)
    except TaskConflict as error:
        assert "already has an answer" in str(error)
    else:
        raise AssertionError("restart should reject answered tasks")


def test_incorrect_scored_answer_can_resubmit_when_tokens_remain() -> None:
    registry = TaskRegistry()
    registry.update_round(round_num=1, status="ACTIVE", resubmits_left=3)
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "worker")
    assert attempt is not None
    assert registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(
            attempt_id=attempt.attempt_id,
            answer_text="bad",
            reasoning="reasoning",
        ),
    )
    _task, _candidate, submission = registry.begin_candidate_submission(task.task_id)
    registry.record_submission_accepted(
        task.task_id,
        submission.local_submission_id,
        "cup-sub",
        tokens_remaining=3,
        correct=False,
        points_awarded=0.0,
    )

    _task, candidate, retry_submission = registry.begin_candidate_submission(task.task_id)

    assert candidate.answer_text == "bad"
    assert retry_submission.local_submission_id != submission.local_submission_id
    assert registry.get(task.task_id).status == TaskStatus.SUBMITTING


def test_resubmit_count_tracks_only_subsequent_cup_submits() -> None:
    registry = TaskRegistry()
    registry.update_round(round_num=1, status="ACTIVE", resubmits_left=3)
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "worker")
    assert attempt is not None
    assert registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(
            attempt_id=attempt.attempt_id,
            answer_text="bad",
            reasoning="reasoning",
        ),
    )
    _task, _candidate, initial = registry.begin_candidate_submission(task.task_id)
    registry.record_submission_accepted(
        task.task_id,
        initial.local_submission_id,
        "cup-sub-1",
        tokens_remaining=3,
        correct=False,
        points_awarded=0.0,
    )
    assert registry.round_state().resubmit_count == 0

    _task, _candidate, retry = registry.begin_candidate_submission(task.task_id)
    registry.record_submission_accepted(
        task.task_id,
        retry.local_submission_id,
        "cup-sub-2",
        tokens_remaining=2,
        correct=False,
        points_awarded=0.0,
    )

    task = registry.get(task.task_id)
    assert task is not None
    assert task.cup_submit_count == 2
    assert registry.round_state().resubmit_count == 1

    registry.update_round(round_num=2, status="ACTIVE", resubmits_left=3)

    assert registry.round_state().resubmit_count == 0


def test_incorrect_scored_answer_cannot_resubmit_without_tokens() -> None:
    registry = TaskRegistry()
    registry.update_round(round_num=1, status="ACTIVE", resubmits_left=0)
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "worker")
    assert attempt is not None
    assert registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(
            attempt_id=attempt.attempt_id,
            answer_text="bad",
            reasoning="reasoning",
        ),
    )
    _task, _candidate, submission = registry.begin_candidate_submission(task.task_id)
    registry.record_submission_accepted(
        task.task_id,
        submission.local_submission_id,
        "cup-sub",
        tokens_remaining=0,
        correct=False,
        points_awarded=0.0,
    )

    try:
        registry.begin_candidate_submission(task.task_id)
    except TaskConflict as error:
        assert "not ready" in str(error)
    else:
        raise AssertionError("resubmit should reject when no tokens remain")


def test_rerun_finished_task_requeues_and_keeps_cup_submit_history() -> None:
    registry = TaskRegistry()
    registry.update_round(round_num=1, status="ACTIVE", resubmits_left=3)
    queues = TaskQueues()
    published: list[bool] = []
    task, _ = registry.create_task(1, "Q1", "prompt")
    attempt = registry.begin_attempt(task.task_id, "worker")
    assert attempt is not None
    assert registry.complete_attempt(
        task.task_id,
        attempt.attempt_id,
        AnswerCandidate(
            attempt_id=attempt.attempt_id,
            answer_text="bad",
            reasoning="reasoning",
        ),
    )
    _task, _candidate, submission = registry.begin_candidate_submission(task.task_id)
    registry.record_submission_accepted(
        task.task_id,
        submission.local_submission_id,
        "cup-sub",
        tokens_remaining=3,
        correct=False,
        points_awarded=0.0,
    )
    endpoint = _command_endpoint(registry, queues, published, "/api/rerun/{task_id:path}")

    response = asyncio.run(endpoint(task.task_id, RestartTaskBody(client_id="me")))

    assert _json(response) == (200, {"ok": True, "task_id": task.task_id})
    task = registry.get(task.task_id)
    assert task is not None
    assert task.status == TaskStatus.QUEUED
    assert task.attempts == []
    assert task.answer_candidates == []
    assert task.submissions == []
    assert task.cup_submit_count == 1
    assert queues.agent.get_nowait() == task.task_id
    assert published == [True]
