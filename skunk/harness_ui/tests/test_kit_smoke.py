"""Smoke test: run the practice harness in-process and drive an agent
against it.

Validates the full agent-facing wire surface end-to-end:

- WS handshake yields a `connected` envelope.
- `round_started` arrives with `opens_at`.
- `POST /v1/submit` with a correct answer returns
  `accepted=True, correct=True`.
- `submission_scored` event is pushed to the WS subscriber.
- Validation rejections fire the right `SubmitRejectionReason` enum
  values.

Run via ``python -m pytest tests/``.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

import pytest

import uvicorn

from cup_kit.client import CupClient
from cup_kit.practice_server import _Config, _parse_args, build_app
from cup_kit.protocol import SubmitRejectionReason


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def _running_harness(round_seconds: int = 8, start_delay: int = 0):
    """Start the practice harness on an ephemeral port; yield base URL."""
    port = _free_port()
    cfg = _Config(
        host="127.0.0.1",
        port=port,
        start_delay=start_delay,
        round_seconds=round_seconds,
        questions_path=Path(__file__).parent.parent / "practice_questions.json",
        once=False,
    )
    app, _state, _stop = build_app(cfg)
    config = uvicorn.Config(app, host=cfg.host, port=cfg.port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # Wait for the server to come up.
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    else:
        raise RuntimeError("harness failed to start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_full_round_cycle_with_correct_answers() -> None:
    """A correct submission scores correct=True and emits
    submission_scored to the WS subscriber."""
    async with _running_harness(round_seconds=6) as base_url:
        async with CupClient(base_url, "smoke-token") as cup:
            seen_round_started = False
            seen_scored = False
            async for ev in cup.events():
                if ev.type == "round_started":
                    seen_round_started = True
                    # Submit the canonical answer for the first
                    # question in the round.
                    q = ev.questions[0]
                    canonical_by_id = {
                        "practice_r1_q1": "60",
                        "practice_r1_q2": "56",
                        "practice_r1_q3": "50",
                        "practice_r2_q1": "1776",
                        "practice_r2_q2": "3.14",
                        "practice_r2_q3": "-80",
                        "practice_r3_q1": "299792458",
                        "practice_r3_q2": "86,400",
                        "practice_r3_q3": "6912",
                    }
                    resp = await cup.submit(
                        q.question_id,
                        canonical_by_id[q.question_id],
                        reasoning=(
                            "Test reasoning that is comfortably longer than the "
                            "AGENT_MIN_REASONING_CHARS floor of 100 characters so "
                            "the practice harness accepts the submission."
                        ),
                        source_docs=["test.pdf#p1"],
                    )
                    assert resp.accepted is True, resp
                    assert resp.score.correct is True
                elif ev.type == "submission_scored":
                    seen_scored = True
                    assert ev.correct is True
                    break
            assert seen_round_started, "did not receive round_started"
            assert seen_scored, "did not receive submission_scored"


@pytest.mark.asyncio
async def test_short_reasoning_rejected() -> None:
    """Reasoning under AGENT_MIN_REASONING_CHARS triggers
    REASONING_TOO_SHORT."""
    async with _running_harness(round_seconds=6) as base_url:
        async with CupClient(base_url, "smoke-token") as cup:
            async for ev in cup.events():
                if ev.type != "round_started":
                    continue
                q = ev.questions[0]
                resp = await cup.submit(
                    q.question_id,
                    "anything",
                    reasoning="too short",
                    source_docs=["test.pdf#p1"],
                )
                assert resp.accepted is False
                assert resp.reason == SubmitRejectionReason.REASONING_TOO_SHORT
                return


@pytest.mark.asyncio
async def test_empty_source_docs_accepted() -> None:
    """source_docs is optional — empty list is accepted."""
    async with _running_harness(round_seconds=6) as base_url:
        async with CupClient(base_url, "smoke-token") as cup:
            async for ev in cup.events():
                if ev.type != "round_started":
                    continue
                q = ev.questions[0]
                resp = await cup.submit(
                    q.question_id,
                    "anything",
                    reasoning=(
                        "Test reasoning that is comfortably longer than the "
                        "AGENT_MIN_REASONING_CHARS floor of 100 characters so "
                        "we exercise the no-citations path cleanly."
                    ),
                    source_docs=[],
                )
                assert resp.accepted is True
                return


@pytest.mark.asyncio
async def test_round_not_yet_open_enforced() -> None:
    """Submission before opens_at is rejected with ROUND_NOT_YET_OPEN."""
    # 2s open delay; agent submits immediately on round_started.
    async with _running_harness(round_seconds=6, start_delay=2) as base_url:
        async with CupClient(base_url, "smoke-token") as cup:
            async for ev in cup.events():
                if ev.type != "round_started":
                    continue
                q = ev.questions[0]
                # Deliberately skip the opens_at sleep that the
                # production runtime does — the harness must reject.
                resp = await cup.submit(
                    q.question_id,
                    "60",
                    reasoning=(
                        "Test reasoning that is comfortably longer than the "
                        "AGENT_MIN_REASONING_CHARS floor of 100 characters so "
                        "we exercise the open-delay gate cleanly."
                    ),
                    source_docs=["test.pdf#p1"],
                )
                assert resp.accepted is False
                assert resp.reason == SubmitRejectionReason.ROUND_NOT_YET_OPEN
                return


@pytest.mark.asyncio
async def test_missing_token_rejected_with_401() -> None:
    """HTTP requests without a bearer token get 401, not silent allow."""
    async with _running_harness(round_seconds=6) as base_url:
        # Bypass CupClient (which always sends a token) and hit the
        # endpoint directly.
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as raw:
            r = await raw.get("/v1/round/current")
            assert r.status_code == 401


def test_default_questions_path_resolves() -> None:
    """The ``--questions`` default must point at a file that exists.
    Catches regressions like the kit-layout move that broke the
    relative path."""
    cfg = _parse_args([])
    assert cfg.questions_path.is_file(), (
        f"default --questions path does not exist: {cfg.questions_path}"
    )
