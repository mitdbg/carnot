"""OfficeQA Cup — local practice harness.

Run this in one terminal, your edited ``reference_agent.py`` in
another, with ``CUP_BASE_URL`` pointed at this server. The harness
drives a 3-round lifecycle on autopilot so you can validate your
agent's plumbing (WS connect, round_started handling, single-call
submit with reasoning + source_docs, submission_scored echo) before
pointing the same agent at the live cup.

The wire surface implemented here is identical to the live cup's
agent-facing surface — request/response validation, enum values, and
scoring decisions all come from ``cup_kit.protocol`` and
``cup_kit.scorer``, which mirror the canonical server-side
definitions. A contract test in research CI guards against drift.

What's different (intentionally) from the live cup:

- Auth is permissive: any non-empty bearer token is accepted; the
  team_id is derived as ``team_practice`` (or
  ``team_<first-8-chars>`` if a token is supplied) for log readability.
- Lifecycle is timer-driven, not operator-driven. After the last
  round, the harness loops back to round 1.
- Single team per harness instance — no leaderboard, no competition.
- No persistence: state is lost on process exit.

Usage::

    python practice_server.py [--port 8765] [--host 127.0.0.1]
                              [--round-seconds 180]
                              [--questions questions/practice_questions.json]
                              [--once] [--verbose]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from cup_kit import VERSION
from cup_kit.protocol import (
    AGENT_MIN_REASONING_CHARS,
    MAX_ANSWER_TEXT_CHARS,
    MAX_REASONING_CHARS,
    MAX_RESUBMITS,
    MAX_SOURCE_DOC_REF_CHARS,
    MAX_SOURCE_DOCS_ENTRIES,
    ROUND_OPEN_DELAY_SECONDS,
    QuestionFragment,
    RoundCurrentResponse,
    RoundStartedEvent,
    RoundStateEvent,
    RoundStatus,
    RoundType,
    ScoreFragment,
    SubmissionHistoryEntry,
    SubmissionScoredEvent,
    SubmissionType,
    SubmitAcceptedResponse,
    SubmitRejectedResponse,
    SubmitRejectionReason,
    SubmitRequest,
    TeamQuestionStatus,
    TeamStatusResponse,
)
from cup_kit.scorer import Scorer

logger = logging.getLogger("cup_kit.practice")


# ---------------------------------------------------------------------------
# Colored log output (best-effort; falls back to plain text if no TTY).
# ---------------------------------------------------------------------------


def _supports_color() -> bool:
    return sys.stdout.isatty()


_C_GREEN = "\033[32m" if _supports_color() else ""
_C_RED = "\033[31m" if _supports_color() else ""
_C_DIM = "\033[2m" if _supports_color() else ""
_C_YELLOW = "\033[33m" if _supports_color() else ""
_C_RESET = "\033[0m" if _supports_color() else ""


def _ok(msg: str) -> None:
    print(f"{_C_GREEN}[OK]{_C_RESET} {msg}", flush=True)


def _info(msg: str) -> None:
    print(f"{_C_DIM}[..]{_C_RESET} {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"{_C_YELLOW}[!!]{_C_RESET} {msg}", flush=True)


# ---------------------------------------------------------------------------
# In-memory state.
# ---------------------------------------------------------------------------


@dataclass
class _Question:
    question_id: str
    round_num: int
    prompt: str
    canonical_answer: str


@dataclass
class _Round:
    round_num: int
    status: RoundStatus = RoundStatus.PRE_ROUND
    round_type: RoundType = "normal"
    started_at: datetime | None = None
    ends_at: datetime | None = None
    questions: list[_Question] = field(default_factory=list)


@dataclass
class _SubmissionRecord:
    submission_id: str
    question_id: str
    answer_text: str
    submitted_at: datetime
    correct: bool
    points_awarded: float
    superseded: bool = False


@dataclass
class _State:
    """Single mutable harness state, swapped under one asyncio Lock."""

    rounds: list[_Round]
    start_delay_seconds: int = ROUND_OPEN_DELAY_SECONDS
    current_idx: int = 0
    resubmits_left: int = MAX_RESUBMITS
    # question_id → ordered list (oldest first) of submissions
    submissions: dict[str, list[_SubmissionRecord]] = field(default_factory=dict)
    connected_ws: set[WebSocket] = field(default_factory=set)
    cycle: int = 0  # increments each time the lifecycle wraps


# ---------------------------------------------------------------------------
# Question loading.
# ---------------------------------------------------------------------------


def _load_questions(path: Path) -> list[_Round]:
    """Load a questions JSON file into harness rounds.

    Expected shape::

        {
          "rounds": [
            {
              "round_num": 1,
              "questions": [
                {
                  "question_id": "...",
                  "prompt": "...",
                          "canonical_answer": "..."
                }
              ]
            }
          ]
        }

    A bare list at the top level (``[{round_num, questions: [...]}, ...]``)
    also works.
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        rounds_in = raw.get("rounds", [])
    else:
        rounds_in = raw
    rounds: list[_Round] = []
    for r in rounds_in:
        round_num = int(r["round_num"])
        questions: list[_Question] = []
        for q in r.get("questions", []):
            questions.append(
                _Question(
                    question_id=str(q["question_id"]),
                    round_num=round_num,
                    prompt=str(q["prompt"]),
                    canonical_answer=str(q["canonical_answer"]),
                )
            )
        rounds.append(_Round(round_num=round_num, questions=questions))
    rounds.sort(key=lambda r: r.round_num)
    return rounds


def _display_label(rounds: list[_Round], idx: int) -> str:
    if idx < 0 or idx >= len(rounds):
        return ""
    return f"Round {rounds[idx].round_num}"


# ---------------------------------------------------------------------------
# Wire-shape helpers.
# ---------------------------------------------------------------------------


def _question_fragment(q: _Question) -> QuestionFragment:
    return QuestionFragment(
        question_id=q.question_id,
        round_num=q.round_num,
        prompt=q.prompt,
    )


def _round_current_response(state: _State) -> RoundCurrentResponse:
    if not state.rounds:
        # Cold-start defensive shape.
        return RoundCurrentResponse(
            round_num=0,
            status=RoundStatus.PRE_ROUND,
            resubmits_left=state.resubmits_left,
        )
    r = state.rounds[state.current_idx]
    questions = (
        [_question_fragment(q) for q in r.questions]
        if r.status in (RoundStatus.ACTIVE, RoundStatus.CLOSED, RoundStatus.RESULTS)
        else []
    )
    return RoundCurrentResponse(
        round_num=r.round_num,
        status=r.status,
        ends_at=r.ends_at,
        questions=questions,
        resubmits_left=state.resubmits_left,
        round_type=r.round_type,
        display_label=_display_label(state.rounds, state.current_idx),
    )


def _team_status_response(state: _State, team_id: str) -> TeamStatusResponse:
    r = state.rounds[state.current_idx] if state.rounds else None
    per_question: list[TeamQuestionStatus] = []
    if r is not None:
        for q in r.questions:
            subs = state.submissions.get(q.question_id, [])
            history = [
                SubmissionHistoryEntry(
                    submission_id=s.submission_id,
                    answer_text=s.answer_text,
                    submitted_at=s.submitted_at,
                    correct=s.correct,
                    points_awarded=s.points_awarded,
                    superseded=s.superseded,
                )
                for s in subs
            ]
            latest = subs[-1] if subs else None
            per_question.append(
                TeamQuestionStatus(
                    question_id=q.question_id,
                    submission_id=latest.submission_id if latest else None,
                    answer_text=latest.answer_text if latest else None,
                    correct=latest.correct if latest else None,
                    points_awarded=latest.points_awarded if latest else None,
                    speed_bonus=False,
                    submission_history=history,
                )
            )
    return TeamStatusResponse(
        team_id=team_id,
        team_name="Practice Team",
        sponsor="Practice",
        round_num=r.round_num if r else 0,
        resubmits_left=state.resubmits_left,
        per_question=per_question,
    )


# ---------------------------------------------------------------------------
# Submission validation. Mirrors the live cup's submit handler.
# ---------------------------------------------------------------------------


def _validate_submit(
    body: SubmitRequest, current_round: _Round | None, start_delay_seconds: int
) -> SubmitRejectedResponse | None:
    # Empty-answer guard.
    if not body.answer_text.strip():
        return SubmitRejectedResponse(
            reason=SubmitRejectionReason.ANSWER_EMPTY,
            tokens_remaining=0,
        )
    if len(body.answer_text) > MAX_ANSWER_TEXT_CHARS:
        return SubmitRejectedResponse(
            reason=SubmitRejectionReason.ANSWER_TOO_LONG,
            tokens_remaining=0,
            submitted_chars=len(body.answer_text),
            max_chars=MAX_ANSWER_TEXT_CHARS,
        )
    if body.submission_type == SubmissionType.AGENT:
        if len(body.reasoning) < AGENT_MIN_REASONING_CHARS:
            return SubmitRejectedResponse(
                reason=SubmitRejectionReason.REASONING_TOO_SHORT,
                tokens_remaining=0,
                submitted_chars=len(body.reasoning),
                min_chars=AGENT_MIN_REASONING_CHARS,
            )
    if len(body.reasoning) > MAX_REASONING_CHARS:
        return SubmitRejectedResponse(
            reason=SubmitRejectionReason.REASONING_TOO_LONG,
            tokens_remaining=0,
            submitted_chars=len(body.reasoning),
            max_chars=MAX_REASONING_CHARS,
        )
    # source_docs is optional — empty list is accepted. Per-entry
    # validation below still applies if entries are present.
    if len(body.source_docs) > MAX_SOURCE_DOCS_ENTRIES:
        return SubmitRejectedResponse(
            reason=SubmitRejectionReason.SOURCE_DOCS_TOO_MANY,
            tokens_remaining=0,
            submitted_chars=len(body.source_docs),
            max_chars=MAX_SOURCE_DOCS_ENTRIES,
        )
    for i, doc in enumerate(body.source_docs):
        if not doc.strip():
            return SubmitRejectedResponse(
                reason=SubmitRejectionReason.SOURCE_DOC_EMPTY,
                tokens_remaining=0,
                source_doc_index=i,
            )
        if len(doc) > MAX_SOURCE_DOC_REF_CHARS:
            return SubmitRejectedResponse(
                reason=SubmitRejectionReason.SOURCE_DOC_TOO_LONG,
                tokens_remaining=0,
                source_doc_index=i,
                submitted_chars=len(doc),
                max_chars=MAX_SOURCE_DOC_REF_CHARS,
            )
    # Round-state guards.
    if current_round is None or current_round.status != RoundStatus.ACTIVE:
        return SubmitRejectedResponse(
            reason=SubmitRejectionReason.ROUND_NOT_ACTIVE,
            tokens_remaining=0,
        )
    # Round-open grace gate (matches the live cup's 3-2-1-Go fairness
    # window).
    now = datetime.now(timezone.utc)
    if current_round.started_at is not None:
        opens_at = current_round.started_at + timedelta(seconds=start_delay_seconds)
        if now < opens_at:
            return SubmitRejectedResponse(
                reason=SubmitRejectionReason.ROUND_NOT_YET_OPEN,
                tokens_remaining=0,
            )
    if not any(q.question_id == body.question_id for q in current_round.questions):
        return SubmitRejectedResponse(
            reason=SubmitRejectionReason.QUESTION_NOT_IN_ROUND,
            tokens_remaining=0,
        )
    return None


# ---------------------------------------------------------------------------
# App + lifecycle driver.
# ---------------------------------------------------------------------------


@dataclass
class _Config:
    host: str
    port: int
    start_delay: int
    round_seconds: int
    questions_path: Path
    once: bool


def _ensure_token(authorization: str | None, x_cup_auth: str | None) -> str:
    raw = authorization or x_cup_auth or ""
    if raw.lower().startswith("bearer "):
        token = raw[7:].strip()
    else:
        token = raw.strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
        )
    return token


def _team_id_for(token: str) -> str:
    # Stable, log-readable id derived from the token without leaking
    # the full secret.
    suffix = token[:8] if len(token) > 8 else token
    return f"team_{suffix}"


def _make_app(
    state: _State,
    lock: asyncio.Lock,
    scorer: Scorer,
    cfg: _Config,
    stop: asyncio.Event,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(_drive_lifecycle(state, lock, cfg, stop))
        try:
            yield
        finally:
            stop.set()
            try:
                await asyncio.wait_for(task, timeout=3.0)
            except asyncio.TimeoutError:
                task.cancel()

    app = FastAPI(
        title="OfficeQA Cup — Practice Harness",
        version=VERSION,
        lifespan=lifespan,
    )

    @app.get("/v1/round/current")
    async def round_current(request: Request) -> Any:
        _ensure_token(request.headers.get("authorization"), request.headers.get("x-cup-auth"))
        async with lock:
            return JSONResponse(_round_current_response(state).model_dump(mode="json"))

    @app.get("/v1/team/status")
    async def team_status(request: Request) -> Any:
        token = _ensure_token(request.headers.get("authorization"), request.headers.get("x-cup-auth"))
        async with lock:
            return JSONResponse(_team_status_response(state, _team_id_for(token)).model_dump(mode="json"))

    @app.post("/v1/submit")
    async def submit(request: Request) -> Any:
        token = _ensure_token(request.headers.get("authorization"), request.headers.get("x-cup-auth"))
        try:
            payload = await request.json()
        except Exception as e:  # pragma: no cover - defensive
            raise HTTPException(status_code=400, detail=f"bad json: {e}") from e
        try:
            body = SubmitRequest.model_validate(payload)
        except Exception as e:  # pragma: no cover - pydantic raises ValidationError
            raise HTTPException(status_code=422, detail=str(e)) from e
        async with lock:
            current_round = state.rounds[state.current_idx] if state.rounds else None
            rejection = _validate_submit(body, current_round, state.start_delay_seconds)
            if rejection is not None:
                _warn(
                    f"/v1/submit  team={_team_id_for(token)}  "
                    f"question={body.question_id}  rejected={rejection.reason.value}"
                )
                return JSONResponse(rejection.model_dump(mode="json"))
            assert current_round is not None
            q = next(qq for qq in current_round.questions if qq.question_id == body.question_id)
            result = scorer.score(q.canonical_answer, body.answer_text)
            now = datetime.now(timezone.utc)
            prior = state.submissions.get(body.question_id, [])
            for s in prior:
                s.superseded = True
            sub_id = f"sub_{uuid.uuid4().hex[:12]}"
            record = _SubmissionRecord(
                submission_id=sub_id,
                question_id=body.question_id,
                answer_text=body.answer_text,
                submitted_at=now,
                correct=result.correct,
                points_awarded=result.points_awarded,
            )
            state.submissions.setdefault(body.question_id, []).append(record)
            superseded_id = prior[-1].submission_id if prior else None
            # Only consume a resubmit token on a non-no-op resubmit. A
            # no-op (same normalized answer) doesn't burn a token —
            # mirrors the live cup behavior.
            no_op = False
            if prior:
                prev = prior[-1]
                if scorer.is_same_answer(prev.answer_text, body.answer_text):
                    no_op = True
                else:
                    state.resubmits_left = max(0, state.resubmits_left - 1)
            response = SubmitAcceptedResponse(
                submission_id=sub_id,
                no_op=no_op,
                score=ScoreFragment(correct=result.correct, points_awarded=result.points_awarded),
                tokens_remaining=state.resubmits_left,
                superseded_submission_id=superseded_id if not no_op else None,
            )
            _ok(
                f"/v1/submit  team={_team_id_for(token)}  "
                f"question={body.question_id}  reasoning={len(body.reasoning)} chars  "
                f"source_docs={len(body.source_docs)}"
            )
            color = _C_GREEN if result.correct else _C_RED
            verdict = "correct" if result.correct else "wrong"
            print(
                f"{color}[OK]{_C_RESET} scored: {verdict}  " f"points={result.points_awarded}",
                flush=True,
            )
            # Push submission_scored to the connected WS so the agent
            # gets the same feedback channel as on the live cup.
            scored_event = SubmissionScoredEvent(
                submission_id=sub_id,
                question_id=body.question_id,
                correct=result.correct,
                points_awarded=result.points_awarded,
            )
            await _broadcast(state, scored_event.model_dump(mode="json"))
            # If this was the final outstanding question, tell the
            # operator the agent is done so they're not left wondering
            # why the harness is silent while the round window
            # continues to tick down.
            answered = len(state.submissions)
            total = len(current_round.questions)
            if answered == total and current_round.started_at is not None:
                close_at = current_round.started_at + timedelta(seconds=cfg.round_seconds)
                remaining = (close_at - datetime.now(timezone.utc)).total_seconds()
                if remaining > 0:
                    _ok(
                        f"all {total} questions answered; "
                        f"round {current_round.round_num} closes in {int(remaining)}s "
                        f"(use --round-seconds to change practice round length)"
                    )
            return JSONResponse(response.model_dump(mode="json"))

    @app.websocket("/v1/team/events")
    async def team_events(ws: WebSocket) -> None:
        # WebSocket auth: prefer headers, fall back to ?token=.
        token_header = ws.headers.get("authorization") or ws.headers.get("x-cup-auth") or ""
        token_qs = ws.query_params.get("token", "")
        raw = token_header[7:].strip() if token_header.lower().startswith("bearer ") else token_header.strip()
        if not raw:
            raw = token_qs.strip()
        if not raw:
            await ws.close(code=4401)
            return
        team_id = _team_id_for(raw)
        await ws.accept()
        await ws.send_json({"type": "connected", "team_id": team_id})
        _ok(f"WS connected: team={team_id}")
        async with lock:
            state.connected_ws.add(ws)
        try:
            # Keep the connection alive; the lifecycle driver pushes
            # all events. The client never sends frames after the
            # handshake.
            while True:
                msg = await ws.receive_text()
                # Drain client frames silently; not part of the wire
                # contract but harmless.
                logger.debug("ignoring client frame: %s", msg[:80])
        except WebSocketDisconnect:
            pass
        finally:
            async with lock:
                state.connected_ws.discard(ws)
            _info(f"WS disconnected: team={team_id}")

    return app


async def _broadcast(state: _State, payload: dict[str, Any]) -> None:
    """Send a JSON frame to every connected WS, removing dead sockets."""
    text = json.dumps(payload, default=str)
    dead: list[WebSocket] = []
    for ws in list(state.connected_ws):
        try:
            await ws.send_text(text)
        except (RuntimeError, WebSocketDisconnect):
            dead.append(ws)
    for ws in dead:
        state.connected_ws.discard(ws)


async def _drive_lifecycle(state: _State, lock: asyncio.Lock, cfg: _Config, stop: asyncio.Event) -> None:
    """Background task that cycles rounds on a timer."""

    async def run_one_round(round_idx: int) -> None:
        async with lock:
            r = state.rounds[round_idx]
            r.status = RoundStatus.ACTIVE
            r.started_at = datetime.now(timezone.utc)
            r.ends_at = r.started_at + timedelta(seconds=cfg.round_seconds)
            state.current_idx = round_idx
            state.submissions.clear()
            state.resubmits_left = MAX_RESUBMITS
            opens_at = r.started_at + timedelta(seconds=cfg.start_delay)
            event = RoundStartedEvent(
                round_num=r.round_num,
                ends_at=r.ends_at,
                opens_at=opens_at,
                questions=[_question_fragment(q) for q in r.questions],
            )
            _info(
                f"round {r.round_num} starts: {len(r.questions)} questions, "
                f"closes in {cfg.round_seconds}s "
                f"(use --round-seconds to change practice round length)"
            )
            await _broadcast(state, event.model_dump(mode="json"))

        # Wait until the close, with a "closes in 10s" warning along the
        # way. The window ticks regardless of whether the agent has
        # submitted — matches the real cup's fixed-window behavior, so
        # slow agents will see late submits rejected just as they would
        # on the day.
        warn_at = max(0.0, cfg.round_seconds - 10.0)
        if warn_at > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=warn_at)
                return
            except asyncio.TimeoutError:
                pass
            if not stop.is_set():
                _info(
                    f"round {r.round_num} closes in 10s — "
                    f"use --round-seconds to change practice round length"
                )
        remaining = min(10.0, float(cfg.round_seconds))
        try:
            await asyncio.wait_for(stop.wait(), timeout=remaining)
            return
        except asyncio.TimeoutError:
            pass
        async with lock:
            r = state.rounds[round_idx]
            r.status = RoundStatus.CLOSED
            await _broadcast(
                state,
                RoundStateEvent(round_num=r.round_num, status=r.status).model_dump(mode="json"),
            )
            _info(f"round {r.round_num} → CLOSED")
        # Brief gap before RESULTS so a watching team sees the
        # transition.
        await asyncio.sleep(1.0)
        if stop.is_set():
            return
        async with lock:
            r = state.rounds[round_idx]
            r.status = RoundStatus.RESULTS
            await _broadcast(
                state,
                RoundStateEvent(round_num=r.round_num, status=r.status).model_dump(mode="json"),
            )
            _info(f"round {r.round_num} → RESULTS")
        await asyncio.sleep(2.0)

    # Wait for first WS connect before starting the lifecycle, so a
    # team that launches the harness first and the agent second
    # doesn't miss round 1.
    _info(f"HTTP serving on http://{cfg.host}:{cfg.port}")
    _info("waiting for first WS connect before starting round 1")
    while not stop.is_set():
        async with lock:
            connected = bool(state.connected_ws)
        if connected:
            break
        await asyncio.sleep(0.25)
    if stop.is_set():
        return
    # Small extra delay so the first round_started doesn't race the
    # WS handshake.
    await asyncio.sleep(0.5)

    while not stop.is_set():
        for idx in range(len(state.rounds)):
            if stop.is_set():
                break
            await run_one_round(idx)
        async with lock:
            for r in state.rounds:
                r.status = RoundStatus.PRE_ROUND
                r.started_at = None
                r.ends_at = None
            state.submissions.clear()
            state.cycle += 1
        if stop.is_set():
            # External shutdown (Ctrl+C / server stop). Don't claim
            # we finished a cycle cleanly — the operator interrupted.
            return
        if cfg.once:
            _ok(f"practice cycle {state.cycle} complete; exiting (--once)")
            stop.set()
            return
        _info(
            f"looping back to round 1 for another practice cycle "
            f"(use Ctrl+C to stop the server)"
        )
        await asyncio.sleep(1.0)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str]) -> _Config:
    p = argparse.ArgumentParser(description="OfficeQA Cup practice harness")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument(
        "--round-seconds",
        type=int,
        default=180,
        help="active window per round in seconds (default: %(default)s)",
    )
    # Bundled question sets live in questions/, next to cup_kit/.
    p.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "questions" / "practice_questions.json",
    )
    p.add_argument("--once", action="store_true", help="run one full cycle then exit 0 (for CI)")
    p.add_argument("--verbose", action="store_true")
    ns = p.parse_args(argv)
    return _Config(
        host=ns.host,
        port=ns.port,
        # ``start_delay`` is not surfaced as a CLI flag in the kit —
        # it's a live-cup fairness gate (matches the audience countdown)
        # that's not useful during local practice. The smoke test still
        # exercises the gate by constructing ``_Config`` directly.
        start_delay=0,
        round_seconds=ns.round_seconds,
        questions_path=ns.questions,
        once=ns.once,
    )


def build_app(cfg: _Config) -> tuple[FastAPI, _State, asyncio.Event]:
    """Build the FastAPI app + harness state. Exposed for tests."""
    rounds = _load_questions(cfg.questions_path)
    if not rounds:
        raise ValueError(f"no rounds loaded from {cfg.questions_path}")
    state = _State(rounds=rounds, start_delay_seconds=cfg.start_delay)
    lock = asyncio.Lock()
    scorer = Scorer()
    stop = asyncio.Event()
    app = _make_app(state, lock, scorer, cfg, stop)
    return app, state, stop


def main(argv: list[str] | None = None) -> int:
    cfg = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if False else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app, _state, _stop = build_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")
    return 0
