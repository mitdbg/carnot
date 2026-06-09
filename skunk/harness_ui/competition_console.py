"""Local operator console for the OfficeQA Cup API.

Runs a browser UI that connects to the Cup endpoint, displays incoming
questions, starts local reasoning jobs, and lets a human submit each completed
answer manually.

Usage:
    export CUP_BASE_URL=http://127.0.0.1:8765
    export CUP_TEAM_TOKEN=anything
    python competition_console.py --reasoner reference_agent:solve
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from websockets.exceptions import WebSocketException

from cup_kit.agent_runtime import AgentAnswer
from cup_kit.client import CupAPIError, CupClient
from cup_kit.protocol import (
    AGENT_MIN_REASONING_CHARS,
    MAX_SOURCE_DOC_REF_CHARS,
    QuestionFragment,
    RoundStatus,
    SubmissionScoredEvent,
    SubmitAcceptedResponse,
    SubmitRejectedResponse,
)

logger = logging.getLogger("competition_console")

Reasoner = Callable[[str], AgentAnswer | Awaitable[AgentAnswer] | dict[str, Any]]


@dataclass
class QuestionState:
    key: str
    round_num: int
    question_id: str
    prompt: str
    status: str = "queued"
    answer: str | None = None
    reasoning: str | None = None
    source_docs: list[str] = field(default_factory=list)
    error: str | None = None
    submission_id: str | None = None
    correct: bool | None = None
    points_awarded: float | None = None
    rejection_reason: str | None = None
    updated_at: str = field(default_factory=lambda: _now_iso())
    started_at: str | None = None
    finished_at: str | None = None
    elapsed_s: float | None = None


@dataclass
class ConsoleConfig:
    cup_base_url: str
    cup_team_token: str
    reasoner_ref: str
    concurrency: int
    reconnect_backoff_s: float = 1.0


class ConsoleState:
    def __init__(self, config: ConsoleConfig, reasoner: Reasoner) -> None:
        self.config = config
        self.reasoner = reasoner
        self.questions: dict[str, QuestionState] = {}
        self.round_num: int | None = None
        self.round_status: str = "DISCONNECTED"
        self.ends_at: str | None = None
        self.opens_at: str | None = None
        self.resubmits_left: int | None = None
        self.connection_status: str = "starting"
        self.last_event: str = ""
        self._lock = asyncio.Lock()
        self._ws_clients: set[WebSocket] = set()
        self._reasoning_tasks: dict[str, asyncio.Task[None]] = {}
        self._reasoning_sem = asyncio.Semaphore(max(1, config.concurrency))

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "cup_base_url": self.config.cup_base_url,
            "reasoner": self.config.reasoner_ref,
            "concurrency": self.config.concurrency,
            "connection_status": self.connection_status,
            "round_num": self.round_num,
            "round_status": self.round_status,
            "ends_at": self.ends_at,
            "opens_at": self.opens_at,
            "resubmits_left": self.resubmits_left,
            "last_event": self.last_event,
            "questions": [asdict(q) for q in self.questions.values()],
        }

    async def broadcast(self) -> None:
        async with self._lock:
            payload = json.dumps(self._snapshot_unlocked())
            clients = list(self._ws_clients)
        dead: list[WebSocket] = []
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._ws_clients.discard(ws)

    async def add_ws(self, ws: WebSocket) -> None:
        async with self._lock:
            self._ws_clients.add(ws)
            payload = json.dumps(self._snapshot_unlocked())
        await ws.send_text(payload)

    async def remove_ws(self, ws: WebSocket) -> None:
        async with self._lock:
            self._ws_clients.discard(ws)

    async def set_connection(self, status: str, event: str = "") -> None:
        async with self._lock:
            self.connection_status = status
            if event:
                self.last_event = event
        await self.broadcast()

    async def upsert_round(
        self,
        *,
        round_num: int,
        status: str,
        questions: list[QuestionFragment],
        ends_at: datetime | None,
        opens_at: datetime | None = None,
        resubmits_left: int | None = None,
        event: str = "",
    ) -> None:
        async with self._lock:
            if self.round_num is not None and self.round_num != round_num:
                self._cancel_reasoning_tasks()
                self.questions.clear()
            self.round_num = round_num
            self.round_status = status
            self.ends_at = ends_at.isoformat() if ends_at else None
            self.opens_at = opens_at.isoformat() if opens_at else self.opens_at
            self.resubmits_left = resubmits_left if resubmits_left is not None else self.resubmits_left
            if event:
                self.last_event = event
            for q in questions:
                key = _key(q.round_num, q.question_id)
                if key not in self.questions:
                    self.questions[key] = QuestionState(
                        key=key,
                        round_num=q.round_num,
                        question_id=q.question_id,
                        prompt=q.prompt,
                    )
        await self.broadcast()
        for q in questions:
            await self.ensure_reasoning_started(_key(q.round_num, q.question_id))

    async def close_round(self, round_num: int, status: str) -> None:
        async with self._lock:
            self.round_num = round_num
            self.round_status = status
            self.last_event = f"round state {status}; cleared questions"
            self._cancel_reasoning_tasks()
            self.questions.clear()
        await self.broadcast()

    def _cancel_reasoning_tasks(self) -> None:
        for task in self._reasoning_tasks.values():
            task.cancel()
        self._reasoning_tasks.clear()

    async def ensure_reasoning_started(self, key: str) -> None:
        if key in self._reasoning_tasks:
            return
        async with self._lock:
            q = self.questions.get(key)
            if q is None or q.status not in {"queued", "failed"}:
                return
            q.status = "running"
            q.error = None
            q.rejection_reason = None
            q.started_at = _now_iso()
            q.updated_at = q.started_at
        task = asyncio.create_task(self._run_reasoning(key))
        self._reasoning_tasks[key] = task
        task.add_done_callback(lambda _task: self._reasoning_tasks.pop(key, None))
        await self.broadcast()

    async def _run_reasoning(self, key: str) -> None:
        async with self._reasoning_sem:
            async with self._lock:
                q = self.questions.get(key)
                if q is None:
                    return
                prompt = q.prompt
                start = time.perf_counter()
            try:
                raw = self.reasoner(prompt)
                ans = await raw if inspect.isawaitable(raw) else raw
                normalized = _normalize_answer(ans)
                elapsed = round(time.perf_counter() - start, 3)
                async with self._lock:
                    q = self.questions.get(key)
                    if q is None:
                        return
                    q.status = "ready"
                    q.answer = normalized.answer
                    q.reasoning = normalized.reasoning
                    q.source_docs = _clean_source_docs(normalized.source_docs)
                    q.error = None
                    q.finished_at = _now_iso()
                    q.updated_at = q.finished_at
                    q.elapsed_s = elapsed
            except Exception as e:
                elapsed = round(time.perf_counter() - start, 3)
                logger.exception("reasoning failed for %s", key)
                async with self._lock:
                    q = self.questions.get(key)
                    if q is None:
                        return
                    q.status = "failed"
                    q.error = str(e)
                    q.finished_at = _now_iso()
                    q.updated_at = q.finished_at
                    q.elapsed_s = elapsed
            await self.broadcast()

    async def retry_reasoning(self, round_num: int, question_id: str) -> QuestionState:
        key = _key(round_num, question_id)
        async with self._lock:
            q = self.questions.get(key)
            if q is None:
                raise KeyError(key)
            q.status = "queued"
            q.answer = None
            q.reasoning = None
            q.source_docs = []
            q.error = None
            q.rejection_reason = None
            q.correct = None
            q.points_awarded = None
            q.submission_id = None
            q.updated_at = _now_iso()
        await self.ensure_reasoning_started(key)
        await self.broadcast()
        return q

    async def mark_submitting(self, round_num: int, question_id: str) -> QuestionState:
        key = _key(round_num, question_id)
        async with self._lock:
            q = self.questions.get(key)
            if q is None:
                raise KeyError(key)
            if q.status not in {"ready", "rejected", "submitted", "scored"}:
                raise ValueError(f"question is not ready to submit; status={q.status}")
            if not q.answer or not q.answer.strip():
                raise ValueError("answer is empty")
            q.status = "submitting"
            q.rejection_reason = None
            q.updated_at = _now_iso()
            return q

    async def mark_submit_response(
        self,
        round_num: int,
        question_id: str,
        resp: SubmitAcceptedResponse | SubmitRejectedResponse,
    ) -> None:
        key = _key(round_num, question_id)
        async with self._lock:
            q = self.questions[key]
            if resp.accepted:
                q.status = "submitted"
                q.submission_id = resp.submission_id
                q.correct = resp.score.correct
                q.points_awarded = resp.score.points_awarded
                q.rejection_reason = None
                self.resubmits_left = resp.tokens_remaining
            else:
                q.status = "rejected"
                q.rejection_reason = resp.reason.value
            q.updated_at = _now_iso()
        await self.broadcast()

    async def ready_to_submit(self) -> list[tuple[int, str]]:
        async with self._lock:
            return [
                (q.round_num, q.question_id)
                for q in self.questions.values()
                if q.status in {"ready", "rejected"} and q.answer and q.answer.strip()
            ]

    async def mark_scored(self, ev: SubmissionScoredEvent) -> None:
        async with self._lock:
            for q in self.questions.values():
                if q.question_id == ev.question_id and q.submission_id == ev.submission_id:
                    q.status = "scored"
                    q.correct = ev.correct
                    q.points_awarded = ev.points_awarded
                    q.updated_at = _now_iso()
                    break
            self.last_event = f"scored {ev.question_id}: {ev.correct}"
        await self.broadcast()


async def cup_listener(state: ConsoleState) -> None:
    while True:
        try:
            await state.set_connection("connecting")
            async with CupClient(state.config.cup_base_url, state.config.cup_team_token) as cup:
                current = await cup.get_current_round()
                if current.status == RoundStatus.ACTIVE:
                    await state.upsert_round(
                        round_num=current.round_num,
                        status=current.status.value,
                        questions=current.questions,
                        ends_at=current.ends_at,
                        resubmits_left=current.resubmits_left,
                        event="loaded current round",
                    )
                else:
                    await state.close_round(current.round_num, current.status.value)
                await state.set_connection("connected")
                async for ev in cup.events():
                    if ev.type == "round_started":
                        await state.upsert_round(
                            round_num=ev.round_num,
                            status=RoundStatus.ACTIVE.value,
                            questions=ev.questions,
                            ends_at=ev.ends_at,
                            opens_at=ev.opens_at,
                            event=f"round {ev.round_num} started",
                        )
                    elif ev.type == "round_state":
                        if ev.status == RoundStatus.ACTIVE:
                            async with state._lock:
                                state.round_num = ev.round_num
                                state.round_status = ev.status.value
                                state.last_event = f"round state {ev.status.value}"
                            await state.broadcast()
                        else:
                            await state.close_round(ev.round_num, ev.status.value)
                    elif ev.type == "submission_scored":
                        await state.mark_scored(ev)
        except (
            CupAPIError,
            httpx.HTTPError,
            OSError,
            asyncio.IncompleteReadError,
            WebSocketException,
        ) as e:
            logger.warning("Cup connection dropped: %s", e)
            await state.set_connection("disconnected", str(e))
            await asyncio.sleep(state.config.reconnect_backoff_s)


def build_app(config: ConsoleConfig, reasoner: Reasoner) -> FastAPI:
    state = ConsoleState(config, reasoner)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(cup_listener(state))
        try:
            yield
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title="OfficeQA Cup Local Console", lifespan=lifespan)

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(_INDEX_HTML)

    @app.get("/api/state")
    async def api_state() -> JSONResponse:
        return JSONResponse(await state.snapshot())

    @app.post("/api/questions/{round_num}/{question_id}/retry")
    async def api_retry(round_num: int, question_id: str) -> JSONResponse:
        try:
            await state.retry_reasoning(round_num, question_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail="question not found") from e
        return JSONResponse(await state.snapshot())

    async def submit_one(round_num: int, question_id: str) -> None:
        try:
            q = await state.mark_submitting(round_num, question_id)
        except KeyError as e:
            raise HTTPException(status_code=404, detail="question not found") from e
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e

        reasoning = q.reasoning or ""
        if len(reasoning) < AGENT_MIN_REASONING_CHARS:
            reasoning = (
                reasoning.rstrip()
                + "\n\nOperator note: The local reasoning module produced the answer above. "
                "This padding records that the answer was generated by the configured "
                "reasoning harness and is being manually submitted from the local console."
            )
        try:
            async with CupClient(config.cup_base_url, config.cup_team_token) as cup:
                resp = await cup.submit(
                    question_id,
                    q.answer or "",
                    reasoning=reasoning,
                    source_docs=_clean_source_docs(q.source_docs),
                )
            await state.mark_submit_response(round_num, question_id, resp)
        except (CupAPIError, httpx.HTTPError) as e:
            async with state._lock:
                q2 = state.questions[_key(round_num, question_id)]
                q2.status = "ready"
                q2.error = f"submit failed: {e}"
                q2.updated_at = _now_iso()
            await state.broadcast()
            raise HTTPException(status_code=502, detail=str(e)) from e

    @app.post("/api/questions/{round_num}/{question_id}/submit")
    async def api_submit(round_num: int, question_id: str) -> JSONResponse:
        await submit_one(round_num, question_id)
        return JSONResponse(await state.snapshot())

    @app.post("/api/submit_all")
    async def api_submit_all() -> JSONResponse:
        ready = await state.ready_to_submit()
        submitted = 0
        errors: list[str] = []
        for round_num, question_id in ready:
            try:
                await submit_one(round_num, question_id)
                submitted += 1
            except HTTPException as e:
                errors.append(f"{question_id}: {e.detail}")
        async with state._lock:
            state.last_event = f"submit_all submitted={submitted} errors={len(errors)}"
            if errors:
                state.last_event += " " + "; ".join(errors[:3])
        await state.broadcast()
        return JSONResponse(await state.snapshot())

    @app.websocket("/ws")
    async def ui_ws(ws: WebSocket) -> None:
        await ws.accept()
        await state.add_ws(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            await state.remove_ws(ws)

    return app


def _load_reasoner(ref: str) -> Reasoner:
    if ":" not in ref:
        raise ValueError("reasoner must be in module:function form")
    module_name, func_name = ref.split(":", 1)
    harness_dir = Path(__file__).resolve().parent
    for p in (str(harness_dir), os.getcwd()):
        if p not in sys.path:
            sys.path.insert(0, p)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"{ref} is not callable")
    return fn


def _normalize_answer(raw: AgentAnswer | dict[str, Any]) -> AgentAnswer:
    if isinstance(raw, AgentAnswer):
        return raw
    if isinstance(raw, dict):
        return AgentAnswer(
            answer=str(raw.get("answer", "")),
            reasoning=str(raw.get("reasoning", "")),
            source_docs=[str(x) for x in raw.get("source_docs", [])],
        )
    answer = getattr(raw, "answer", "")
    reasoning = getattr(raw, "reasoning", "")
    source_docs = getattr(raw, "source_docs", [])
    return AgentAnswer(
        answer=str(answer),
        reasoning=str(reasoning),
        source_docs=[str(x) for x in source_docs],
    )


def _clean_source_docs(source_docs: list[str]) -> list[str]:
    out: list[str] = []
    for doc in source_docs:
        s = str(doc).strip()
        if not s:
            continue
        out.append(s[:MAX_SOURCE_DOC_REF_CHARS])
    return out[:64]


def _key(round_num: int, question_id: str) -> str:
    return f"{round_num}:{question_id}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OfficeQA Cup local operator console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--cup-base-url", default=os.environ.get("CUP_BASE_URL", ""))
    parser.add_argument("--team-token", default=os.environ.get("CUP_TEAM_TOKEN", ""))
    parser.add_argument("--reasoner", default=os.environ.get("CONSOLE_REASONER", "reference_agent:solve"))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("CONSOLE_CONCURRENCY", "3")))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if not args.cup_base_url:
        raise SystemExit("CUP_BASE_URL or --cup-base-url is required")
    if not args.team_token:
        raise SystemExit("CUP_TEAM_TOKEN or --team-token is required")
    config = ConsoleConfig(
        cup_base_url=args.cup_base_url,
        cup_team_token=args.team_token,
        reasoner_ref=args.reasoner,
        concurrency=max(1, args.concurrency),
    )
    reasoner = _load_reasoner(args.reasoner)
    app = build_app(config, reasoner)
    uvicorn.run(app, host=args.host, port=args.port)


_INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OfficeQA Cup Console</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5e6b78;
      --line: #d9dee5;
      --accent: #1f7a68;
      --accent-2: #314f9f;
      --warn: #ad5b00;
      --bad: #b42318;
      --good: #157348;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    html, body { min-height: 100%; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      box-shadow: var(--shadow);
    }
    .topbar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 16px;
      align-items: center;
    }
    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
    }
    .meta {
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 10px;
      background: #fff;
      color: var(--muted);
      white-space: nowrap;
    }
    .pill strong { color: var(--ink); font-weight: 650; margin-left: 4px; }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 20px 32px;
    }
    .summary {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      box-shadow: var(--shadow);
      min-height: 64px;
    }
    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .metric strong {
      display: block;
      font-size: 20px;
      line-height: 1.1;
    }
    .event {
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: #fff;
      margin-bottom: 12px;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(320px, 440px) minmax(0, 1fr);
      gap: 12px;
      align-items: stretch;
      min-height: calc(100vh - 230px);
    }
    .list-panel,
    .detail-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
      align-items: center;
    }
    .head-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .panel-title,
    .qid {
      font-weight: 700;
      word-break: break-word;
    }
    .question-list {
      max-height: calc(100vh - 285px);
      overflow: auto;
    }
    .question-row {
      width: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      padding: 12px;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      background: #fff;
      color: var(--ink);
      text-align: left;
      cursor: pointer;
      min-height: 74px;
    }
    .question-row:hover,
    .question-row.selected {
      background: #f1f6f4;
    }
    .row-main {
      min-width: 0;
    }
    .row-title {
      font-weight: 700;
      margin-bottom: 3px;
    }
    .row-preview {
      color: var(--muted);
      overflow: hidden;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
    }
    .status {
      display: inline-flex;
      align-items: center;
      height: 24px;
      border-radius: 999px;
      padding: 2px 9px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
      border: 1px solid var(--line);
      white-space: nowrap;
    }
    .queued { color: var(--muted); background: #f5f6f8; }
    .running, .submitting { color: var(--accent-2); background: #eef3ff; }
    .ready { color: var(--accent); background: #eaf7f3; }
    .submitted, .scored { color: var(--good); background: #e9f7ef; }
    .failed, .rejected { color: var(--bad); background: #fff0ed; }
    .detail-body {
      padding: 14px;
      max-height: calc(100vh - 285px);
      overflow: auto;
    }
    .prompt {
      margin: 0 0 12px;
      color: var(--ink);
      font-size: 15px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin: 12px 0 4px;
      text-transform: uppercase;
    }
    .kv {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0 14px;
    }
    .kv div {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #fbfcfd;
      min-width: 0;
    }
    .kv span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 2px;
    }
    .kv strong {
      display: block;
      overflow-wrap: anywhere;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f5f7f9;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      max-height: 190px;
      overflow: auto;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .answer {
      font-size: 15px;
      font-weight: 700;
      background: #eef8f5;
    }
    .docs {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 6px;
    }
    .doc {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 8px;
      color: var(--muted);
      background: #fff;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      padding: 12px;
      border-top: 1px solid var(--line);
      background: #fbfcfd;
    }
    button {
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 0 12px;
      font-weight: 700;
      cursor: pointer;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    .message {
      color: var(--bad);
      font-size: 13px;
      margin-top: 8px;
      word-break: break-word;
    }
    .empty {
      padding: 40px 16px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      color: var(--muted);
      text-align: center;
      background: #fff;
    }
    .detail-empty {
      min-height: calc(100vh - 285px);
      display: grid;
      place-items: center;
      padding: 24px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 760px) {
      .topbar { grid-template-columns: 1fr; }
      .meta { justify-content: flex-start; }
      .summary { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .workspace { grid-template-columns: 1fr; }
      .question-list,
      .detail-body,
      .detail-empty { max-height: none; min-height: auto; }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>OfficeQA Cup Console</h1>
      <div class="meta">
        <span class="pill">Cup <strong id="cup">-</strong></span>
        <span class="pill">Connection <strong id="conn">-</strong></span>
        <span class="pill">Round <strong id="round">-</strong></span>
        <span class="pill">Status <strong id="roundStatus">-</strong></span>
        <span class="pill">Resubmits <strong id="resubmits">-</strong></span>
      </div>
    </div>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>Total</span><strong id="mTotal">0</strong></div>
      <div class="metric"><span>Running</span><strong id="mRunning">0</strong></div>
      <div class="metric"><span>Ready</span><strong id="mReady">0</strong></div>
      <div class="metric"><span>Submitted</span><strong id="mSubmitted">0</strong></div>
      <div class="metric"><span>Correct</span><strong id="mCorrect">0</strong></div>
    </section>
    <div id="lastEvent" class="event">Waiting for competition events...</div>
    <section class="workspace">
      <aside class="list-panel">
        <div class="panel-head">
          <div class="panel-title">Questions</div>
          <div class="head-actions">
            <button id="submitAllBtn" class="primary" onclick="submitAll()" disabled>Submit All</button>
            <span class="pill"><strong id="listCount">0</strong></span>
          </div>
        </div>
        <div id="questionList" class="question-list"></div>
      </aside>
      <section class="detail-panel">
        <div class="panel-head">
          <div id="detailTitle" class="panel-title">Question Details</div>
          <span id="detailStatus" class="status queued">none</span>
        </div>
        <div id="detailContent" class="detail-empty">Select a question from the list.</div>
      </section>
    </section>
  </main>
  <script>
    let state = null;
    let selectedKey = null;
    const $ = (id) => document.getElementById(id);
    const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const js = (s) => String(s ?? "").replace(/\\/g, "\\\\").replace(/'/g, "\\'");
    function counts(questions) {
      const by = (fn) => questions.filter(fn).length;
      return {
        total: questions.length,
        running: by(q => q.status === "running" || q.status === "submitting"),
        ready: by(q => q.status === "ready" || q.status === "rejected"),
        submitted: by(q => ["submitted", "scored"].includes(q.status)),
        correct: by(q => q.correct === true),
      };
    }
    function selectQuestion(key) {
      selectedKey = key;
      render(state);
    }
    function render(next) {
      if (!next) return;
      state = next;
      $("cup").textContent = next.cup_base_url || "-";
      $("conn").textContent = next.connection_status || "-";
      $("round").textContent = next.round_num ?? "-";
      $("roundStatus").textContent = next.round_status || "-";
      $("resubmits").textContent = next.resubmits_left ?? "-";
      const c = counts(next.questions || []);
      $("mTotal").textContent = c.total;
      $("mRunning").textContent = c.running;
      $("mReady").textContent = c.ready;
      $("mSubmitted").textContent = c.submitted;
      $("mCorrect").textContent = c.correct;
      $("lastEvent").textContent = next.last_event || "Waiting for competition events...";
      $("listCount").textContent = (next.questions || []).length;
      const readyForSubmit = (next.questions || []).filter(q => ["ready", "rejected"].includes(q.status) && q.answer).length;
      $("submitAllBtn").disabled = readyForSubmit === 0;
      if (!next.questions || !next.questions.length) {
        selectedKey = null;
        $("questionList").innerHTML = `<div class="empty">No questions received yet.</div>`;
        renderDetail(null);
        return;
      }
      if (!selectedKey || !next.questions.some(q => q.key === selectedKey)) {
        selectedKey = next.questions[0].key;
      }
      $("questionList").innerHTML = next.questions.map(q => {
        return `
          <button class="question-row ${q.key === selectedKey ? "selected" : ""}" onclick="selectQuestion('${js(q.key)}')">
            <div class="row-main">
              <div class="row-title">R${esc(q.round_num)} / ${esc(q.question_id)}</div>
              <div class="row-preview">${esc(q.prompt)}</div>
            </div>
            <span class="status ${esc(q.status)}">${esc(q.status)}</span>
          </button>`;
      }).join("");
      renderDetail(next.questions.find(q => q.key === selectedKey));
    }
    function renderDetail(q) {
      const status = $("detailStatus");
      const content = $("detailContent");
      if (!q) {
        $("detailTitle").textContent = "Question Details";
        status.className = "status queued";
        status.textContent = "none";
        content.className = "detail-empty";
        content.innerHTML = "Select a question from the list.";
        return;
      }
      $("detailTitle").textContent = `R${q.round_num} / ${q.question_id}`;
      status.className = `status ${q.status}`;
      status.textContent = q.status;
      const canSubmit = ["ready", "rejected", "submitted", "scored"].includes(q.status) && q.answer;
      const docs = (q.source_docs || []).map(d => `<span class="doc" title="${esc(d)}">${esc(d)}</span>`).join("");
      const verdict = q.correct === null || q.correct === undefined ? "" : `<div class="label">Score</div><pre>${q.correct ? "correct" : "wrong"}${q.points_awarded !== null && q.points_awarded !== undefined ? " / " + q.points_awarded + " pts" : ""}</pre>`;
      const err = q.error || q.rejection_reason;
      content.className = "detail-body";
      content.innerHTML = `
        <p class="prompt">${esc(q.prompt)}</p>
        <div class="kv">
          <div><span>Status</span><strong>${esc(q.status)}</strong></div>
          <div><span>Elapsed</span><strong>${q.elapsed_s !== null && q.elapsed_s !== undefined ? esc(q.elapsed_s) + "s" : "-"}</strong></div>
          <div><span>Updated</span><strong>${esc(q.updated_at || "-")}</strong></div>
        </div>
        ${q.answer ? `<div class="label">Answer</div><pre class="answer">${esc(q.answer)}</pre>` : ""}
        ${q.reasoning ? `<div class="label">Reasoning</div><pre>${esc(q.reasoning)}</pre>` : ""}
        ${docs ? `<div class="label">Source Docs</div><div class="docs">${docs}</div>` : ""}
        ${verdict}
        ${err ? `<div class="message">${esc(err)}</div>` : ""}
        <div class="actions">
          <button class="primary" onclick="submitAnswer(${Number(q.round_num)}, '${js(q.question_id)}')" ${canSubmit ? "" : "disabled"}>Submit</button>
          <button onclick="retry(${Number(q.round_num)}, '${js(q.question_id)}')">Retry</button>
        </div>`;
    }
    async function submitAnswer(roundNum, questionId) {
      await fetch(`/api/questions/${roundNum}/${encodeURIComponent(questionId)}/submit`, {method: "POST"});
    }
    async function retry(roundNum, questionId) {
      await fetch(`/api/questions/${roundNum}/${encodeURIComponent(questionId)}/retry`, {method: "POST"});
    }
    async function submitAll() {
      await fetch("/api/submit_all", {method: "POST"});
    }
    function connect() {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${scheme}://${location.host}/ws`);
      ws.onmessage = (ev) => render(JSON.parse(ev.data));
      ws.onclose = () => setTimeout(connect, 1000);
    }
    fetch("/api/state").then(r => r.json()).then(render);
    connect();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
