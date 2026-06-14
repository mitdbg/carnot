"""Competition-console adapter for the Skunk OfficeQA reasoner."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from cup_kit.agent_runtime import AgentAnswer

HERE = Path(__file__).resolve().parent
SKUNK_ROOT = HERE.parent
REPO_ROOT = SKUNK_ROOT.parent
SKUNK_SRC = SKUNK_ROOT / "src"

if str(SKUNK_SRC) not in sys.path:
    sys.path.insert(0, str(SKUNK_SRC))


def _set_default_env() -> None:
    os.environ.setdefault(
        "SKUNK_PAGE_INDEX_DIR",
        str(SKUNK_ROOT / "cache/build_v3"),
    )
    os.environ.setdefault(
        "OFFICEQA_PARSED_JSON_DIR",
        str(REPO_ROOT / "data/officeqa/treasury_bulletins_parsed/jsons"),
    )
    os.environ.setdefault(
        "OFFICEQA_PDF_DIR",
        str(REPO_ROOT / "data/officeqa/treasury_bulletin_pdfs"),
    )


async def solve(
    prompt: str,
    *,
    human_intervention_handler=None,
    trace_event_handler=None,
    human_review_register=None,
    recompute_sink=None,
) -> AgentAnswer:
    return await SkunkReasoner().solve(
        prompt,
        human_intervention_handler=human_intervention_handler,
        trace_event_handler=trace_event_handler,
        human_review_register=human_review_register,
        recompute_sink=recompute_sink,
    )


async def solve_with_trace(
    prompt: str,
    *,
    human_intervention_handler=None,
    trace_event_handler=None,
    human_review_register=None,
    recompute_sink=None,
) -> tuple[AgentAnswer, list[dict]]:
    return await SkunkReasoner().solve_with_trace(
        prompt,
        human_intervention_handler=human_intervention_handler,
        trace_event_handler=trace_event_handler,
        human_review_register=human_review_register,
        recompute_sink=recompute_sink,
    )


async def recompute(state: dict, overrides: dict) -> str:
    """Revise an answer after a human review resolves: re-run ONLY compute over the first
    attempt's cached entries (`state`, an orchestrator.RecomputeState JSON) with the human's
    corrections (`overrides`: branch_id -> raw response JSON) swapped in. No re-plan / retrieve /
    extract — deterministic and cheap. Returns the revised answer text."""
    from skunk import SkunkConfig
    from skunk.common import ExecutionContext
    from skunk.orchestrator import RecomputeState, recompute_answer

    _set_default_env()
    config = SkunkConfig.from_env()
    if not Path(config.prompt_overrides_path).is_absolute():
        config.prompt_overrides_path = str(SKUNK_ROOT / config.prompt_overrides_path)
    snapshot = RecomputeState.from_jsonable(state)
    # Each override is the human's source-indexed review items (a JSON list of
    # {_src, description?, unit?, value?}); recompute_answer applies them per branch.
    parsed: dict[int, list[dict]] = {}
    for branch_id, response in overrides.items():
        if not (response or "").strip():
            continue
        items = json.loads(response)
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, list):
            parsed[int(branch_id)] = [i for i in items if isinstance(i, dict)]
    ctx = ExecutionContext(question=snapshot.question, config=config)
    try:
        return await recompute_answer(snapshot, parsed, ctx)
    finally:
        ctx.close()


class SkunkReasoner:
    @staticmethod
    def set_default_env() -> None:
        _set_default_env()

    async def solve(
        self,
        prompt: str,
        *,
        human_intervention_handler=None,
        trace_event_handler=None,
        human_review_register=None,
        recompute_sink=None,
    ) -> AgentAnswer:
        answer, _events = await self.solve_with_trace(
            prompt,
            human_intervention_handler=human_intervention_handler,
            trace_event_handler=trace_event_handler,
            human_review_register=human_review_register,
            recompute_sink=recompute_sink,
        )
        return answer

    async def solve_with_trace(
        self,
        prompt: str,
        *,
        human_intervention_handler=None,
        trace_event_handler=None,
        human_review_register=None,
        recompute_sink=None,
    ) -> tuple[AgentAnswer, list[dict]]:
        answer, events = await self.execute(
            prompt,
            human_intervention_handler=human_intervention_handler,
            trace_event_handler=trace_event_handler,
            human_review_register=human_review_register,
            recompute_sink=recompute_sink,
        )
        source_docs = self.source_docs_from_events(events)
        return (
            AgentAnswer(
                answer=answer,
                reasoning=self.reasoning_summary(events, source_docs),
                source_docs=source_docs,
            ),
            events,
        )

    async def execute(
        self,
        prompt: str,
        *,
        human_intervention_handler=None,
        trace_event_handler=None,
        human_review_register=None,
        human_reviews_discard=None,
        recompute_sink=None,
    ) -> tuple[str, list[dict]]:
        from skunk import (
            MissingData,
            Orchestrator,
            SkunkConfig,
            StepFailed,
            load_prompt_overrides,
        )

        _set_default_env()
        config = SkunkConfig.from_env()
        if not Path(config.prompt_overrides_path).is_absolute():
            config.prompt_overrides_path = str(
                SKUNK_ROOT / config.prompt_overrides_path
            )

        overrides_path = Path(config.prompt_overrides_path)
        prompt_overrides = (
            load_prompt_overrides(overrides_path) if overrides_path.exists() else ()
        )

        # Master switch for ALL human-in-the-loop behavior — the verify/figure/lookup gates
        # AND the broker recovery flow both hinge on the handler being present. Default ON
        # keeps the competition-server behavior; SKUNK_HUMAN_INTERVENTION=0 drops the handler
        # for a fully autonomous run (the orchestrator then uses the console channel, which
        # is inert while the SKUNK_HUMAN_* gates are off, and a MissingData fails the attempt
        # instead of paging a human).
        if os.environ.get("SKUNK_HUMAN_INTERVENTION", "1").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            human_intervention_handler = None
            human_review_register = None
            human_reviews_discard = None

        orch = Orchestrator(
            prompt,
            config=config,
            prompt_overrides=prompt_overrides,
            verbose=os.environ.get("SKUNK_CONSOLE_VERBOSE", "").lower()
            in {"1", "true", "yes"},
            human_intervention_handler=human_intervention_handler,
            human_review_register=human_review_register,
            human_reviews_discard=human_reviews_discard,
        )
        if trace_event_handler is not None:
            original_emit = orch.ctx.emit

            def emit_and_forward(
                message: str,
                level: str | None = None,
                *,
                kind: str | None = None,
                data: dict | None = None,
            ) -> None:
                original_emit(message, level, kind=kind, data=data)
                trace_event_handler(dict(orch.ctx.events[-1]))

            orch.ctx.emit = emit_and_forward  # type: ignore[method-assign]

        def ship_snapshot() -> None:
            # Hand the recompute snapshot to the server so a later human-review resolve can
            # revise this answer without re-planning (see human_work_broker). Shipped on BOTH
            # the success and failure paths: a task whose compute failed still has a snapshot of
            # its branch entries, so a human correction can re-run compute and possibly fix it.
            if recompute_sink is not None and orch.recompute_state is not None:
                try:
                    recompute_sink(orch.recompute_state.to_jsonable())
                except Exception:
                    pass  # best-effort; never fail the answer over it

        try:
            try:
                answer = await orch.execute()
            except (MissingData, StepFailed) as e:
                # Dump the trace BEFORE re-raising so failures (the interesting case for
                # debugging MissingData / StepFailed) are inspectable offline.
                _dump_console_trace(prompt, orch.ctx, error=f"{type(e).__name__}: {e}")
                ship_snapshot()
                raise RuntimeError(f"Skunk failed: {e}") from e
            except asyncio.CancelledError:
                # Round closed → the worker pool cancelled this run to free the worker.
                # Dump the partial trace (how far it got before the deadline) for inspection,
                # then propagate the cancellation — never swallow it.
                _dump_console_trace(
                    prompt,
                    orch.ctx,
                    error="CancelledError: round closed before completion",
                    status="cancelled",
                )
                raise
            _dump_console_trace(prompt, orch.ctx, answer=answer)
            ship_snapshot()
            return answer, list(orch.ctx.events)
        finally:
            orch.ctx.close()

    @staticmethod
    def reasoning_summary(events: list[dict], source_docs: list[str]) -> str:
        return _reasoning_summary(events, source_docs)

    @staticmethod
    def structured_reasoning_payload(
        events: list[dict],
        source_docs: list[str],
    ) -> dict[str, Any]:
        return _structured_reasoning_payload(events, source_docs)

    @staticmethod
    def branch_search_details(branch: dict[str, Any]) -> dict[str, Any]:
        return _branch_search_details(branch)

    @staticmethod
    def source_docs_from_events(events: list[dict]) -> list[str]:
        return _source_docs_from_events(events)

    @staticmethod
    def append_doc(docs: list[str], seen: set[str], month: str, page: str) -> None:
        _append_doc(docs, seen, month, page)


def _dump_console_trace(
    prompt: str,
    ctx,
    *,
    answer: str | None = None,
    error: str | None = None,
    status: str | None = None,
) -> None:
    """Best-effort per-question trace dump for the competition console (the UI path,
    unlike `eval_e2e.py`, otherwise persists nothing). Writes `<prompt-slug>-<hash>.json`
    — the full prompt, status, answer/error, and the question's event stream — into
    `SKUNK_CONSOLE_TRACE_DIR` when that env var is set (the launcher points it at a
    per-run dir). `status` overrides the inferred ok/failed (e.g. "cancelled" for a run
    abandoned at round close). Never raises: a trace-dump failure must not break the run."""
    import hashlib

    trace_dir = os.environ.get("SKUNK_CONSOLE_TRACE_DIR")
    if not trace_dir:
        return
    try:
        out_dir = Path(trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower())[:60].strip("-") or "question"
        payload = {
            "prompt": prompt,
            "status": status or ("failed" if error is not None else "ok"),
            "error": error,
            "answer": answer,
            "events": list(ctx.events),
        }
        (out_dir / f"{slug}-{digest}.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
    except Exception:
        pass  # best-effort only


def _reasoning_summary(events: list[dict], source_docs: list[str]) -> str:
    return json.dumps(
        _structured_reasoning_payload(events, source_docs), ensure_ascii=False
    )


def _structured_reasoning_payload(
    events: list[dict], source_docs: list[str]
) -> dict[str, Any]:
    branches_by_id: dict[int, dict[str, Any]] = {}
    branch_order: list[int] = []
    branch_steps: dict[int, list[dict[str, Any]]] = {}
    compute_code: str | None = None
    compute_attempt: int | None = None
    compute_attempts: list[dict[str, Any]] = []
    replan_count = 0
    counted_recovery_rounds: set[int] = set()
    last_plan_label = None
    human_directed_retrieval = False

    for evt in events:
        kind = evt.get("kind")
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        message = str(evt.get("message", ""))

        if message == "human_directed_retrieval":
            human_directed_retrieval = True

        if kind == "plan" and data:
            last_plan_label = data.get("label")
            recovery_round = data.get("recovery_round")
            if last_plan_label in {"replan_pending", "replan"}:
                if isinstance(recovery_round, int):
                    if recovery_round not in counted_recovery_rounds:
                        counted_recovery_rounds.add(recovery_round)
                        replan_count += 1
                elif last_plan_label == "replan":
                    replan_count += 1
            for branch in data.get("branches", []):
                if not isinstance(branch, dict):
                    continue
                branch_id = branch.get("branch_id")
                if not isinstance(branch_id, int):
                    continue
                branches_by_id[branch_id] = branch
                if branch_id not in branch_order:
                    branch_order.append(branch_id)
            continue

        if kind == "step" and isinstance(data, dict):
            branch_id = data.get("branch_id")
            if isinstance(branch_id, int):
                branch_steps.setdefault(branch_id, []).append(evt)
            if evt.get("op") == "compute":
                code = data.get("code")
                if isinstance(code, str) and code:
                    compute_code = code
                attempt = data.get("attempt")
                if isinstance(attempt, int):
                    compute_attempt = attempt

        if message.startswith("codegen_code"):
            code = data.get("code")
            attempt = data.get("attempt")
            if not isinstance(code, str) or not code:
                match = re.match(
                    r"codegen_code attempt=(\d+) code=(.*)", message, re.DOTALL
                )
                if match:
                    attempt = int(match.group(1))
                    try:
                        parsed_code = ast.literal_eval(match.group(2))
                    except (SyntaxError, ValueError):
                        parsed_code = None
                    if isinstance(parsed_code, str):
                        code = parsed_code
            if isinstance(code, str) and code:
                compute_code = code
            if isinstance(attempt, int):
                compute_attempt = attempt
                compute_attempts.append(
                    {"attempt": attempt, "code": code, "status": "generated"}
                )
            continue

        if message.startswith("exec_failed"):
            match = re.match(
                r"exec_failed attempt=(\d+) error=(.*)", message, re.DOTALL
            )
            if match:
                attempt = int(match.group(1))
                for item in reversed(compute_attempts):
                    if item["attempt"] == attempt:
                        item["status"] = "failed"
                        try:
                            item["result"] = ast.literal_eval(match.group(2))
                        except (SyntaxError, ValueError):
                            item["result"] = match.group(2)
                        break
            continue

        if message.startswith("exec_result"):
            match = re.match(r"exec_result attempt=(\d+) text=(.*)", message, re.DOTALL)
            if match:
                attempt = int(match.group(1))
                for item in reversed(compute_attempts):
                    if item["attempt"] == attempt:
                        item["status"] = "ok"
                        try:
                            item["result"] = ast.literal_eval(match.group(2))
                        except (SyntaxError, ValueError):
                            item["result"] = match.group(2)
                        break

    branches: list[dict[str, Any]] = []
    for branch_id in branch_order:
        branch = branches_by_id.get(branch_id)
        if branch is None:
            continue
        steps = branch_steps.get(branch_id, [])
        last_step = steps[-1] if steps else None
        last_step_data = (
            last_step.get("data")
            if isinstance(last_step, dict) and isinstance(last_step.get("data"), dict)
            else {}
        )
        output = last_step_data.get("summary") or last_step_data.get("error") or None
        branches.append(
            {
                "branch_id": branch_id,
                "kind": branch.get("kind", "branch"),
                "searched": _branch_search_details(branch),
                "blocks": output.get("blocks", []) if isinstance(output, dict) else [],
                "output": output,
                "output_step": last_step.get("op")
                if isinstance(last_step, dict)
                else None,
                "status": "failed" if last_step_data.get("error") else "ok",
            }
        )

    return {
        "summary": {
            "pipeline": last_plan_label or "plan",
            "source_docs": source_docs[:8],
            "branch_count": len(branches),
            "replan_count": replan_count,
            "human_directed_retrieval": human_directed_retrieval,
        },
        "branches": branches,
        "python_code": compute_code,
        "python_attempt": compute_attempt,
        "python_attempts": compute_attempts,
    }


def _branch_search_details(branch: dict[str, Any]) -> dict[str, Any]:
    if branch.get("kind") == "retrieve":
        return {
            "key": branch.get("key"),
            "period": branch.get("period"),
            "visual_only": branch.get("visual_only"),
        }
    return {
        "target": branch.get("target"),
        "src": branch.get("src"),
    }


def _source_docs_from_events(events: list[dict]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for evt in events:
        data = evt.get("data")
        if isinstance(data, dict):
            summary = data.get("summary")
            if isinstance(summary, dict):
                for value in summary.get("values", []):
                    if not isinstance(value, dict):
                        continue
                    month = value.get("bulletin")
                    pages = value.get("pages")
                    if isinstance(month, str) and isinstance(pages, list):
                        for page in pages:
                            if isinstance(page, int):
                                _append_doc(docs, seen, month, str(page))
                for page_ref in summary.get("pages", []):
                    if not isinstance(page_ref, dict):
                        continue
                    month = page_ref.get("month")
                    page = page_ref.get("page")
                    if isinstance(month, str) and isinstance(page, int):
                        _append_doc(docs, seen, month, str(page))
        msg = str(evt.get("message", ""))
        for month, page in re.findall(
            r"PageRef\([^)]*month=([0-9]{4}-[0-9]{2})[^)]*page=(\d+)", msg
        ):
            _append_doc(docs, seen, month, page)
        for month, page in re.findall(
            r"'bulletin': '([0-9]{4}-[0-9]{2})', 'page': (\d+)", msg
        ):
            _append_doc(docs, seen, month, page)
        for month, page in re.findall(
            r'"bulletin": "([0-9]{4}-[0-9]{2})", "page": (\d+)', msg
        ):
            _append_doc(docs, seen, month, page)
    return docs[:64]


def _append_doc(docs: list[str], seen: set[str], month: str, page: str) -> None:
    ref = f"Treasury Bulletin {month} PDF page {int(page)}"
    if ref not in seen:
        seen.add(ref)
        docs.append(ref)
