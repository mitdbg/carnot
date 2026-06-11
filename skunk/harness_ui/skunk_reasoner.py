"""Competition-console adapter for the Skunk OfficeQA reasoner."""

from __future__ import annotations

import ast
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


async def solve(prompt: str) -> AgentAnswer:
    return await SkunkReasoner().solve(prompt)


async def solve_with_trace(prompt: str) -> tuple[AgentAnswer, list[dict]]:
    return await SkunkReasoner().solve_with_trace(prompt)


class SkunkReasoner:
    @staticmethod
    def set_default_env() -> None:
        _set_default_env()

    async def solve(self, prompt: str) -> AgentAnswer:
        answer, _events = await self.solve_with_trace(prompt)
        return answer

    async def solve_with_trace(self, prompt: str) -> tuple[AgentAnswer, list[dict]]:
        answer, events = await self.execute(prompt)
        source_docs = self.source_docs_from_events(events)
        return (
            AgentAnswer(
                answer=answer,
                reasoning=self.reasoning_summary(events, source_docs),
                source_docs=source_docs,
            ),
            events,
        )

    async def execute(self, prompt: str) -> tuple[str, list[dict]]:
        from skunk import MissingData, Orchestrator, SkunkConfig, StepFailed, load_prompt_overrides

        _set_default_env()
        config = SkunkConfig.from_env()
        if not Path(config.prompt_overrides_path).is_absolute():
            config.prompt_overrides_path = str(SKUNK_ROOT / config.prompt_overrides_path)

        overrides_path = Path(config.prompt_overrides_path)
        prompt_overrides = load_prompt_overrides(overrides_path) if overrides_path.exists() else ()

        orch = Orchestrator(
            prompt,
            config=config,
            prompt_overrides=prompt_overrides,
            verbose=os.environ.get("SKUNK_CONSOLE_VERBOSE", "").lower() in {"1", "true", "yes"},
        )
        try:
            try:
                answer = await orch.execute()
            except (MissingData, StepFailed) as e:
                raise RuntimeError(f"Skunk failed: {e}") from e
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


def _reasoning_summary(events: list[dict], source_docs: list[str]) -> str:
    return json.dumps(_structured_reasoning_payload(events, source_docs), ensure_ascii=False)


def _structured_reasoning_payload(events: list[dict], source_docs: list[str]) -> dict[str, Any]:
    branches_by_id: dict[int, dict[str, Any]] = {}
    branch_order: list[int] = []
    branch_steps: dict[int, list[dict[str, Any]]] = {}
    compute_code: str | None = None
    compute_attempt: int | None = None
    compute_attempts: list[dict[str, Any]] = []
    replan_count = 0
    last_plan_label = None

    for evt in events:
        kind = evt.get("kind")
        data = evt.get("data") if isinstance(evt.get("data"), dict) else {}
        message = str(evt.get("message", ""))

        if kind == "plan" and data:
            last_plan_label = data.get("label")
            if last_plan_label == "replan":
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
                match = re.match(r"codegen_code attempt=(\d+) code=(.*)", message, re.DOTALL)
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
            match = re.match(r"exec_failed attempt=(\d+) error=(.*)", message, re.DOTALL)
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
        last_step_data = last_step.get("data") if isinstance(last_step, dict) and isinstance(last_step.get("data"), dict) else {}
        branches.append(
            {
                "branch_id": branch_id,
                "kind": branch.get("kind", "branch"),
                "searched": _branch_search_details(branch),
                "output": last_step_data.get("summary") or last_step_data.get("error") or None,
                "output_step": last_step.get("op") if isinstance(last_step, dict) else None,
                "status": "failed" if last_step_data.get("error") else "ok",
            }
        )

    return {
        "summary": {
            "pipeline": last_plan_label or "plan",
            "source_docs": source_docs[:8],
            "branch_count": len(branches),
            "replan_count": replan_count,
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
            "as_of": branch.get("as_of"),
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
        for month, page in re.findall(r"PageRef\([^)]*month=([0-9]{4}-[0-9]{2})[^)]*page=(\d+)", msg):
            _append_doc(docs, seen, month, page)
        for month, page in re.findall(r"'bulletin': '([0-9]{4}-[0-9]{2})', 'page': (\d+)", msg):
            _append_doc(docs, seen, month, page)
        for month, page in re.findall(r'"bulletin": "([0-9]{4}-[0-9]{2})", "page": (\d+)', msg):
            _append_doc(docs, seen, month, page)
    return docs[:64]


def _append_doc(docs: list[str], seen: set[str], month: str, page: str) -> None:
    ref = f"Treasury Bulletin {month} PDF page {int(page)}"
    if ref not in seen:
        seen.add(ref)
        docs.append(ref)
