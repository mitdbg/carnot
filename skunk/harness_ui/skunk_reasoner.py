"""Competition-console adapter for the Skunk OfficeQA reasoner."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from cup_kit.agent_runtime import AgentAnswer

HERE = Path(__file__).resolve().parent
SKUNK_ROOT = HERE.parent
REPO_ROOT = SKUNK_ROOT.parent
SKUNK_SRC = SKUNK_ROOT / "src"

if str(SKUNK_SRC) not in sys.path:
    sys.path.insert(0, str(SKUNK_SRC))


def _set_default_env() -> None:
    os.environ.setdefault(
        "OFFICEQA_PARSED_JSON_DIR",
        str(REPO_ROOT / "data/officeqa/treasury_bulletins_parsed/jsons"),
    )
    os.environ.setdefault(
        "OFFICEQA_PDF_DIR",
        str(REPO_ROOT / "data/officeqa/treasury_bulletin_pdfs"),
    )


async def solve(prompt: str) -> AgentAnswer:
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
        source_docs = _source_docs_from_events(orch.ctx.events)
        return AgentAnswer(
            answer=answer,
            reasoning=_reasoning_summary(orch.ctx.events, source_docs),
            source_docs=source_docs,
        )
    finally:
        orch.ctx.close()


def _reasoning_summary(events: list[dict], source_docs: list[str]) -> str:
    steps = [
        f"{evt.get('op')}: {evt.get('message')}"
        for evt in events
        if evt.get("message", "").startswith("step ")
    ]
    source_text = ", ".join(source_docs[:8]) if source_docs else "no page citations parsed from trace"
    summary = (
        "Skunk ran the OfficeQA pipeline: planner, retrieval/external lookup branches, "
        "extraction, and compute. Source documents: "
        f"{source_text}. Step summary: "
        + " | ".join(steps[-8:])
    )
    return summary[:200_000]


def _source_docs_from_events(events: list[dict]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for evt in events:
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
