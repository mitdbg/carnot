"""extract subagent — stub."""

from __future__ import annotations

from skunk.dsl import DocHandle, OpNode, TypedValue
from skunk.subagents.base import HarnessContext, StepFailed


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    # TODO: per-page tier dispatch over prev.refs: CSV tables → OCR text → vision render.
    concept = op.args.get("concept") or op.args.get("source", "?")
    mode = op.args.get("mode", "value")
    n = len(prev.refs) if isinstance(prev, DocHandle) else "?"
    raise StepFailed("extract", f"[stub] not implemented: concept={concept!r} mode={mode!r} pages={n}")
