"""extract subagent — stub."""

from __future__ import annotations

from skunk.dsl import DocHandle, OpNode, TypedValue
from skunk.subagents.base import HarnessContext, Subagent


class ExtractSubagent(Subagent):
    op_name = "extract"

    def run(
        self,
        op: OpNode,
        prev: DocHandle | None,
        ctx: HarnessContext,
    ) -> TypedValue:
        # TODO: per-page tier dispatch over prev.refs: CSV tables → OCR text → vision render.
        # mode='value' → scalar, 'list' → list, 'table' → DataFrame.
        concept = op.args.get("concept") or op.args.get("source", "?")
        mode = op.args.get("mode", "value")
        n = len(prev.refs) if isinstance(prev, DocHandle) else "?"
        print(f"[STUB extract] concept={concept!r} mode={mode!r} from {n} pages")
        return TypedValue(value=None, dtype=f"scalar:{mode}", desc=f"[stub: {concept}]")
