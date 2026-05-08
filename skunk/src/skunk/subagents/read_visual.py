"""read_visual subagent — stub."""

from __future__ import annotations

from skunk.dsl import DocHandle, OpNode, TypedValue
from skunk.subagents.base import HarnessContext, Subagent


class ReadVisualSubagent(Subagent):
    op_name = "read_visual"

    def run(
        self,
        op: OpNode,
        prev: DocHandle | None,
        ctx: HarnessContext,
    ) -> TypedValue:
        # TODO: always send to vision LLM.
        concept = op.args.get("concept") or op.args.get("source", "?")
        n = len(prev.refs) if isinstance(prev, DocHandle) else "?"
        print(f"[STUB read_visual] concept={concept!r} from {n} pages")
        return TypedValue(value=None, dtype="visual", desc=f"[stub: {concept}]")
