"""compute subagent — stub."""

from __future__ import annotations

from skunk.dsl import OpNode, TypedValue
from skunk.subagents.base import HarnessContext, Subagent


class ComputeSubagent(Subagent):
    op_name = "compute"

    def run(
        self,
        op: OpNode,
        prev: object,
        ctx: HarnessContext,
    ) -> TypedValue:
        # TODO: LLM writes Python body; sandboxed exec with numpy/pandas/statsmodels.
        code = op.args.get("code") or op.args.get("formula") or op.args.get("source", "?")
        preview = str(code)[:60]
        prev_desc = repr(prev)[:80]
        print(f"[STUB compute] code={preview!r} prev={prev_desc}")
        return TypedValue(value=None, dtype="computed", desc="[stub compute]")
