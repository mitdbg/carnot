"""lookup_external subagent — stub."""

from __future__ import annotations

from skunk.dsl import OpNode, TypedValue
from skunk.subagents.base import HarnessContext, Subagent


class LookupExternalSubagent(Subagent):
    op_name = "lookup_external"

    def run(
        self,
        op: OpNode,
        prev: None,
        ctx: HarnessContext,
    ) -> TypedValue:
        # TODO: cache-first CSV read for CPI-U, FX, BLS; live API fallback when cache_only=False.
        # event_year/event_date resources for knowledge-bound period questions.
        resource = op.args.get("resource") or op.args.get("source", "?")
        print(f"[STUB lookup_external] resource={resource!r} args={op.args}")
        return TypedValue(value=None, dtype="external", desc=f"[stub: {resource}]")
