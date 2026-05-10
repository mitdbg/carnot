"""retrieve subagent — stub."""

from __future__ import annotations

from skunk.dsl import DocHandle, OpNode
from skunk.subagents.base import HarnessContext


def run(op: OpNode, prev: None, ctx: HarnessContext) -> DocHandle:
    # TODO: build page index over corpus; map (concept, period) → ranked PageRef list.
    if ctx.golden_handle is not None:
        print(f"[STUB retrieve] golden bypass → {len(ctx.golden_handle.refs)} pages")
        return ctx.golden_handle
    concept = op.args.get("concept") or op.args.get("source", "?")
    period = op.args.get("period", "?")
    print(f"[STUB retrieve] concept={concept!r} period={period!r}")
    return DocHandle(refs=[], desc=f"[stub: {concept} / {period}]")
