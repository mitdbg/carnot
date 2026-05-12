"""retrieve subagent — stub."""

from __future__ import annotations

from skunk.common import HarnessContext
from skunk.dsl import DocHandle, OpNode
from skunk.subagents.base import StepFailed


def run(op: OpNode, prev: None, ctx: HarnessContext) -> DocHandle:
    # TODO: build page index over corpus; map (concept, period) → ranked PageRef list.
    if ctx.config.golden_handle is not None:
        ctx.emit("retrieve", "golden bypass",
                 n_pages=len(ctx.config.golden_handle.refs),
                 refs=[str(r) for r in ctx.config.golden_handle.refs])
        return ctx.config.golden_handle
    concept = op.args.get("concept") or op.args.get("source", "?")
    period = op.args.get("period", "?")
    raise StepFailed(
        "retrieve",
        f"[STUB] no page index built — cannot retrieve concept={concept!r} period={period!r}. "
        "Provide --golden or implement the page index.",
    )
