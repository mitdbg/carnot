"""read_visual subagent — vision-only extraction; delegates to extract with visual_only=True."""

from __future__ import annotations

import skunk.subagents.extract as _extract
from skunk.common.context import HarnessContext
from skunk.dsl import DocHandle, OpNode, TypedValue


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    return _extract.run(OpNode(op="extract", args={**op.args, "visual_only": True}), prev, ctx)
