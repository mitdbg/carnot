"""read_visual subagent — vision-only extraction; delegates to extract with visual_only=True."""

from __future__ import annotations

import skunk.subagents.extract as _extract
from skunk.common.context import HarnessContext
from skunk.dsl import DocHandle, OpNode, TypedValue


def run(op: OpNode, prev: DocHandle | None, ctx: HarnessContext) -> TypedValue:
    # read_visual takes no args; force the extract path through Tier 3 (vision).
    return _extract.run(OpNode(op="extract", args={"visual_only": True}), prev, ctx)
