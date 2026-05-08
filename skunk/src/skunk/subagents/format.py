"""format subagent — stub."""

from __future__ import annotations

from skunk.dsl import FormattedString, OpNode
from skunk.subagents.base import HarnessContext, Subagent


class FormatSubagent(Subagent):
    op_name = "format"

    def run(
        self,
        op: OpNode,
        prev: object,
        ctx: HarnessContext,
    ) -> FormattedString:
        # TODO: deterministic application of precision/unit/layout kwargs;
        print(f"[STUB format] args={op.args} prev={repr(prev)[:80]}")
        return FormattedString(text="[stub answer]", desc="[stub format]")
