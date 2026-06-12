from __future__ import annotations

from skunk.lookup_tools import DEFAULT_PRIORITIZATION, resolve_lookup_tools
from skunk.common import AnnotatedValue, ExecutionContext
from skunk.multi_turn_agent import MultiTurnAgent, Tool
from skunk.plan import LookupBranch


class LookupAgent(MultiTurnAgent):
    name = "lookup_external"
    warn_steps_remaining = 1

    briefing = (
        "You find the external value(s) a request asks for and commit them as one "
        'result. Each request is a JSON object {"target": "<value(s)>", '
        '"src": "<source | null>"}. If `src` is non-null, the answer must come from '
        "that publisher; if null, any authoritative public source is fine.\n\n"
        + DEFAULT_PRIORITIZATION
    )

    final_answer_doc = """\
A JSON object with these keys (literals only — copy values out of your
observations; you cannot reference variables here):

  {"description": "<names the value + its source>", "value": <...>,
   "unit": "<e.g. pct, usd, fx_rate>",
   "kind": "scalar" | "vector" | "table",
   "index_name": "<dim>",                      # vector only
   "row_name": "<dim>", "col_name": "<dim>"}   # table only

Use `kind="vector"` — `value` a dict keyed by the period/label, with
`index_name` — whenever the answer is a series the downstream step will rank,
select, or aggregate (e.g. "which year did X peak"). Use a plain scalar (or a
list) only when the labels don't matter. Cells must be primitive (no nesting).
Fold the source into `description`. For a long series, `print(json.dumps(...))`
the dict in a tool step first, then copy the printed JSON here.
```json
{"description": "USD to GBP spot rate, 2002-06-30 (MeasuringWorth)",
 "value": 0.6549, "unit": "fx_rate"}
```
```json
{"description": "U.S. personal saving rate, 1950-1990 (FRED PSAVERT)",
 "kind": "vector", "index_name": "year",
 "value": {"1950": 9.4, "1951": 11.1, "1990": 8.5}, "unit": "pct"}
```

Tool steps have `math`, `statistics`, `datetime`, `numpy as np`,
`pandas as pd`, `json` in scope (plus the tools above)."""

    def __init__(self, *, max_steps: int, tools: list[Tool]):
        super().__init__(tools, max_steps=max_steps)

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        # Shape only — no numeric-grounding check: the agent reaches every value
        # through a tool call, so tool use is itself the proof of grounding.
        # `AnnotatedValue`'s own pydantic validator IS the shape gate: build it
        # and surface any error back to the agent as feedback.
        if not isinstance(payload, dict):
            return (
                "Emit a JSON object of AnnotatedValue fields, e.g. "
                '{"description": ..., "value": ..., "unit": ...}.'
            )
        try:
            AnnotatedValue.model_validate(payload)
        except Exception as e:
            return f"final-answer JSON is not a valid AnnotatedValue: {e}"
        return None


class LookupExternalOp:
    async def run(self, ctx: ExecutionContext, branch: LookupBranch) -> list[AnnotatedValue]:
        agent = LookupAgent(
            max_steps=ctx.config.lookup_max_steps,
            tools=resolve_lookup_tools(ctx.config),
        )
        # Cap each agent turn like SearchAgent/SelectAgent: an uncapped lookup turn
        # hung 250s+ (thinking-only generation) and stalled the whole question.
        agent.max_output_tokens = ctx.config.select_agent_max_output_tokens
        agent.request_timeout_s = ctx.config.select_agent_request_timeout_s
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True,
        )
        # The agent commits a dict of `AnnotatedValue` fields (shape-gated by
        # `validate_final_answer`); build the typed object on the trusted side.
        payload = await agent.call(ctx, user_msg)
        return [AnnotatedValue.model_validate(payload)]
