from __future__ import annotations

from skunk.lookup_agent.lookup_tools import DEFAULT_PRIORITIZATION, resolve_lookup_tools
from skunk.common import AnnotatedValue, ExecutionContext
from skunk.config import LookupAgentConfig
from skunk.multi_turn_agent import MultiTurnAgent, Tool
from skunk.plan import LookupBranch


class LookupAgent(MultiTurnAgent):
    name = "lookup_external"
    warn_steps_remaining = 1

    briefing = '''\
You find the external value(s) a request asks for and commit them as one result. Each request is a JSON object {"target": "<value(s)>", "src": "<source | null>"}.

`src` sets where the value may come from and how hard you search:
- Pinned (non-null): the value MUST come from that publisher. Keep searching until you read it from that named source; never substitute another publisher's figure, even one offered first — same-named series (e.g. a BLS/FRED series vs the pinned publisher) can differ, and that difference is the point.
- Null: any authoritative public source is fine. Commit the first hit you have verified answers the target — historical values vary slightly across sources, so don't keep searching for confirmation.

Before committing, verify the value matches the target's period/date, unit, and geography — a number for the wrong year or basis is the main failure here. For a multi-value target (a series/range or a row×column breakdown) collect the whole set, not the first cell, and emit it as a vector/table.

Tool steps have `math`, `statistics`, `datetime`, `numpy as np`, `pandas as pd`, `json` in scope (plus the tools above).

''' + DEFAULT_PRIORITIZATION

    final_answer_doc = '''\
A JSON object with these keys (literals only — copy values out of your
observations; you cannot reference variables here):

  {"description": "<names the value>", "value": <...>,
   "unit": "<free-form, e.g. percent, millions of dollars, yen per U.S. dollar>",
   "source": "<publisher/origin of the value>",
   "kind": "scalar" | "vector" | "table",
   "index_name": "<dim>",                      # vector only
   "row_name": "<dim>", "col_name": "<dim>"}   # table only

Pick the shape that best fits the requested data. Cells should be simple number or string — no nested cells.
Put the value's publisher/origin in `source`. Examples:
```json
{"description": "USD to GBP spot rate, 2002-06-30 (British pounds per U.S. dollar)",
 "value": 0.6549, "unit": "British pounds per U.S. dollar", "source": "MeasuringWorth"}
```
```json
{"description": "U.S. personal saving rate, 1950-1990",
 "kind": "vector", "index_name": "year", "source": "FRED PSAVERT",
 "value": {"1950": 9.4, "1951": 11.1, "1990": 8.5}, "unit": "percent"}
```
'''

    def __init__(self, config: LookupAgentConfig, *, max_steps: int, tools: list[Tool]):
        super().__init__(tools, max_steps=max_steps, cost_budget=config.cost_budget, latency_budget=config.latency_budget)

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
        cfg = ctx.config.lookup
        tools = resolve_lookup_tools(cfg)
        # LookupAgent derives its cost/latency budgets from the config it is handed.
        agent = LookupAgent(
            cfg,
            max_steps=cfg.lookup_max_steps,
            tools=tools,
        )
        # Cap each agent turn like SearchAgent: an uncapped lookup turn
        # hung 250s+ (thinking-only generation) and stalled the whole question.
        agent.max_output_tokens = cfg.lookup_agent_max_output_tokens
        agent.request_timeout_s = cfg.lookup_agent_request_timeout_s
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True,
        )
        # The agent commits a dict of `AnnotatedValue` fields (shape-gated by
        # `validate_final_answer`); build the typed object on the trusted side.
        payload = await agent.call(ctx, user_msg)
        return [AnnotatedValue.model_validate(payload)]
