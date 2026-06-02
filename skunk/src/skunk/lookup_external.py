"""lookup_external operator — multi-turn code-as-proof external lookup. Spins up a
per-branch `LookupAgent` whose tools fetch from FRED / BLS / World Bank / Tavily
(see `lookup_tools.py`), then translates the `{value, unit, source}` dict into
`list[AnnotatedValue]`.

The agent's tool set is pluggable: `LookupExternalOp.run` resolves a list of `Tool`s
(explicit override → `config.lookup_tools` → all tools) plus a prioritization
string, and the agent renders its `## Tools` prompt section (each tool's `doc`) +
prioritization guidance from them. `final_answer` is the always-injected loop
terminator and is never part of the pluggable set."""

from __future__ import annotations

from skunk.extract import _cell_in_text, _cells_with_path
from skunk.lookup_tools import DEFAULT_PRIORITIZATION, resolve_lookup_tools
from skunk.common import AnnotatedValue, HarnessContext
from skunk.multi_turn_agent import MultiTurnAgent, Tool, render_tools_into
from skunk.plan import LookupBranch
from skunk.prompted_call import PromptedCall


class LookupAgent(MultiTurnAgent):
    _SYSTEM_PROMPT = """\
You find the external value(s) a request asks for and commit them as one
`AnnotatedValue`. Each request is a JSON object
{"target": "<value(s)>", "src": "<source | null>"}.

You have ≤8 steps. Each step, output ONE ```python``` block calling
one tool. The tool's output appears as your next observation.

{{ prioritization }}

If `src` is non-null, the answer must come from that publisher. If
null, any authoritative public source is fine.

## Tools (already imported)

{{ tools_doc }}

### final_answer(result)
Build your answer as ONE `AnnotatedValue` and commit it (call exactly once):

  AnnotatedValue(description="<names the value + its source>", value=<...>,
                 unit="<e.g. pct, usd, fx_rate>",
                 kind="scalar" | "vector" | "table",
                 index_name="<dim>",                  # vector only
                 row_name="<dim>", col_name="<dim>")  # table only

Use `kind="vector"` — `value` a dict keyed by the period/label, with
`index_name` — whenever the answer is a series the downstream step will rank,
select, or aggregate (e.g. "which year did X peak"). Use a plain scalar (or a
list) only when the labels don't matter. Cells must be primitive (no nesting).
Fold the source into `description`.
```python
final_answer(AnnotatedValue(
    description="USD to GBP spot rate, 2002-06-30 (MeasuringWorth)",
    value=0.6549, unit="fx_rate"))
final_answer(AnnotatedValue(
    description="U.S. personal saving rate, 1950-1990 (FRED PSAVERT)",
    kind="vector", index_name="year",
    value={"1950": 9.4, "1951": 11.1, "1990": 8.5}, unit="pct"))
```

`AnnotatedValue` is in scope (like the tools). Also available: `math`,
`statistics`, `datetime`, `numpy as np`, `pandas as pd`, `json`.
{{ default_tail }}"""

    def __init__(self, branch: LookupBranch, max_steps: int, tools: list[Tool], prioritization: str):
        super().__init__(
            PromptedCall(
                name="lookup_external",
                system_prompt=render_tools_into(self._SYSTEM_PROMPT, tools),
                default_effort="off",
                template_vars=lambda _ctx: {"prioritization": prioritization},
            ),
            tools,
        )
        self._branch = branch
        self.max_steps = max_steps

    def tools(self) -> dict:
        """Bind `AnnotatedValue` into the executor namespace (alongside the tools
        and `final_answer`) so the agent can construct its result object directly."""
        return super().tools() | {"AnnotatedValue": AnnotatedValue}

    def validate_final_answer(self, payload: object, observations: list[str]) -> str | None:
        if not isinstance(payload, AnnotatedValue):
            return (
                "Call final_answer with a constructed AnnotatedValue(...), e.g. "
                "final_answer(AnnotatedValue(description=..., value=..., unit=...))."
            )
        # Grounding: every numeric cell must appear verbatim in some tool output
        # (shared with extract's per-cell verifier; covers scalar/vector/table).
        joined = "\n".join(observations)
        missing = [
            c for _, c in _cells_with_path(payload)
            if isinstance(c, (int, float)) and not isinstance(c, bool)
            and not _cell_in_text(c, joined)
        ]
        if not missing:
            return None
        return (
            f"Grounding check failed: {missing!r} doesn't appear in any tool "
            f"output above. Either fix the value(s) to match what the tools "
            f"returned verbatim, or fetch the data first."
        )


class LookupExternalOp:
    def run(self, ctx: HarnessContext, branch: LookupBranch,
            tools: list[Tool] | None = None, prioritization: str | None = None) -> list[AnnotatedValue]:
        tools = resolve_lookup_tools(ctx.config, tools)
        agent = LookupAgent(
            branch, max_steps=ctx.config.lookup_max_steps,
            tools=tools, prioritization=prioritization or DEFAULT_PRIORITIZATION,
        )
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True,
        )
        # The agent commits a constructed `AnnotatedValue` (enforced by
        # `validate_final_answer`), so return it directly.
        return [agent.call(ctx, user_msg)]
