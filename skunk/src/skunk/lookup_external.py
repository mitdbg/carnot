"""lookup_external operator — multi-turn code-as-proof external lookup. Spins up a
per-branch `LookupAgent` whose tools fetch from FRED / BLS / World Bank / Tavily
(see `lookup_tools.py`), then translates the `{value, unit, source}` dict into
`list[AnnotatedValue]`.

The agent's tool set is pluggable: `LookupExternal.run` resolves a list of `Tool`s
(explicit override → `config.lookup_tools` → all tools) plus a prioritization
string, and the agent renders its `## Tools` prompt section (each tool's `doc`) +
prioritization guidance from them. `final_answer` is the always-injected loop
terminator and is never part of the pluggable set."""

from __future__ import annotations

from typing import Any

from skunk.extract import _cell_in_text
from skunk.lookup_tools import DEFAULT_PRIORITIZATION, resolve_lookup_tools
from skunk.common import AnnotatedValue, HarnessContext
from skunk.multi_turn_agent import MultiTurnAgent, Tool
from skunk.plan import LookupBranch
from skunk.prompted_call import PromptedCall


def _flatten_numbers(value: Any) -> list[float | int]:
    """Recursively collect primitive numbers (not bools); non-numeric leaves skipped."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, list):
        return [n for item in value for n in _flatten_numbers(item)]
    return []


class LookupAgent(MultiTurnAgent):
    _SYSTEM_PROMPT = """\
You find one external value per request — or a list of values when the
target names a series across multiple periods. Each request is a JSON
object {"target": "<value(s)>", "src": "<source | null>"}.

You have ≤8 steps. Each step, output ONE ```python``` block calling
one tool. The tool's output appears as your next observation.

{{ prioritization }}

If `src` is non-null, the answer must come from that publisher. If
null, any authoritative public source is fine.

## Tools (already imported)

{{ tools_doc }}

### final_answer(payload)
Commit. Call exactly once.
```python
final_answer({"value": 4.03, "unit": "fx_rate", "source": "measuringworth"})
final_answer({"value": [99.34, 99.21, 101.43, 99.57], "unit": "usd",
              "source": "treasurydirect"})
```

Also in scope: `math`, `statistics`, `datetime`, `numpy as np`,
`pandas as pd`, `json`.
{{ default_tail }}"""

    def __init__(self, branch: LookupBranch, max_steps: int, tools: list[Tool], prioritization: str):
        super().__init__(
            PromptedCall(
                name="lookup_external",
                system_prompt=self._SYSTEM_PROMPT.replace(
                    "{{ tools_doc }}", "\n\n".join(t.doc for t in tools)),
                default_effort="off",
                template_vars=lambda _ctx: {"prioritization": prioritization},
            ),
            tools,
        )
        self._branch = branch
        self.max_steps = max_steps

    def validate_final_answer(self, payload: dict, observations: list[str]) -> str | None:
        nums = _flatten_numbers(payload.get("value"))
        if not nums:
            return None
        joined = "\n".join(observations)
        missing = [n for n in nums if not _cell_in_text(n, joined)]
        if not missing:
            return None
        return (
            f"Grounding check failed: {missing!r} doesn't appear in any "
            f"tool output above. Either revise final_answer to match what "
            f"the tools returned verbatim, or fetch the data first."
        )


class LookupExternal:
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
        payload = agent.call(ctx, user_msg)
        return [AnnotatedValue(
            description=branch.target,
            value=payload["value"],
            unit=payload.get("unit", ""),
        )]
