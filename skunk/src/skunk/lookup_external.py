"""lookup_external operator — multi-turn code-as-proof external lookup.

`LookupExternalPromptedCall.run(ctx, branch)` is the orchestrator-facing
entry. It spins up a per-branch `LookupAgent` (a `MultiTurnAgent`)
whose tools fetch from FRED / BLS / World Bank / Tavily, runs the
loop, and translates the `{value, unit, source}` dict into the
`list[AnnotatedValue]` shape compute expects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from skunk.extract import _cell_in_text
from skunk.lookup_helpers import (
    fetch_bls, fetch_fred, fetch_url, fetch_world_bank, tavily_search,
)
from skunk.models import AnnotatedValue, HarnessContext
from skunk.multi_turn_agent import MultiTurnAgent, final_answer
from skunk.plan import LookupBranch


def _flatten_numbers(value: Any) -> list[float | int]:
    """Recursively collect primitive numbers (ints/floats, not bools).
    Non-numeric leaves are silently skipped — grounding is numeric-only."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, list):
        return [n for item in value for n in _flatten_numbers(item)]
    return []


class LookupAgent(MultiTurnAgent):
    name: str = "lookup_external"
    system_prompt: str = """\
You find one data point from an external source per request. Each
request is a JSON object {"target": "<value>", "src": "<source | null>"}.

You have ≤8 steps. Each step, output ONE ```python``` block calling
one tool. The tool's output appears as your next observation.

When you see a number that plausibly answers the target in some tool
output, your VERY NEXT block should be final_answer. Many historical
lookups have no single canonical precision — multiple sources may
report slightly different values. Pick the first plausible hit and
commit. Searching for confirmation is the dominant failure mode.

If `src` is non-null, the answer must come from that publisher. If
null, any authoritative public source is fine.

## Tools (already imported)

### fetch_fred(series_id, date)
FRED API. `date`: YYYY (annual mean), YYYY-MM, YYYY-MM-DD.
```python
val = fetch_fred("CPIAUCSL", "1953")     # annual mean of CPI in 1953
val = fetch_fred("DGS10", "2020-03-15")  # 10y yield near date
```

### fetch_bls(series_id, date)
BLS API. Same date format.
```python
val = fetch_bls("CUUR0000SA0", "1953")   # CPI-U 1953 annual
```

### fetch_world_bank(iso3, indicator, year)
Annual indicators.
```python
val = fetch_world_bank("DEU", "NY.GDP.MKTP.CD", "1996")  # Germany nominal GDP
```

### tavily_search(query, max_results=5)
Web search. Returns [{title, url, content, score}]. The `content`
snippet often carries the number directly.
```python
hits = tavily_search("annual average GBP USD exchange rate 1941")
```

### fetch_url(url)
Fetch and return cleaned page text. Soft-fails on errors.
```python
text = fetch_url("https://example.com/historical-rates")
```

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

    def __init__(self, branch: LookupBranch, max_steps: int = 8):
        self._branch = branch
        self.max_steps = max_steps

    def tools(self) -> dict[str, Callable]:
        return {
            "fetch_fred": fetch_fred,
            "fetch_bls": fetch_bls,
            "fetch_world_bank": fetch_world_bank,
            "tavily_search": tavily_search,
            "fetch_url": fetch_url,
            "final_answer": final_answer,
        }

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


class LookupExternalPromptedCall:
    name: str = "lookup_external"

    def run(self, ctx: HarnessContext, branch: LookupBranch) -> list[AnnotatedValue]:
        agent = LookupAgent(branch, max_steps=ctx.config.lookup_max_steps)
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True,
        )
        payload = agent.call(ctx, user_msg)
        return [AnnotatedValue(
            description=branch.target,
            value=payload["value"],
            unit=payload.get("unit", ""),
        )]
