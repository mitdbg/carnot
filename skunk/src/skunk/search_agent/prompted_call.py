"""make_search_agent_prompt — the search agent's system prompt, packaged as
a `PromptedCall`.

This module owns the Jinja template that used to live in
`src/skunk/search_agent/prompts.yaml` and was rendered by hand in
`SearchAgent.__init__()`. Routing it through `PromptedCall` means:

- `{{ max_steps }}` / `{{ max_pages }}` are supplied by the
  `template_vars` provider passed to the `PromptedCall`, pulled from
  `ctx.config.agent_max_steps` / `ctx.config.agent_max_pages_per_tool_call`.
- The trailing `{{ default_tail }}` picks up any corpus / few-shot /
  lessons overrides addressed to `search_agent` (or `"*"`) from
  `prompt_overrides.yaml` — same mechanism as every other call-site.

The dataset blurb that the abandoned `{{ special_notes }}` placeholder
in the old YAML was reaching for now lives in `prompt_overrides.yaml`
as `section: corpus, targets: [search_agent]` and renders under the
`## Dataset` header at the end of the SYSTEM block.
"""

from __future__ import annotations

from collections.abc import Iterable

from skunk.common import HarnessContext
from skunk.multi_turn_agent import Tool, render_tools_into
from skunk.prompted_call import PromptedCall


_SEARCH_AGENT_SYSTEM_PROMPT = """\
You are a helpful assistant for retrieving relevant information from a large collection of files. You will be given a question, and your task is to identify which files are relevant to answering the question. You will have {{ max_steps }} steps to retrieve relevant information, so you do not need to retrieve everything in a single tool call. Instead, use early steps to explore files of potential relevance, and then refine your searches in later steps based on what you find. Some questions require looking up information that is not contained in the files; this will be handled by a separate agent, so you should only focus on retrieving relevant files for the remainder of the question.

## Response format
On every step, output exactly ONE fenced python code block of the form:

```python
# your code here
```

The code block must contain a single call to one of the tools listed below. Do not output any other code blocks, and do not output multiple tool calls in a single step. You may write a brief "Thoughts:" line before the code block.

## Tools
In each step, you must invoke one of the following tools:

{{ tools_doc }}

### final_answer(payload: dict)
This tool should be called when you have retrieved all relevant information and are ready to provide a final answer. Pass a JSON dict with the page keys you identified as relevant, in the format "year_month_page_id". Call this exactly once per question, and only when you are confident. Be sure to use the page_id that corresponds to the index of the page in the bulletin, not the page number printed on the page itself.

Schema:
```python
final_answer({"page_keys": ["2002_06_1", "2002_12_26"]})
```

## Task
You will now be presented with the question and must execute your search before responding with a final list of relevant page keys in the format "year_month_page_id". Remember to use the tools iteratively to refine your search results, and only call final_answer when you are confident that you have identified all relevant information.{{ default_tail }}"""


def _search_agent_vars(ctx: HarnessContext) -> dict:
    return {
        "max_steps": ctx.config.agent_max_steps,
        "max_pages": ctx.config.agent_max_pages_per_tool_call,
    }


def make_search_agent_prompt(tools: Iterable[Tool]) -> PromptedCall:
    """Build the search-agent system-prompt `PromptedCall`. The `## Tools` section
    is generated from each tool's `doc` (spliced into the `{{ tools_doc }}` marker);
    runtime vars `max_steps` / `max_pages` are read from `ctx.config` at render time
    (and any `{{ max_pages }}` inside a tool doc renders in that same pass)."""
    return PromptedCall(
        name="search_agent",
        system_prompt=render_tools_into(_SEARCH_AGENT_SYSTEM_PROMPT, list(tools)),
        template_vars=_search_agent_vars,
    )
