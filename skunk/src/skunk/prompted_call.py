"""PromptedCall — shared prompt-assembly base for every LLM-prompted call.

Operators (`retrieve`, `extract`, `lookup_external`, `compute`) are the query-plan
nodes; a `PromptedCall` is the runtime helper that handles one LLM call inside
an operator. Each prompted call-site in the harness (planner, extract.text,
extract.vision, extract.dedup, compute.codegen, compute.critique,
lookup_external, search_agent) is one subclass of `PromptedCall`. The subclass
sets `name` (the override key used by the prompt-overrides YAML) and
`system_prompt` (a Jinja template); the base class renders it.

Rendering is one engine: Jinja. The override registry
(`prompt_overrides.yaml` → `gather_corpus / gather_few_shots / gather_lessons`)
is one source of variables that flow into the template, exposed in template
scope as `corpus` (str), `few_shots` (tuple[str, ...]), `lessons` (tuple[str, ...]),
and `default_tail` (the conventional `## Dataset` / `## Few-shot examples` /
`## Lessons learned` block, pre-rendered in Python so the layout is testable
without Jinja whitespace surprises). Subclasses that want the conventional
layout end their template with `{{ default_tail }}`; subclasses that want a
different layout (e.g. corpus before the schema section) reference the raw
variables directly. Subclasses needing runtime values (e.g. an agent's
`max_steps`) override `template_vars(ctx)` to add more entries.

Override content is passed as plain strings; the template inserts them via
`{{ corpus }}` etc. Jinja does NOT re-parse the override content, so authors
of `prompt_overrides.yaml` need never know about Jinja syntax.

Escape rule for SYSTEM templates that need a literal `{{` or `{%` (e.g. a
JSON example showing `{{nested}}`): wrap the region in
`{% raw %}…{% endraw %}`. Grep confirmed zero existing collisions across
the in-tree subclasses; the planner JSON examples use single braces only.

Adding a new conventional section to the assembled prompt = edit the
`_build_default_tail` helper in this file. Adding a new variable a subclass
can interpolate = override `template_vars(ctx)`.
"""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined

from skunk.models import HarnessContext
from skunk.prompt_overrides import (
    gather_corpus,
    gather_few_shots,
    gather_lessons,
)

# Module-level singleton. `StrictUndefined` so a typo in a `{{ var }}`
# reference fails loudly at render time instead of silently emitting "".
# `keep_trailing_newline=True` preserves the trailing `\n` that triple-quoted
# subclass templates end with — needed for byte-identical output vs. the
# previous string-concat assembly.
_ENV = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)


def _build_default_tail(
    corpus: str,
    few_shots: tuple[str, ...],
    lessons: tuple[str, ...],
) -> str:
    """Build the conventional trailing block in Python.

    Layout (each section prefixed with `\\n` so it separates cleanly from a
    SYSTEM template that ends with its own trailing newline):

        \\n## Dataset
        {corpus}
        \\n## Few-shot examples
        {shot_1}
        \\n
        {shot_2}
        \\n## Lessons learned
        - {lesson_1}
        - {lesson_2}

    Empty when no overrides target this call-site — concatenates to nothing.
    """
    parts: list[str] = []
    if corpus:
        parts.append("\n## Dataset\n" + corpus)
    if few_shots:
        parts.append("\n## Few-shot examples\n" + "\n\n".join(few_shots))
    if lessons:
        parts.append("\n## Lessons learned\n" + "\n".join(f"- {l}" for l in lessons))
    return "".join(parts)


class PromptedCall:
    """One LLM-prompted call-site. Subclasses set `name` and `system_prompt`
    (a Jinja template) as class attributes; `assemble_system_prompt(ctx)`
    renders it against the override registry (`corpus`, `few_shots`,
    `lessons`, `default_tail`) plus any extras the subclass returns from
    `template_vars(ctx)`."""

    name: str = ""
    system_prompt: str = ""

    def template_vars(self, ctx: HarnessContext) -> dict:
        """Subclass hook: return additional variables to expose to the Jinja
        SYSTEM template. Default: empty dict — subclasses without `{{ ... }}`
        markers render to themselves."""
        return {}

    def assemble_system_prompt(self, ctx: HarnessContext) -> str:
        corpus = gather_corpus(ctx.prompt_overrides, self.name)
        few_shots = gather_few_shots(ctx.prompt_overrides, self.name)
        lessons = gather_lessons(ctx.prompt_overrides, self.name)
        variables = {
            **self.template_vars(ctx),
            "corpus": corpus,
            "few_shots": few_shots,
            "lessons": lessons,
            "default_tail": _build_default_tail(corpus, few_shots, lessons),
        }
        return _ENV.from_string(self.system_prompt).render(**variables)
