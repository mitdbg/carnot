"""SkunkExecutor — shared prompt-assembly base for every LLM-prompted call.

Operators (`retrieve`, `extract`, `lookup_external`, `compute`) are the query-plan
nodes; an Executor is the runtime helper that handles one LLM call inside an
operator. Each prompted call-site in the harness (planner, extract.text,
extract.vision, extract.dedup, compute.codegen, compute.critique,
lookup_external) is one subclass of `SkunkExecutor`. The subclass sets two
attributes — `name` (the override key used by the prompt-overrides YAML) and
`system_prompt` (the static SYSTEM template) — and the base class composes
the final prompt.

Final system prompt shape:

    [SYSTEM]      → subclass `system_prompt` (or `static_system_prompt(ctx)` when runtime-conditional)
    [CORPUS]      → `corpus` overrides addressed to this executor
    [FEW-SHOTS]   → `few_shots` overrides, joined verbatim (YAML pre-formats them)
    [LESSONS]     → `lessons` overrides addressed to this executor

Adding a new section to the executor prompt = edit this one file.
"""

from __future__ import annotations

from skunk.models import HarnessContext
from skunk.prompt_overrides import (
    gather_corpus,
    gather_few_shots,
    gather_lessons,
    render_corpus_block,
    render_lessons_block,
)


class SkunkExecutor:
    """One LLM-prompted executor. Subclasses set `name` and `system_prompt` as
    class attributes; `assemble_system_prompt(ctx)` composes [SYSTEM] [CORPUS]
    [FEW-SHOTS] [LESSONS] from `ctx.prompt_overrides`. Override
    `static_system_prompt(ctx)` when the static SYSTEM block itself is
    conditional at runtime."""

    name: str = ""
    system_prompt: str = ""

    def static_system_prompt(self, ctx: HarnessContext) -> str:
        """Return the static SYSTEM block. Override when the block has
        runtime-conditional content. Default: return `self.system_prompt`."""
        return self.system_prompt

    def assemble_system_prompt(self, ctx: HarnessContext) -> str:
        parts: list[str] = [self.static_system_prompt(ctx)]

        corpus = render_corpus_block(gather_corpus(ctx.prompt_overrides, self.name))
        if corpus:
            parts.append("\n" + corpus)

        shots = gather_few_shots(ctx.prompt_overrides, self.name)
        if shots:
            parts.append("\n## Few-shot examples\n" + "\n\n".join(shots))

        lessons = render_lessons_block(gather_lessons(ctx.prompt_overrides, self.name))
        if lessons:
            parts.append("\n" + lessons)

        return "".join(parts)
