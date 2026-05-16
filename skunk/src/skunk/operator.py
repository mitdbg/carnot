"""SkunkOperator — shared prompt-assembly base for every LLM-prompted operator.

Each prompted call-site in the harness (planner, extract.text, extract.vision,
extract.dedup, compute.codegen.final, compute.codegen.intermediate,
compute.critique, lookup_external) is one subclass of `SkunkOperator`. The
subclass sets two attributes — `name` (the override key used by the
prompt-overrides YAML) and `system` (the static SYSTEM template) — and the
base class composes the final prompt.

Final system prompt shape:

    [SYSTEM]      → subclass `system` (or `system_text(ctx)` when runtime-conditional)
    [CORPUS]      → `corpus` overrides addressed to this operator
    [FEW-SHOTS]   → `few_shots` overrides, joined verbatim (YAML pre-formats them)
    [LESSONS]     → `lessons` overrides addressed to this operator

Adding a new section to the operator prompt = edit this one file.
"""

from __future__ import annotations

from skunk.common import HarnessContext
from skunk.prompt_overrides import (
    gather_corpus,
    gather_few_shots,
    gather_lessons,
    render_corpus_block,
    render_lessons_block,
)


class SkunkOperator:
    """One LLM-prompted operator. Subclasses set `name` and `system` as class
    attributes; `build_system(ctx)` composes [SYSTEM] [CORPUS] [FEW-SHOTS]
    [LESSONS] from `ctx.prompt_overrides`. Override `system_text(ctx)` when
    the static SYSTEM block itself is conditional at runtime."""

    name: str = ""
    system: str = ""

    def system_text(self, ctx: HarnessContext) -> str:
        """Return the static SYSTEM block. Override when the block has
        runtime-conditional content. Default: return `self.system`."""
        return self.system

    def build_system(self, ctx: HarnessContext) -> str:
        parts: list[str] = [self.system_text(ctx)]

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
