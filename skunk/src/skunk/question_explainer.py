"""question_explainer — whole-question concept extraction. One LLM call reads the
question and extracts the non-obvious concepts a code-writing agent needs (each with a
short definition), injected into the compute codegen prompt as `## Concept references`.

Output is markdown (one `### <concept>` section each) so LaTeX doesn't break parsing."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from skunk.common import ExecutionContext
from skunk.prompted_call import PromptedCall


class ConceptExplanation(BaseModel):
    """One non-obvious concept paired with its operational definition."""

    model_config = ConfigDict(frozen=True)
    concept: str
    explanation: str


# Splits the markdown reply on `### <concept>` headers.
_SECTION_RE = re.compile(r"^###\s+", re.MULTILINE)


_QUESTION_SYSTEM_PROMPT = """\
Read the question. Identify any non-obvious mathematical, statistical,
or domain concepts a code-writing agent must know to answer correctly.
For each, give a short explanation — only the information the agent
needs to apply the concept to THIS question.

Hard constraints:
  - Cover ONE formulation per concept — the one the question implies.
    Do NOT introduce alternative formulas, alternative orientations,
    or variants the question doesn't mention. If the question doesn't
    pin a choice, pick the most standard form and stop there. No
    "alternatively…" / "another approach…" / "or you could…".
  - Skip trivial operations (sum, average, max, ratio, difference).
    Skip pure unit / scope modifiers ("in millions of dollars",
    "for FY 2023").
  - Include concepts whose misimplementation would yield the
    wrong answer — named statistical operations (geometric mean,
    Zipf exponent, arc elasticity, expected shortfall, Gini,
    H-spread, CAGR, linear regression, etc.), domain-specific
    conventions, or any phrase whose canonical meaning a non-expert
    might guess wrong.

Output format: one markdown section per concept. Each section starts
with `### <Concept Name>` on its own line, followed by a 1-3 sentence
explanation. Plain text or LaTeX is fine — write naturally, no JSON
escapes. Separate sections with a blank line.

If no non-obvious concepts apply, output nothing.
"""


def _parse_concepts(raw: str, ctx: ExecutionContext) -> list[ConceptExplanation]:
    """Parse the markdown reply into per-`### ` concept sections. Empty body →
    `[]`; prose without headers → one anonymous block so codegen still sees it."""
    raw = raw.strip()
    if not raw:
        return []
    sections = _SECTION_RE.split(raw)
    if sections and not sections[0].strip():  # drop the pre-header preamble
        sections = sections[1:]
    out: list[ConceptExplanation] = []
    for sect in sections:
        sect = sect.strip()
        if not sect:
            continue
        head, _, body = sect.partition("\n")  # first line = name, rest = explanation
        concept = head.strip()
        explanation = body.strip()
        if concept:
            out.append(ConceptExplanation(concept=concept, explanation=explanation))
    if not out and raw:
        # Prose with no `### ` headers — wrap as one anonymous block.
        out.append(ConceptExplanation(concept="(referenced concepts)", explanation=raw))
    return out


class QuestionExplainer:
    """One LLM call per question → extracted concepts (empty list when none apply)."""

    _prompt = PromptedCall(
        name="question_explainer",
        system_prompt=_QUESTION_SYSTEM_PROMPT,
        default_effort="low",
        parse=_parse_concepts,
        output_instruction=(
            "Output one `### <Concept>` markdown section per concept "
            "(1–3 sentences each), or nothing if only obvious concepts apply."
        ),
    )

    async def run(self, ctx: ExecutionContext, *, question: str) -> list[ConceptExplanation]:
        return await self._prompt.call(ctx, f"Question:\n{question}")
