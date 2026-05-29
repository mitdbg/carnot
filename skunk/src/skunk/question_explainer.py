"""question_explainer — whole-question concept extraction + explanation.

Prototype alternative to `qualifier_explainer`. Instead of iterating over
`plan.computation.qualifiers` (which misses named operations the planner
places in `task` rather than `qualifiers`), this version reads the entire
question text and extracts the non-obvious mathematical / statistical /
domain concepts a code-writing agent needs to know to answer correctly,
each with a short definition + formula.

Single LLM call per question. Output is markdown — one `### <concept>`
section per non-obvious concept — so LaTeX in the explanations doesn't
break parsing (the old JSON path choked on `\\alpha`/`\\frac`). Parsed
into `list[ConceptExplanation]` and injected into the compute codegen
prompt as the `## Concept references` section.

`qualifier_explainer.py` is kept in tree for comparison but is no longer
wired into the orchestrator.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from skunk.models import HarnessContext
from skunk.prompted_call import PromptedCall


class ConceptExplanation(BaseModel):
    """One non-obvious concept paired with its operational definition."""

    model_config = ConfigDict(frozen=True)
    concept: str
    explanation: str


# Matches a section header at the start of a line: "### <concept name>".
# Used to split the markdown reply into per-concept blocks.
_SECTION_RE = re.compile(r"^###\s+", re.MULTILINE)


class QuestionExplainerPromptedCall(PromptedCall):
    name: str = "question_explainer"
    default_effort = "low"
    system_prompt: str = """\
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
  - Include only concepts whose misimplementation would yield the
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
{{ default_tail }}"""

    def explain(self, ctx: HarnessContext, question: str) -> list[ConceptExplanation]:
        resp = self.call(ctx, f"Question:\n{question}")
        raw = resp.text.strip()
        if not raw:
            return []
        # Split on `### ` headers. First piece is the preamble before any
        # header (usually empty); drop it.
        sections = _SECTION_RE.split(raw)
        if sections and not sections[0].strip():
            sections = sections[1:]
        out: list[ConceptExplanation] = []
        for sect in sections:
            sect = sect.strip()
            if not sect:
                continue
            # First line = concept name, remainder = explanation.
            head, _, body = sect.partition("\n")
            concept = head.strip()
            explanation = body.strip()
            if concept:
                out.append(
                    ConceptExplanation(concept=concept, explanation=explanation)
                )
        if not out and raw:
            # LLM emitted prose without any `### ` headers — wrap as one
            # anonymous block so codegen still sees the content.
            out.append(
                ConceptExplanation(concept="(referenced concepts)", explanation=raw)
            )
        return out


class QuestionExplainer:
    """Operator-level wrapper: one LLM call per question, returns
    extracted concepts with explanations. Empty list when the explainer
    finds no non-obvious concepts (LLM emits empty body)."""

    def __init__(self) -> None:
        self._caller = QuestionExplainerPromptedCall()

    def run(self, ctx: HarnessContext, *, question: str) -> list[ConceptExplanation]:
        return self._caller.explain(ctx, question)
