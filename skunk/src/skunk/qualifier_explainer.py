"""qualifier_explainer — expand each `Computation.qualifiers` phrase into
an actionable operationalization blurb for the codegen agent.

After the planner emits a `Plan`, this step walks `plan.computation.qualifiers`
in parallel (one LLM call per qualifier) and produces, for each, a 1-4 line
"how to apply this" blurb — formula for named operations, conversion
procedure for unit modifiers, slicing rule for temporal scopes, etc. The
blurbs are threaded into the compute codegen prompt so the code-writing
agent has a grounded reference for every constraint rather than relying
on first-pass intuition (a prominent failure mode for named statistical
operations).

Best-effort: a single failed explain call is dropped, not fatal — the
rest of the qualifiers still get their explanations through.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, ConfigDict

from skunk.models import HarnessContext
from skunk.plan import Plan
from skunk.prompted_call import PromptedCall


class QualifierExplanation(BaseModel):
    """One qualifier paired with its operationalization blurb."""

    model_config = ConfigDict(frozen=True)
    qualifier: str
    explanation: str


class QualifierExplainerPromptedCall(PromptedCall):
    name: str = "qualifier_explainer"
    default_effort = "low"
    system_prompt: str = """\
Define the given phrase in one short paragraph. Include the formula
when one exists. No preamble.
{{ default_tail }}"""

    def explain(self, ctx: HarnessContext, phrase: str) -> QualifierExplanation:
        resp = self.call(ctx, f"Qualifier: {phrase}")
        return QualifierExplanation(qualifier=phrase, explanation=resp.text.strip())


class QualifierExplainer:
    """Operator-level wrapper. Parallelizes one explain call per qualifier
    using the shared `config.max_parallel_workers` budget. Order-preserving:
    output list aligns with `plan.computation.qualifiers` order so the
    codegen-prompt block reads in the same order as the computation JSON
    dump (easier to cross-reference in traces). A single failed call is
    dropped, not fatal."""

    def __init__(self) -> None:
        self._caller = QualifierExplainerPromptedCall()

    def run(self, ctx: HarnessContext, *, plan: Plan) -> list[QualifierExplanation]:
        qualifiers = plan.computation.qualifiers
        if not qualifiers:
            return []
        results: list[QualifierExplanation | None] = [None] * len(qualifiers)
        with ThreadPoolExecutor(
            max_workers=ctx.config.max_parallel_workers
        ) as pool:
            futures = {
                pool.submit(self._caller.explain, ctx, q): i
                for i, q in enumerate(qualifiers)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    ctx.emit(
                        "qualifier_explainer",
                        "explain failed; dropping qualifier",
                        qualifier=qualifiers[i],
                        error=str(e),
                    )
        return [r for r in results if r is not None]
