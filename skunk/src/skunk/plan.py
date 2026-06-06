"""Plan + planner. `Planner.plan(question, ctx)` emits a `Plan`; the orchestrator
walks `Plan.branches`. The Plan AST mirrors the wire JSON exactly, so
`model_validate_json` / `model_dump_json` round-trip with no custom translation.
Branches are a discriminated union keyed by `kind` (`retrieve` / `lookup_external`)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from skunk.common import strip_code_fence
from skunk.errors import ParseError, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.common import AnnotatedValue, ExecutionContext, input_values_desc


def _strip_non_empty(v: str) -> str:
    if not v.strip():
        raise ValueError("must be non-empty")
    return v


NonEmptyStr = Annotated[str, AfterValidator(_strip_non_empty)]


class RetrieveBranch(BaseModel):
    """A corpus-retrieval branch. `visual_only` skips the parsed-text tier
    downstream and goes straight to vision."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["retrieve"] = "retrieve"
    key: NonEmptyStr            # NL phrase describing the data to find
    period: str | None = None   # YYYY-MM month/range the DATA pertains to ("2013-06", "2022-10..2023-09"); None if unpinned
    as_of: str | None = None    # YYYY-MM reporting bulletin month ("2012-09"); None unless pinned
    visual_only: bool = False


class LookupBranch(BaseModel):
    """A lookup_external branch — one external-source request."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["lookup_external"] = "lookup_external"
    target: NonEmptyStr     # NL request for a single external value
    src: str | None = None  # optional NL source hint, biases routing


Branch = Annotated[
    Union[RetrieveBranch, LookupBranch],
    Field(discriminator="kind"),
]


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    branches: list[Branch] = Field(min_length=1)


class PlanDiff(BaseModel):
    """A replan result expressed as a delta against the prior plan."""

    model_config = ConfigDict(frozen=True)

    add: list[Branch] = Field(default_factory=list)
    drop: list[int] = Field(default_factory=list)

    def apply(self, plan: Plan) -> tuple[Plan, list[int]]:
        drop = {i for i in self.drop if 0 <= i < len(plan.branches)}
        kept = [i for i in range(len(plan.branches)) if i not in drop]
        new_plan = plan.model_copy(
            update={"branches": [*(plan.branches[i] for i in kept), *self.add]}
        )
        return new_plan, kept


def _json_parser[T: BaseModel](model: type[T]) -> Callable[[str, ExecutionContext], T]:
    """Build a `PromptedCall` parse hook that validates the reply as `model`,
    raising `ParseError` (so the retry loop can echo it back) on malformed/invalid
    JSON. Used for both the planner (`Plan`) and replanner (`PlanDiff`)."""

    def parse(raw: str, ctx: ExecutionContext) -> T:
        try:
            return model.model_validate_json(strip_code_fence(raw).strip())
        except ValidationError as e:
            raise ParseError(raw, str(e)) from e

    return parse


class Planner:
    _INITIAL_PLAN_PROMPT = """\
You are a query planner. Given a question, emit a JSON plan that, when executed, produces the answer.

## Output format

{
  "branches": [
    {"kind": "retrieve",
     "key": "<natural-language lookup string>",
     "period": "<str | null>",
     "as_of": "<str | null>",
     "visual_only": <bool>},
    {"kind": "lookup_external",
     "target": "<natural-language request for a single value>",
     "src": "<natural-language description of requested source, if applicable | null>"}
  ]
}

Branches run in parallel. A `retrieve` branch pulls information from the corpus.
A `lookup_external` branch fetches a single value from outside the corpus. Use 'lookup_external' only when you are sure the corpus does not contain the answer,
when the question explicitly asks for an external lookup from a source, or when previous lookups in the corpus failed. A final compute step reads the
gathered values and the verbatim question to produce the answer.

## Field semantics

retrieve branch fields:
  key           natural-language phrase describing the data to find. Focus only on a singular, cohesive concept.
                Favor separate branches if the question requires retrieval of multiple values. Examples:
                "national defense expenditures", "weekly average discount rate for new 91-day bills".
  period        The period the DATA VALUE PERTAINS TO, as canonical months: a single
                "YYYY-MM" or an inclusive "YYYY-MM..YYYY-MM" range (expand fiscal years,
                calendar years, and quarters to month ranges yourself — see the corpus
                notes). This is what extract uses to pick the row/column. It is NOT the
                bulletin/report date. Null when the question doesn't pin one.
  as_of         The bulletin/vintage the value is REPORTED IN / AS OF, as the bulletin
                month "YYYY-MM", when the question pins one (e.g. "as reported at the end
                of FY 2013" → "2013-06"). This selects WHICH DOCUMENT to read, not which
                row. Null otherwise (the common case).
  visual_only   true only if question explicitly asks for visual understanding of charts/figures.

lookup_external branch fields:
  target        natural-language request for one external value, OR for the same series
                across consecutive periods (e.g. "U.S. CPI-U for July 1953", or for a
                multi-period series "TreasuryDirect index ratios for the 2-3/8% Jan-2027
                TIPS across January-August 2007").
                When the question needs the same series across N periods, emit ONE branch
                with a multi-period target; the lookup agent returns a labeled vector
                (period→value). Do NOT split into N separate branches — that multiplies failure risk.
  src           Set to a publisher name if and only if the question requests a
                single, unambiguous external source. Otherwise null.
"""

    _REPLAN_INSTRUCTIONS = """\
You are now replanning: a plan you produced could not be completed. Revise it by
emitting a DIFF against the prior plan — not a whole new plan. The same
branch/field semantics above still apply to any branch you add.

Return a PlanDiff JSON object with these fields:
  add            list of NEW branches to run (same retrieve/lookup_external
                 schema as a plan branch). Empty list if you add nothing.
  drop           list of prior-branch INDICES (from the numbered prior_plan.branches
                 list in the question) to remove. Removing a branch also discards the
                 data it gathered, so drop a branch only when its data is wrong or
                 must be re-fetched differently. Empty list if you drop nothing.

Rules:
  - A prior branch you wish to keep appear in NEITHER list: leave it
    alone and its gathered data is reused as-is. Do NOT re-add branches whose
    data is already in `prev`.
  - If data is missing, `add` retrieve/lookup branches that close the gap.
  - For each FAILED branch you still need, read its "what the attempt found / why
    it was blocked" note and formulate an alternative query. Pay special attention to whether
    you are asking for information at the right granularity, and whether you are correctly
    assuming whether a piece of information is in the corpus or should be fetched externally with
    the appropriate src.
"""

    # Planner and replanner share the initial-plan instructions (same branch/field
    # semantics); the replanner's system prompt extends them with the diff
    # instructions, so it carries the full planning context plus how to revise. They
    # differ in the parsed output shape — `Plan` vs `PlanDiff` (a delta steered by the
    # dynamic context in the user message). Both are stateless, so they live on the
    # class rather than being rebuilt per instance.
    _prompt = PromptedCall(
        name="planner",
        system_prompt=_INITIAL_PLAN_PROMPT,
        default_effort="medium",
        parse=_json_parser(Plan),
        output_instruction="Output the Plan as a single bare JSON object — no markdown fences, no prose.",
    )
    _replan_prompt = PromptedCall(
        name="replanner",
        system_prompt=f"{_INITIAL_PLAN_PROMPT}\n\n{_REPLAN_INSTRUCTIONS}",
        default_effort="medium",
        parse=_json_parser(PlanDiff),
        output_instruction="Output the PlanDiff as a single bare JSON object — no markdown fences, no prose.",
    )

    async def plan(self, question: str, ctx: ExecutionContext) -> Plan:
        return await self._prompt.call(ctx, f"Question: {question}")

    @staticmethod
    def _failed_section(
        prior_plan: Plan, failed_branches: list[tuple["Branch", StepFailed]]
    ) -> str:
        """Render the FAILED-branches block of the replan message ("" when none
        failed). Each entry carries the branch's prior index, its JSON, and the
        first-hand diagnostic the failed attempt recorded."""
        if not failed_branches:
            return ""
        branch_index = {id(b): i for i, b in enumerate(prior_plan.branches)}
        blocks = []
        for branch, err in failed_branches:
            idx = branch_index.get(id(branch))
            block = (
                f"  - [{idx}] {branch.model_dump_json(exclude_none=True)}\n"
                f"    reason: {err.reason}"
            )
            if err.diagnostic:
                diag = "\n".join("      " + ln for ln in err.diagnostic.splitlines())
                block += f"\n    what the attempt found / why it was blocked:\n{diag}"
            blocks.append(block)
        return (
            "Branches that FAILED in the prior run (their output is NOT in prev):\n"
            + "\n".join(blocks)
        )

    async def replan(
        self,
        ctx: ExecutionContext,
        prior_plan: Plan,
        prev: list[AnnotatedValue],
        failed_branches: list[tuple["Branch", StepFailed]],
        missing_reason: str,
        missing: list[str],
    ) -> PlanDiff:
        numbered = "\n".join(
            f"  [{i}] {b.model_dump_json(exclude_none=True)}"
            for i, b in enumerate(prior_plan.branches)
        )
        parts = [
            f"Question: {ctx.question}",
            f"prior_plan.branches (reference `drop` by these indices):\n{numbered}",
            "input_values (data already gathered; treat as available, do NOT request again):\n"
            f"{input_values_desc(prev)}",
        ]
        failed = self._failed_section(prior_plan, failed_branches)
        if failed:
            parts.append(failed)
        parts.append(
            f"What was missing:\n  description: {missing_reason}\n  missing:     {missing!r}"
        )
        return await self._replan_prompt.call(ctx, "\n\n".join(parts))
