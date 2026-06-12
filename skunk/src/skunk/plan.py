"""Plan + planner. `Planner.plan(question, ctx)` emits a `Plan`; the orchestrator
walks `Plan.branches`. The Plan AST mirrors the wire JSON exactly, so
`model_validate_json` / `model_dump_json` round-trip with no custom translation.
Branches are a discriminated union keyed by `kind` (`retrieve` / `lookup_external`)."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable
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
    key: NonEmptyStr  # NL phrase describing the data to find
    period: str | None = (
        None  # YYYY-MM month/range/comma-list the DATA pertains to ("2013-06", "2022-10..2023-09", "1940-01..1940-12, 1953-01..1953-12"); None if unpinned
    )
    as_of: str | list[str | None] | None = (
        None  # bulletin ISSUE (publication month) to read values from. A single "YYYY-MM" pins the WHOLE branch (era prune + year filter; block_select tie-breaker). A list pins per period entry — aligned 1:1 with the comma-separated `period` entries, None for unpinned slots (parity enforced in the planner parse hooks). Stamped onto extracted values as provenance
    )
    visual_only: bool = False


class LookupBranch(BaseModel):
    """A lookup_external branch — one external-source request."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["lookup_external"] = "lookup_external"
    target: NonEmptyStr  # NL request for a single external value
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


_MONTH_RE = re.compile(r"\d{4}-\d{2}")


def _as_of_problem(b: RetrieveBranch) -> str | None:
    """An `as_of` shape complaint for the branch, or None when well-formed. A list
    `as_of` must align 1:1 with the comma-separated `period` entries (None slots =
    unpinned) and hold YYYY-MM months. Raised through the parse hooks so the retry
    loop has the model re-emit."""
    if not isinstance(b.as_of, list):
        return None
    n_entries = len([p for p in (b.period or "").split(",") if p.strip()])
    if len(b.as_of) != n_entries:
        return (
            f"as_of list has {len(b.as_of)} slots but period {b.period!r} has "
            f"{n_entries} entries — they must align 1:1 (use null for unpinned slots)"
        )
    bad = [m for m in b.as_of if m is not None and not _MONTH_RE.fullmatch(m)]
    if bad:
        return f"as_of months must be YYYY-MM or null, got {bad!r}"
    return None


def _norm_for_match(s: str) -> str:
    """Normalize text for the verbatim-key check: NFKC folds unicode fractions /
    ligatures (so "2⅜" matches "2-3/8"-style spans the way a reader sees them),
    casefold makes it case-insensitive, and whitespace is collapsed so multi-space
    / newline differences don't matter. Punctuation is preserved — the key is meant
    to be copied, so "2-3/8%" must reproduce its hyphen and slash."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()


def _key_verbatim_violations(branches: Iterable[Branch], question: str) -> list[str]:
    """Retrieve `key`s that are NOT a verbatim span of the question (normalized). A
    retrieve `key` points into the corpus, so it must be named in the question's own
    words — substituting an invented table name is the failure behind UID0042
    ("Unemployment Trust Fund state accounts" for "unemployment insurance tax
    receipts") and UID0085 ("by State" for "from California"). The matching predicate
    is isolated here so strict contiguous-substring can be swapped for a looser rule
    (e.g. token-subset) in one place. Enforced on the INITIAL plan only — a failed
    key is exactly what the replanner is free to reword."""
    q = _norm_for_match(question)
    return [
        b.key
        for b in branches
        if isinstance(b, RetrieveBranch) and _norm_for_match(b.key) not in q
    ]


def _src_verbatim_violations(branches: Iterable[Branch], question: str) -> list[str]:
    """Non-null lookup_external `src`s that are NOT a verbatim span of the question.
    A `src` pins a publisher, and a lookup may pin one only when the question itself
    names it (else null) — this keeps the replanner from over-constraining a lookup
    to a guessed source. lookup `target` is exempt: it is reworded into the external
    source's vocabulary by design (steered by a prompt rule, not this check). Lookups
    are emitted only at replan, so this check runs in `_parse_plan_diff`, not
    `_parse_plan`."""
    q = _norm_for_match(question)
    return [
        b.src
        for b in branches
        if isinstance(b, LookupBranch)
        and b.src is not None
        and _norm_for_match(b.src) not in q
    ]


def _parse_plan(raw: str, ctx: ExecutionContext) -> Plan:
    """The initial planner's parse hook. Validates the reply as a `Plan`, then:
    (1) forbids `lookup_external` — the first pass is retrieve-only, every value is
    assumed to live in the corpus, and external lookups are injected only at replan
    when retrieval cannot find a value; (2) enforces well-formed `as_of` pins (list
    parity with `period`); (3) enforces that every retrieve `key` is copied VERBATIM
    from the question text (modulo case/unicode/whitespace) — the planner must name
    retrieval targets in the question's own words rather than substituting an
    invented table name (the failure mode behind UID0042 and UID0085). Violations
    raise `ParseError` so the retry loop re-emits.

    The key-verbatim check is INITIAL-plan only — `_parse_plan_diff` does not apply
    it, since rewording a failed key is exactly why we replan. The lookup `src`
    verbatim rule lives in `_parse_plan_diff` instead, because lookups are emitted
    only at replan."""
    plan = _json_parser(Plan)(raw, ctx)
    lookups = [b for b in plan.branches if isinstance(b, LookupBranch)]
    if lookups:
        raise ParseError(
            raw,
            "lookup_external is not allowed on the initial plan: assume every value "
            "the question needs is in the corpus and emit a retrieve branch for it "
            "(external lookups are added automatically at replan if retrieval fails). "
            f"Re-emit these as retrieve branches: {[b.target for b in lookups]!r}",
        )
    for b in plan.branches:
        if isinstance(b, RetrieveBranch) and (problem := _as_of_problem(b)):
            raise ParseError(raw, f"branch {b.key!r}: {problem}")
    bad = _key_verbatim_violations(plan.branches, ctx.question)
    if bad:
        raise ParseError(
            raw,
            "every retrieve `key` must be copied verbatim from the question text "
            "(case/whitespace-insensitive); these are not spans of the question and "
            f"must be reworded to use the question's exact wording: {bad!r}",
        )
    return plan


def _parse_plan_diff(raw: str, ctx: ExecutionContext) -> PlanDiff:
    """The replanner's parse hook: a valid `PlanDiff` whose added retrieve branches
    have null `as_of` — a replan-time pin on the wrong issue excludes the right one
    outright, so issue choice is left to retrieval. Added retrieve `key`s are EXEMPT
    from the verbatim check (rewording a failed key is the point of replanning), but
    a non-null lookup `src` must still name a publisher the question itself names
    (else null) — that rule lives here because lookups are emitted only at replan.
    Violations raise `ParseError`, and the retry loop has the model re-emit."""
    diff = _json_parser(PlanDiff)(raw, ctx)
    pinned = [
        b.key
        for b in diff.add
        if isinstance(b, RetrieveBranch)
        and (b.as_of if not isinstance(b.as_of, list) else any(b.as_of))
    ]
    if pinned:
        raise ParseError(
            raw, f"added retrieve branches must have null as_of: {pinned!r}"
        )
    src_bad = _src_verbatim_violations(diff.add, ctx.question)
    if src_bad:
        raise ParseError(
            raw,
            "a non-null lookup_external `src` must name a publisher the question "
            "itself names, copied verbatim (case/whitespace-insensitive); set `src` "
            f"to null when the question names no source: {src_bad!r}",
        )
    return diff


class Planner:
    _INITIAL_PLAN_PROMPT = """\
You are a query planner. Given a question, emit a JSON plan that, when executed, produces the answer.

## Output format

{
  "branches": [
    {"kind": "retrieve",
     "key": "<natural-language lookup string>",
     "period": "<str | null>",
     "as_of": "<str | array | null>",
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
  key           natural-language phrase describing the data to find — one singular, cohesive
                concept per branch; favor separate branches for distinct concepts. Examples:
                "national defense expenditures", "weekly average discount rate for new 91-day bills".
                Never emit two branches for the same underlying table/statistic worded
                differently (one auction's bids, allotments, and totals = one branch).
                Same concept over several periods = one branch with a comma-separated period.
  period        The period the data value pertains to as canonical months: a single
                "YYYY-MM", an inclusive "YYYY-MM..YYYY-MM" range, or a comma-separated
                list of these ("1940-01..1940-12, 1953-01..1953-12") — expanding fiscal
                years, calendar years, and quarters to month ranges. Null when the
                question doesn't pin a data period.
  as_of         The exact ISSUE (publication month) values must be read from — only
                when the question explicitly names a file. A single "YYYY-MM"
                applies to the whole branch. When the named issue applies to only some
                period entries, give an array aligned 1:1 with the comma-separated
                period entries, null for unpinned slots:
                period "1980-10..1981-09, 1979-10..1980-09" with as_of [null, "1981-11"]
                = FY1981 unpinned, FY1980 as printed in the November 1981 issue.
                Null (common case).
  visual_only   true only if question explicitly asks for visual understanding of charts/figures.

lookup_external branch fields:
  target        natural-language request for one external value, or for the same series
                across consecutive periods (e.g. "U.S. CPI-U for July 1953"). The same
                series across N periods is one branch with a multi-period target, not N
                branches. Keep the question's exact wording for any identifier it
                supplies (security descriptor, coupon, maturity, date); reword only the
                series/source name.
  src           a publisher name, set if and only if the question itself names a single,
                unambiguous external source — copied verbatim from the question. Never
                infer a publisher the question does not name. Otherwise null.
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
  - A prior branch you wish to keep appears in neither list: leave it alone and
    its gathered data is reused as-is. Do not re-add branches whose data is
    already in `prev`.
  - If data is missing, `add` retrieve/lookup branches that close the gap.
  - For each failed branch you still need, read its "what the attempt found /
    why it was blocked" note and formulate an alternative query — check that
    you are asking at the right granularity, and that the value really is (or
    is not) in the corpus before choosing retrieve vs lookup_external.
  - Added retrieve branches must have null `as_of`.
  - A non-null lookup_external `src` must name a publisher the question itself
    names, copied verbatim; otherwise leave `src` null.
"""

    # Planner-only addenda, appended to the planner system prompt alone (the replanner
    # gets `_REPLAN_INSTRUCTIONS` instead). `_VERBATIM_RULE`: retrieve keys must be
    # question spans — a failed key is exactly what the replanner reformulates, so it
    # is exempt. `_INITIAL_PASS_RULE`: no lookup_external on the first pass (retrieve
    # everything; lookups are injected at replan). The lookup `src` verbatim rule
    # lives in `_parse_plan_diff`, since lookups are emitted only at replan.
    _VERBATIM_RULE = """\
## Retrieve-key rule
Each retrieve `key` must be copied verbatim from the question: an exact span of
the question text, in the question's own wording.
"""

    _INITIAL_PASS_RULE = """\
## Initial-pass rule
This is the first pass: emit only `retrieve` branches, never `lookup_external`.
Assume every value the question needs lives in the Treasury Bulletin corpus,
including values that look external (CPI, GDP, FX rates), and key each retrieve
by the question's wording. External lookups are injected at replan if needed.
"""

    # Planner and replanner share the initial-plan instructions (same branch/field
    # semantics); the replanner's system prompt extends them with the diff
    # instructions, so it carries the full planning context plus how to revise. They
    # differ in the parsed output shape — `Plan` vs `PlanDiff` (a delta steered by the
    # dynamic context in the user message). Both are stateless, so they live on the
    # class rather than being rebuilt per instance.
    _prompt = PromptedCall(
        name="planner",
        system_prompt=f"{_INITIAL_PLAN_PROMPT}{_VERBATIM_RULE}{_INITIAL_PASS_RULE}",
        default_effort="medium",
        parse=_parse_plan,
        output_instruction="Output the Plan as a single bare JSON object — no markdown fences, no prose.",
    )
    _replan_prompt = PromptedCall(
        name="replanner",
        system_prompt=f"{_INITIAL_PLAN_PROMPT}\n\n{_REPLAN_INSTRUCTIONS}",
        default_effort="medium",
        parse=_parse_plan_diff,
        output_instruction="Output the PlanDiff as a single bare JSON object — no markdown fences, no prose.",
    )

    async def plan(self, question: str, ctx: ExecutionContext) -> Plan:
        return await self._prompt.call(ctx, f"Question: {question}", temperature=0.0)

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
        return await self._replan_prompt.call(ctx, "\n\n".join(parts), temperature=0.4)
