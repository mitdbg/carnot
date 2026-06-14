"""Plan + planner. `Planner.plan(question, ctx)` emits a `Plan`; the orchestrator
walks `Plan.branches`. The Plan AST mirrors the wire JSON exactly, so
`model_validate_json` / `model_dump_json` round-trip with no custom translation.
Branches are a discriminated union keyed by `kind` (`retrieve` / `lookup_external`)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
from skunk.common import (
    AnnotatedValue,
    ExecutionContext,
    input_values_desc,
)


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
        None  # bulletin ISSUE (publication month) to prefer values from. SOFT hint only — retrieval does NOT hard-filter on it (issue/print choice is selection's job); used as a search-agent hint and stamped onto extracted values as provenance. A single "YYYY-MM" hints the whole branch; a list hints per period entry (aligned 1:1 with the comma-separated `period`, None for unpinned slots)
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


@dataclass(frozen=True)
class AttemptRecord:
    """One branch tried in some sweep, with its fate — rendered into the replan
    message so the replanner can reword failed branches instead of re-emitting
    them verbatim, and never re-requests data a succeeded branch already gathered."""

    round_idx: int  # 0 = initial sweep
    branch: Branch
    n_entries: int  # values produced (0 when failed)
    error: StepFailed | None  # set iff the branch failed; carries reason + diagnostic


def _json_parser[T: BaseModel](model: type[T]) -> Callable[[str, ExecutionContext], T]:
    """Build a `PromptedCall` parse hook that validates the reply as `model`,
    raising `ParseError` (so the retry loop can echo it back) on malformed/invalid
    JSON. Used by both the planner and replanner parse hooks (both emit a `Plan`)."""

    def parse(raw: str, ctx: ExecutionContext) -> T:
        try:
            return model.model_validate_json(strip_code_fence(raw).strip())
        except ValidationError as e:
            raise ParseError(raw, str(e)) from e

    return parse


_parse_plan = _json_parser(Plan)
_parse_replan = _json_parser(Plan)


class Planner:
    _INITIAL_PLAN_PROMPT = """\
You are a query planner. Given a question, emit a JSON plan that, when executed, produces the answer.

## Output format

{
  "branches": [
    {"kind": "retrieve",
     "key": "<natural-language lookup string>",
     "period": "<str | null>",
     "as_of": "<YYYY-MM | [YYYY-MM | null, ...] | null>",
     "visual_only": <bool>},
    {"kind": "lookup_external",
     "target": "<natural-language request for a single value>",
     "src": "<natural-language description of requested source, if applicable | null>"}
  ]
}

Branches run in parallel; a final compute step reads the gathered values and the
verbatim question to produce the answer. Choose per value: prefer a `retrieve` branch
whenever the data could reasonably be expected to exist in the corpus; use a
`lookup_external` branch for a value that is unlikely to be in the corpus, or when the
question names a clear external source.

## Field semantics

retrieve branch fields:
  key           natural-language phrase describing the data to find — one singular, cohesive
                concept per branch; favor separate branches for distinct concepts. Examples:
                "national defense expenditures", "weekly average discount rate for new 91-day bills".
                Never emit two branches for the same underlying table/statistic worded
                differently (one auction's bids, allotments, and totals = one branch).
                Same concept over several periods = one branch with a comma-separated period.
  period        the period the data pertains to, as canonical months: "YYYY-MM", an
                inclusive "YYYY-MM..YYYY-MM" range, or a comma-separated list of these —
                expand fiscal years, calendar years, and quarters to month ranges. Null
                when the question doesn't pin a data period.
  as_of         the bulletin ISSUE (publication month, "YYYY-MM") to prefer values from,
                only when the question names a specific print/vintage; else null. A SOFT
                hint — selection still chooses the issue — so do not guess. A list aligns
                1:1 with the comma-separated `period` entries (null for unpinned slots).
  visual_only   true only if question explicitly asks for visual understanding of charts/figures.

lookup_external branch fields:
  target        natural-language request for one external value, or for the same series
                across consecutive periods (e.g. "U.S. CPI-U for July 1953"). The same
                series across N periods is one branch with a multi-period target, not N
                branches. Keep the question's exact wording for any identifier it
                supplies (security descriptor, coupon, maturity, date); reword only the
                series/source name.
  src           a publisher name, set when the question names a clear external source.
                Otherwise null.
"""

    _REPLAN_INSTRUCTIONS = """\
You are replanning: the data gathered so far was insufficient to answer.
Emit a fresh Plan JSON (schema above) containing ONLY the branches to run
NOW — the new gathering actions. At least one branch.

Rules:
  - DIAGNOSE FIRST: compare the previous plans, the values compute kept,
    and what compute says is missing — work out why the gathering so far
    did not satisfy compute, then emit branches that fix exactly that.
    Never re-emit an approach that already failed the same way.
  - Values under "Values compute KEPT" stay available to compute; never
    emit a branch for data already there. Everything else compute saw was
    retrieved and dropped as not useful in its current form.
  - Read each failed attempt's diagnostic before retrying it: reword the
    key, check the granularity asked for, and whether the value really is
    in the corpus (retrieve) or external (lookup_external).
  - lookup_external is unrestricted here; use it when the corpus has no
    home for the value.
  - When a committed computed intermediate pins the period of a missing
    value, set the new branch's `period` to exactly that period.
  - Human-provided resolutions explicitly map prior missing identifiers to
    entries already in the kept value pool. Treat those identifiers as resolved
    by those entries; do not request them again unless the latest missing-data
    signal still names them (meaning the supplied information was insufficient).
"""

    # Planner and replanner share the initial-plan instructions (same branch/field
    # semantics); the replanner's system prompt extends them with the fresh-compose
    # instructions, so it carries the full planning context plus how to revise. Both
    # parse to a `Plan`, but the replanner's holds only the branches to run NOW,
    # steered by the pool/archive/attempt-history context in the user message. Both
    # are stateless, so they live on the class rather than being rebuilt per instance.
    _prompt = PromptedCall(
        name="planner",
        system_prompt=_INITIAL_PLAN_PROMPT,
        default_effort="medium",
        parse=_parse_plan,
        output_instruction="Output the Plan as a single bare JSON object — no markdown fences, no prose.",
    )
    _replan_prompt = PromptedCall(
        name="replanner",
        system_prompt=f"{_INITIAL_PLAN_PROMPT}\n\n{_REPLAN_INSTRUCTIONS}",
        default_effort="medium",
        parse=_parse_replan,
        output_instruction="Output the Plan as a single bare JSON object — no markdown fences, no prose.",
    )

    async def plan(self, question: str, ctx: ExecutionContext) -> Plan:
        return await self._prompt.call(ctx, f"Question: {question}", temperature=0.4)

    @staticmethod
    def _attempts_section(attempts: list[AttemptRecord]) -> str:
        """Render the attempt-history block of the replan message: every branch
        tried so far, grouped by round, with its fate. A failed branch carries its
        reason and the first-hand diagnostic the attempt recorded — fresh-compose
        has no prior plan to anchor on, so this block is what keeps the replanner
        from re-emitting an already-failed branch verbatim."""
        lines = ["Previous plans (every branch tried so far, with outcome):"]
        cur_round = None
        for a in attempts:
            if a.round_idx != cur_round:
                cur_round = a.round_idx
                lines.append(f"  round {cur_round}:")
            head = f"    - {a.branch.model_dump_json(exclude_none=True)}"
            if a.error is None:
                lines.append(f"{head} → ok ({a.n_entries} values gathered)")
            else:
                lines.append(f"{head} → FAILED: {a.error.reason}")
                if a.error.diagnostic:
                    lines.append("      what the attempt found / why it was blocked:")
                    lines.extend(
                        "        " + ln for ln in a.error.diagnostic.splitlines()
                    )
        return "\n".join(lines)

    async def replan(
        self,
        ctx: ExecutionContext,
        pool: list[AnnotatedValue],
        attempts: list[AttemptRecord],
        missing_reason: str,
        missing: list[str],
        human_resolutions: list[tuple[int, list[str]]] | None = None,
    ) -> Plan:
        parts = [
            f"Question: {ctx.question}",
            "Values compute KEPT for the next round (available to compute; do NOT "
            f"request again — everything else it saw was dropped):\n{input_values_desc(pool)}",
            self._attempts_section(attempts),
            f"What compute says is missing:\n  description: {missing_reason}\n  missing:     {missing!r}",
        ]
        if human_resolutions:
            resolved = "\n".join(
                f"  - prior missing {resolved_missing!r} -> input_values[{input_index}]"
                for input_index, resolved_missing in human_resolutions
            )
            parts.append(
                "Human-provided resolutions (explicit mapping to available inputs):\n"
                f"{resolved}"
            )
        return await self._replan_prompt.call(ctx, "\n\n".join(parts), temperature=0.4)
