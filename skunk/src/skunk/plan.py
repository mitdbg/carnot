"""Plan + planner. `Planner.plan(question, ctx)` emits a `Plan`; the orchestrator
walks `Plan.branches`. The Plan AST mirrors the wire JSON exactly, so
`model_validate_json` / `model_dump_json` round-trip with no custom translation.
Branches are a discriminated union keyed by `kind` (`retrieve` / `lookup_external`)."""

from __future__ import annotations

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
from skunk.common import AnnotatedValue, HarnessContext


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
    period: str | None = None   # NL period the DATA pertains to ("FY 2023", "2003"); None if unpinned
    as_of: str | None = None    # NL reporting bulletin/vintage ("June 2013 bulletin"); None unless pinned
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


class Computation(BaseModel):
    """NL spec for the terminal compute phase."""

    model_config = ConfigDict(frozen=True)
    task: str | None = None
    qualifiers: list[str] = Field(default_factory=list)


class Presentation(BaseModel):
    """Output-format hints for the compute phase."""

    model_config = ConfigDict(frozen=True)
    units_out: str | None = None
    precision: int | None = Field(default=None, ge=0)
    answer_form: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(frozen=True)

    branches: list[Branch] = Field(min_length=1)
    computation: Computation = Field(default_factory=Computation)
    presentation: Presentation = Field(default_factory=Presentation)


# Output-format instruction appended to every planner message by `_user_message`.
_OUTPUT_TAIL = (
    "Produce the Plan JSON. Output a single bare JSON object. "
    "No markdown fences. No prose."
)


def _parse_plan(raw: str, ctx: HarnessContext) -> Plan:
    """Parse a planner reply into a validated `Plan`; raise `ParseError` on
    malformed/invalid JSON so the retry loop can echo it back."""
    try:
        return Plan.model_validate_json(strip_code_fence(raw).strip())
    except ValidationError as e:
        raise ParseError(raw, str(e)) from e


class Planner:
    _SYSTEM_PROMPT = """\
You are the planner. Given a question, emit a JSON plan that, when executed, produces the answer.

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
  ],
  "computation": {
     "task": "<natural-language describing the calculation>",
     "qualifiers": [<short qualifier phrases>]
  },
  "presentation": {
    "units_out": "<unit | null>",
    "precision": <int | null>,
    "answer_form": <natural-language describing the output format | null>
  }
}

Branches run in parallel. A `retrieve` branch pulls information from the corpus.
A `lookup_external` branch fetches a single value from outside the corpus. Use 'lookup_external' only when you are sure the corpus does not contain the answer,
when the question explicitly asks for an external lookup from a source, or when previous lookups in the corpus failed. `computation` + `presentation` inform a final
compute step.

## Field semantics

retrieve branch fields:
  key           natural-language phrase describing the data to find. Focus only on a singular, cohesive concept.
                Favor separate branches if the question requires retrieval of multiple values. Examples:
                "national defense expenditures", "weekly average discount rate for new 91-day bills".
  period        The period the DATA VALUE PERTAINS TO — e.g. "FY 2023", "2023-01", "2003".
                This is what extract uses to pick the row/column. It is NOT the bulletin/report
                date. Null when the question doesn't pin one.
  as_of         The bulletin/vintage the value is REPORTED IN / AS OF, when the question pins one
                (e.g. "as reported at the end of FY 2013" → "June 2013 bulletin"). This selects
                WHICH DOCUMENT to read, not which row. Null otherwise (the common case).

                period vs as_of — keep them distinct. A bulletin reprints history, so a value FOR
                year X can be reported IN a later bulletin Y. "values reported at the end of FY 2013
                for 2003 and 2012" → period = "2003" / "2012", as_of = "FY 2013 bulletin".
                NEVER put the report date (2013) in `period`.
  visual_only   true only if question explicitly asks for visual understanding of charts/figures.

lookup_external branch fields:
  target        natural-language request for one external value, OR for the same series
                across consecutive periods (e.g. "U.S. CPI-U for July 1953",
                "JPY/USD spot rate on 2010-06-30",
                "annual average GBP per USD for 1950, 1951, 1952").
                When the question needs the same series across N periods, emit ONE branch
                with a multi-period target; the lookup agent returns a labeled vector
                (period→value). Do NOT split into N separate branches — that multiplies failure risk.
  src           Set to a publisher name if and only if the question requests a
                single, unambiguous external source. Otherwise null.

computation fields:
  task        Natural language describing the calculation to carry out
              once retrieval has returned relevant data. 

  qualifiers  Optional list of short phrases that nail down a specific qualifier. Each
              qualifier is one phrase; the compute operator treats every qualifier as a MUST. Use only when
              the question explicitly pins a choice. Empty list (or omit the field) when the task is
              unambiguous. Preserve qualifier words from the question VERBATIM — do not
              paraphrase, simplify, or otherwise modify them. The compute step
              relies on the exact wording to choose the right operation.

presentation fields:
  How the final answer is rendered. No semantic content about the calculation itself.

  units_out    Unit of the final answer (e.g., "in millions of
                   dollars", "as a percent", "in DEM"). Use exact matching strings from the question. 
                   If the question does NOT name a unit, output `null`.
                   Do not guess; do not invent; do not fall back to
                   "text" or any other placeholder.
  precision    decimal places of the final answer;
               null when not pinned.
  answer_form  examples: no commas, bracketed_list for
               "[a, b, c]", labeled_pair for
               "[year, value]".

The dataset section below enumerates the corpus-specific conventions and points to a few
worked examples.

{{ default_tail }}"""

    def __init__(self) -> None:
        self._prompt = PromptedCall(
            name="planner",
            system_prompt=self._SYSTEM_PROMPT,
            default_effort="medium",
            parse=_parse_plan,
        )

    def _user_message(self, question: str, body: str = "") -> str:
        """Compose a planner user message: question header + optional body
        (replan context) + output-format tail."""
        middle = f"{body.rstrip(chr(10))}\n\n" if body else ""
        return f"Question: {question}\n\n{middle}{_OUTPUT_TAIL}\n"

    def plan(self, question: str, ctx: HarnessContext) -> Plan:
        return self._prompt.call(ctx, self._user_message(question))

    def replan(
        self,
        ctx: HarnessContext,
        prior_plan: Plan,
        prev: list[AnnotatedValue],
        failed_branches: list[tuple["Branch", StepFailed]],
        missing_reason: str,
        missing: list[str],
    ) -> Plan:
        """Re-plan after compute reported MissingData. The orchestrator diffs the
        returned branches against `prior_plan.branches` and executes only the
        additions; `computation`/`presentation` fully replace the prior values.
        `failed_branches` are prior branches that raised `StepFailed` (output NOT
        in `prev`), retryable via structurally-different replacements. Each
        `StepFailed.diagnostic` (the subagent's account of what it tried, what it
        found, and what blocked it) is rendered untruncated so the planner can
        pivot to a source/kind that can actually serve the request."""
        from skunk.compute import prev_desc

        if failed_branches:
            blocks = []
            for branch, err in failed_branches:
                block = (
                    f"  - {branch.model_dump_json(exclude_none=True)}\n"
                    f"    reason: {err.reason}"
                )
                if err.diagnostic:
                    diag = "\n".join("      " + ln for ln in err.diagnostic.splitlines())
                    block += f"\n    what the attempt found / why it was blocked:\n{diag}"
                blocks.append(block)
            failed_section = (
                "\nBranches that FAILED in the prior run (their output is NOT in prev):\n"
                + "\n".join(blocks)
                + "\n"
            )
        else:
            failed_section = ""

        body = f"""\
You produced a plan that could not be completed. Revise it.

prior_plan = {prior_plan.model_dump_json()}

prev (data already gathered; treat as available, do NOT request again):
{prev_desc(prev)}
{failed_section}
What was missing:
  description: {missing_reason}
  missing:     {missing!r}

Return a corrected plan in the same JSON schema. Rules:
  - Include only branches that fetch data you still need. Do not re-list
    anything already present in `prev`, and never emit two branches for
    the same value.
  - For each FAILED branch you still need, READ its "what the attempt
    found / why it was blocked" note and act on it — it is the prior
    attempt's first-hand account. If it says a source had no data for the
    requested period (e.g. a series doesn't reach that far back), do NOT
    re-issue the same `src`: drop or change it, switch the branch kind
    (`lookup_external` ↔ `retrieve`), or retarget pre-API history toward a
    web source. If it names candidate values it found, fold them into a
    `target` aimed at the source that had them. A verbatim repeat — or a
    mere rewording with the same `src`/kind — will fail the same way.
  - If the data is simply incomplete, add retrieve/lookup branches that
    close the gap.
  - If the missing-data signal shows the calculation itself was misframed
    (e.g. a qualifier was misread), also revise `computation` /
    `presentation`; these replace the prior values.
  - Emit nothing extraneous: no commentary, no placeholder branches.
  - If you genuinely have nothing to add and the prior framing was
    correct, re-emit the prior plan unchanged."""
        return self._prompt.call(ctx, self._user_message(ctx.question, body))
