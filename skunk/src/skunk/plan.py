"""Plan + planner.

`PlannerExecutor.plan(question, ctx)` emits a `Plan`; the orchestrator walks
`Plan.branches` and threads `PageRef` / `AnnotatedValue` (both defined in
`skunk.models`) between operators. The compute operator's terminal output
is a bare `str` (the final answer).

The Plan AST mirrors the wire JSON shape:

    Plan(branches=[...],
         computation=Computation(task=..., qualifiers=[...]),
         presentation=Presentation(units_out=..., precision=..., answer_form=...))

Branches are a discriminated union keyed by `kind`:
    RetrieveBranch(kind="retrieve", key, period, visual_only, value_kind)
    LookupBranch(kind="lookup_external", target, src)

`Plan.model_validate_json(s)` / `plan.model_dump_json()` round-trip without
any custom translation — model layout *is* the wire layout. Cached plans
live in `data/dsl_planning_pass.csv` as JSON strings in the `plan_json`
column.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
)

from skunk.errors import StepFailed
from skunk.executor import SkunkExecutor
from skunk.models import HarnessContext


class RetrieveBranch(BaseModel):
    """A corpus-retrieval branch. The implicit extract step downstream uses
    `value_kind` as a shape hint and `visual_only` to skip text/OCR tiers."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["retrieve"] = "retrieve"
    # Free-form NL phrase describing the data to find. Embedded into the
    # corpus ANN query alongside the user's question.
    key: str
    # Free-form NL period — "FY 2023", "2023-01", "January 1940", etc.
    # Planner emits null when the question doesn't pin one; we coerce to "".
    period: str = ""
    visual_only: bool = False
    # Advisory shape qualifier for the extract operator.
    value_kind: Literal["scalar", "vector", "table"] | None = None

    @field_validator("key")
    @classmethod
    def _key_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("retrieve.key is empty")
        return v

    @field_validator("period", mode="before")
    @classmethod
    def _period_none_to_empty(cls, v: Any) -> Any:
        return "" if v is None else v


class LookupBranch(BaseModel):
    """A lookup_external branch — one external-source request."""
    model_config = ConfigDict(frozen=True)

    kind: Literal["lookup_external"] = "lookup_external"
    # Free-form NL request for a single external value.
    target: str
    # Optional NL hint about the preferred source. Threaded into the
    # operator's user prompt to bias source routing.
    src: str | None = None

    @field_validator("target")
    @classmethod
    def _target_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("lookup_external.target is empty")
        return v


Branch = Annotated[
    Union[RetrieveBranch, LookupBranch],
    Field(discriminator="kind"),
]


class Computation(BaseModel):
    """Computation envelope — natural-language spec for the implicit
    terminal compute phase. `task` and any items of `qualifiers` may be
    omitted or null in the wire; both default to empty."""
    model_config = ConfigDict(frozen=True)
    task: str | None = None
    qualifiers: list[str] = Field(default_factory=list)

    @field_validator("qualifiers", mode="before")
    @classmethod
    def _strip_null_items(cls, v: Any) -> Any:
        # planner sometimes emits `[null]` when no qualifiers apply.
        return [q for q in v if q is not None] if isinstance(v, list) else v


class Presentation(BaseModel):
    """Presentation envelope — output-format hints for the compute phase.
    All fields may be omitted or null in the wire; consumers coerce."""
    model_config = ConfigDict(frozen=True)
    units_out: str | None = None
    precision: int | None = Field(default=None, ge=0)
    answer_form: str | None = None


class Plan(BaseModel):
    """The whole plan — model layout mirrors the wire JSON exactly, so
    `model_validate_json` / `model_dump_json` round-trip with no custom
    translation. See module docstring for the wire shape."""
    model_config = ConfigDict(frozen=True)

    branches: list[Branch] = Field(min_length=1)
    computation: Computation = Field(default_factory=Computation)
    presentation: Presentation = Field(default_factory=Presentation)


class PlannerExecutor(SkunkExecutor):
    name: str = "planner"
    system_prompt: str = """\
You are the planner. Given a question, emit a JSON plan that, when executed, produces the answer.

## Output format

{
  "branches": [
    {"kind": "retrieve",
     "key": "<natural-language lookup string>",
     "period": "<str | null>",
     "value_kind": "<scalar|vector|table>",
     "visual_only": <bool>},
    {"kind": "lookup_external",
     "target": "<natural-language request for a single value>",
     "src": "<natural-language description of requested source, if applicable | null>"}
  ],
  "computation": {
     "task": "<natural-language describing the calculation>",
     "qualifiers": [<short qualifier phrases pinned by the question | null>]
  },
  "presentation": {
    "units_out": "<unit | null>",
    "precision": <int | null>,
    "answer_form": <natural-language describing the output format | null>
  }
}

Branches run in parallel. A `retrieve` branch pulls annotated values from
the corpus. A `lookup_external` branch fetches a single value from
outside the corpus. `computation` + `presentation` feed an implicit
final compute phase.

## Field semantics

retrieve branch fields:
  key           natural-language phrase describing the data to find. Focus only on a singular, cohesive concept.
                Favor separate branches if the question requires retrieval of multiple values. Examples:
                "national defense expenditures", "weekly average discount rate for new 91-day bills".
  period        temporal mask in natural language, e.g. "FY 2023", "2023-01", "2023-01-01".
                Null when the question doesn't pin one.
  value_kind    shape to aim for:
                  vector for 1-D series aggregations (means, growth
                    rates, regressions, argmax/argmin over a category).
                  table for 2-D joint analyses (correlations,
                    cross-tabulations).
                  scalar for single-point lookups.
  visual_only   true only if question explicitly asks for visual understanding of charts/figures.

lookup_external branch fields:
  target        natural-language request for a single external value (e.g. "U.S. CPI-U for July 1953",
                "JPY/USD spot rate on 2010-06-30"). Use only for values that the corpus is unlikely to carry or when explicitly instructed to do so.
  src           natural-language description of the source, only if the question explicitly asks for one (e.g. "Bureau of Labor Statistics").

computation fields:
  task        Natural language describing the calculation to carry out
              once retrieval has returned relevant data. 

  qualifiers  Optional list of short phrases that nail down a specific qualifier. Each
              qualifier is one phrase; the compute operator treats every qualifier as a MUST. Use only when
              the question explicitly pins a choice. null or empty list when the task is
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

"""

    def plan(self, question: str, ctx: HarnessContext) -> Plan:
        """Generate a Plan from a natural-language question."""
        system_prompt = self.assemble_system_prompt(ctx)

        base_user_message = f"""\
Question: {question}

Produce the Plan JSON. Output a single bare JSON object. No markdown fences. No prose.
"""

        attempt_errors: list[str] = []
        last_raw: str | None = None
        last_error: str | None = None

        for attempt in range(3):
            if attempt == 0:
                user_message = base_user_message
            else:
                # Each retry shows ONLY the most recent bad response + its
                # error — no history accumulation. The model needs to see
                # what it just emitted to self-correct.
                user_message = (
                    f"{base_user_message}\n"
                    f"Your previous attempt produced this output:\n"
                    f"```\n{last_raw}\n```\n\n"
                    f"It failed with: {last_error}\n"
                    "Fix and return valid JSON only."
                )
            ctx.emit("planner", "attempt", n=attempt + 1, of=3)
            resp = ctx.llm_client.call(system_prompt, user_message, thinking_budget=-1, ctx=ctx)
            raw = resp.text
            try:
                return Plan.model_validate_json(raw.strip())
            except ValidationError as e:
                attempt_errors.append(f"Attempt {attempt + 1}: {e}")
                last_raw = raw
                last_error = str(e)
                ctx.emit(
                    "planner", "parse/validate failed",
                    n=attempt + 1, error_type=type(e).__name__, error=str(e),
                )

        ctx.emit("planner", "exhausted", attempts=len(attempt_errors))
        raise StepFailed(
            "planner",
            f"Failed to produce valid Plan after {len(attempt_errors)} attempts: "
            + "; ".join(attempt_errors),
        )
