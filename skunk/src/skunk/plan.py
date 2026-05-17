"""Plan dataclass + runtime value types + planner.

`PlannerExecutor.plan(question, ctx)` emits a `Plan`; the orchestrator walks
`Plan.branches` and threads `PageRef` / `AnnotatedValue` between operators.
The compute operator's terminal output is a bare `str` (the final answer).

The Plan AST is one flat dataclass:

    Plan(branches=[...], task=..., qualifiers=[...],
         units_out=..., precision=..., answer_form=...)

Branches are `RetrieveBranch(key, period, visual_only, value_kind)` or
`LookupBranch(target, src)`.

Plans cross the wire as JSON. The canonical shape is:

    {"branches": [...],
     "computation": {"task": "...", "qualifiers": ["...", ...]},
     "presentation": {"units_out": "...", "precision": ..., "answer_form": "..."}}

`from_json` / `to_json` round-trip that shape. Cached plans live in
`data/dsl_planning_pass.csv` as JSON strings in the `plan_json` column.

Page number convention:
  PageRef.page = 1-based PDF page index (canonical throughout the codebase).
  The bulletin's printed-page footer is recoverable via
  `skunk.extract.get_printed_page()` for trace/prompt enrichment,
  but is never used as a lookup key.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from skunk.common import HarnessContext, extract_json_object
from skunk.executor import SkunkExecutor
from skunk.operator import StepFailed

@dataclass
class PageRef:
    month: str | None = None        # "YYYY-MM"
    page: int | None = None         # 1-based PDF page index (canonical)

    @property
    def year(self) -> int | None:
        return int(self.month[:4]) if self.month else None

    def __post_init__(self) -> None:
        # parsed-JSON lookup requires month; catch missing month at construction time.
        if self.page is not None and self.month is None:
            raise ValueError(
                f"PageRef with page={self.page} requires month for parsed-JSON lookup"
            )

    def __repr__(self) -> str:
        parts = []
        if self.year:
            parts.append(f"year={self.year}")
        if self.month:
            parts.append(f"month={self.month}")
        if self.page is not None:
            parts.append(f"page={self.page}")
        return f"PageRef({', '.join(parts)})"


@dataclass
class AnnotatedValue:
    """One described, annotated datum. Carries the payload + minimal metadata.

    Fields:
      - description: free-form natural-language label that uniquely distinguishes
        this entry from siblings. Should include everything a reader needs to
        know about what this value represents — series, period, sub-category,
        unit qualifier, etc. There is no separate `dims` / `quote` field; rich
        context goes here as prose.
      - value: payload, shape determined by `kind`.
      - unit: semantic unit token (e.g. "usd_millions", "pct", "year").
      - kind: "scalar" | "vector" | "table".
      - index_name: vector only — name of the varying dim (e.g. "month").
      - row_name / col_name: table only — names of the two varying dims.
      - tag: short machine-readable key (snake_case, like "gross_federal_debt:fy1973-fy1980").
        Used by downstream consumers to select entries unambiguously when
        description substrings overlap. Two entries describing the same
        underlying series + period MUST share the same tag.
      - expected_index_range: vector only — short string like "1969-01..1980-01"
        describing the FULL index range the question requested. Lets compute
        flag gaps (actual vs expected). Default empty (no gap analysis).

    Payload shapes by kind:
      - "scalar": value is int | float | str.
      - "vector": value is dict[str, int|float|str], keyed by index_name labels.
      - "table":  value is dict[str, dict[str, int|float|str]],
                  outer key = row_name label, inner key = col_name label.
    """
    description: str
    value: Any
    unit: str = ""
    kind: str = "scalar"                  # "scalar" | "vector" | "table"
    index_name: str | None = None         # vector only
    row_name: str | None = None           # table only
    col_name: str | None = None           # table only
    tag: str = ""                         # machine-readable selection key
    expected_index_range: str = ""        # vector only — for gap-aware summary

VALUE_KIND_VOCAB: frozenset[str] = frozenset({"scalar", "vector", "table"})

@dataclass
class RetrieveBranch:
    # Free-form natural-language phrase describing the data to find. Embedded
    # into the corpus ANN query alongside the user's question.
    key: str
    # Free-form NL period — "FY 2023", "2023-01", "January 1940", etc.
    # Operators parse it best-effort; no longer enforced against a regex.
    period: str
    visual_only: bool = False       # passes through to the implicit extract step
    # Advisory shape qualifier for the extract operator. Validator pins to
    # VALUE_KIND_VOCAB above.
    value_kind: str | None = None


@dataclass
class LookupBranch:
    # Free-form NL request for a single external value (e.g. "CPI-U for July 1953").
    target: str
    # Optional NL hint about the preferred source (e.g. "Bureau of Labor Statistics",
    # "FRED", "Macrotrends"). Threaded into the operator's user prompt to bias
    # source routing.
    src: str | None = None


Branch = RetrieveBranch | LookupBranch

@dataclass
class Plan:
    """The whole plan in one flat dataclass.

    Mirrors the planner's JSON output:
      - `branches` ← planner's `branches`
      - `task` / `qualifiers` ← planner's `computation.task` / `computation.qualifiers`
      - `units_out` / `precision` / `answer_form` ← planner's `presentation.*`
    """
    branches: list[Branch] = field(default_factory=list)
    task: str = ""                                       # computation.task (free-form NL)
    qualifiers: list[str] = field(default_factory=list)  # computation.qualifiers
    units_out: str | None = None                         # presentation.units_out (free-form NL token)
    precision: int | None = None                         # presentation.precision
    answer_form: str = "scalar"                          # presentation.answer_form (free-form NL; "scalar" is the default)

def to_json(plan: Plan, *, indent: int | None = None) -> str:
    """Serialize a Plan to its canonical JSON string.

    Shape:
      {"branches": [...],
       "computation": {"task": "...", "qualifiers": [...]},  # omitted when both empty
       "presentation": {"units_out": "...", ...}}            # omitted when fully default

    Pass `indent=N` for pretty-printing (e.g. trace dumps).
    """
    branch_dicts: list[dict] = []
    for b in plan.branches:
        if isinstance(b, RetrieveBranch):
            bd: dict = {"kind": "retrieve", "key": b.key, "period": b.period}
            if b.visual_only:
                bd["visual_only"] = True
            if b.value_kind:
                bd["value_kind"] = b.value_kind
        elif isinstance(b, LookupBranch):
            bd = {"kind": "lookup_external", "target": b.target}
            if b.src:
                bd["src"] = b.src
        else:
            raise TypeError(f"Unknown branch type: {type(b)}")
        branch_dicts.append(bd)

    d: dict = {"branches": branch_dicts}
    if plan.task or plan.qualifiers:
        d["computation"] = {"task": plan.task, "qualifiers": list(plan.qualifiers)}
    pres: dict = {}
    if plan.units_out:
        pres["units_out"] = plan.units_out
    if plan.precision is not None:
        pres["precision"] = plan.precision
    if plan.answer_form != "scalar":
        pres["answer_form"] = plan.answer_form
    if pres:
        d["presentation"] = pres
    return json.dumps(d, indent=indent, ensure_ascii=False)


def from_json(data: str | dict) -> Plan:
    """Build a Plan from a JSON string or a pre-parsed JSON dict.

    The dict input is for callers that already parsed the JSON themselves
    (e.g. `planner.py` uses `extract_json_object` to strip a code fence
    and parse). Everyone else should pass the raw string.
    """
    d = json.loads(data) if isinstance(data, str) else data
    if "branches" not in d:
        raise ValueError(f"Plan dict missing 'branches': {d!r}")
    branches: list[Branch] = []
    for bd in d["branches"]:
        kind = bd.get("kind")
        if kind == "retrieve":
            branches.append(RetrieveBranch(
                key=bd["key"],
                period=bd["period"],
                visual_only=bool(bd.get("visual_only", False)),
                value_kind=bd.get("value_kind") or None,
            ))
        elif kind == "lookup_external":
            branches.append(LookupBranch(target=bd["target"], src=bd.get("src") or None))
        else:
            raise ValueError(f"Unknown branch kind: {kind!r}")
    comp = d.get("computation") or {}
    pres = d.get("presentation") or {}
    raw_qualifiers = comp.get("qualifiers") or []
    if not isinstance(raw_qualifiers, list):
        raise ValueError(f"'computation.qualifiers' must be a list, got {raw_qualifiers!r}")
    return Plan(
        branches=branches,
        task=str(comp.get("task") or ""),
        qualifiers=[str(q) for q in raw_qualifiers if q is not None],
        units_out=pres.get("units_out") or None,
        precision=pres.get("precision"),
        answer_form=str(pres.get("answer_form") or "scalar"),
    )

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate(plan: Plan) -> ValidationResult:
    """Validate the plan's structure + a handful of cheap field-level checks.

    Catches semantic issues that survive parsing:
      - at least one branch
      - retrieve.key / lookup_external.target non-empty (after strip)
      - retrieve.value_kind in {scalar, vector, table} when set
      - precision >= 0 when set
    """
    errors: list[str] = []
    if not plan.branches:
        errors.append("Plan has no branches")
    for j, b in enumerate(plan.branches):
        if isinstance(b, RetrieveBranch):
            if not b.key.strip():
                errors.append(f"branches[{j}]: retrieve.key is empty")
            if b.value_kind and b.value_kind not in VALUE_KIND_VOCAB:
                errors.append(
                    f"branches[{j}]: value_kind {b.value_kind!r} "
                    f"not in {sorted(VALUE_KIND_VOCAB)}"
                )
        elif isinstance(b, LookupBranch):
            if not b.target.strip():
                errors.append(f"branches[{j}]: lookup_external.target is empty")
    if plan.precision is not None and plan.precision < 0:
        errors.append(f"precision must be >= 0, got {plan.precision}")
    return ValidationResult(ok=not errors, errors=errors)


# ---------------------------------------------------------------------------
# Planner: Question → Plan via a single LLM call.
#
# When the orchestrator detects MissingData during execution, it appends a
# one-shot recovery lesson to `ctx.prompt_overrides` (targeting "planner")
# and re-invokes `PlannerExecutor.plan` for a fresh attempt.
# ---------------------------------------------------------------------------

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
              once retrieval has returned relevant data. Preserve
              qualifier words from the question VERBATIM — do not
              paraphrase, simplify, or drop them. The compute step
              relies on the exact wording to choose the right operation.

  qualifiers  Optional list of short phrases that nail down a calculation
              choice the task sentence leaves underdetermined. Each
              qualifier is one phrase; the compute operator treats every
              qualifier as a MUST. Use only when the question explicitly
              pins a choice. null or empty list when the task is
              unambiguous.

presentation fields:
  How the final answer is rendered. No semantic content about the calculation itself.

  units_out    Unit of the final answer. Three rules, in order:

               (1) If the question NAMES a unit anywhere ("in millions of
                   dollars", "as a percent", "in DEM", "how many months"),
                   use the matching unit token. Examples drawn from real
                   questions:
                     "in millions of dollars" / "millions of nominal dollars"
                                                         → "usd_millions"
                     "in billions of yen"                → "jpy_billions"
                     "in 1962 dollars" / "in 2020 dollars"
                                                         → "usd_millions" (or matching scale)
                     "percentage points" / "as a percent" / "yield … in %"
                                                         → "pct"
                     "how many calendar months"          → "count"
                     "ratio of X to Y"                   → "ratio"
                     "spot exchange rate … in DEM per USD"
                                                         → "dem_per_usd"
                     "in Deutsche marks" / "in DEM"      → "dem"
               (2) If the question does NOT name a unit, output `null`.
                   Do not guess; do not invent; do not fall back to
                   "text" or any other placeholder. `null` is the right
                   answer here.
               (3) "text" is reserved STRICTLY for entity-name / string
                   answers — e.g. "Which agency had the largest …" → an
                   agency name. NEVER use "text" for a numeric answer or
                   a list of numbers; if the answer is numeric, rules (1)
                   and (2) apply (a unit token or `null`).
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

Produce the Plan JSON. Output ONLY a JSON code block — no prose, no explanation.
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
            print(f"[planner] attempt {attempt + 1}/3", file=sys.stderr, flush=True)
            resp = ctx.llm_client.call(system_prompt, user_message, thinking_budget=-1, ctx=ctx)
            raw = resp.text
            try:
                plan_dict = extract_json_object(raw)
                p = from_json(plan_dict)
                if not isinstance(p, Plan):
                    raise ValueError(f"from_json returned {type(p).__name__}, expected Plan")
                result = validate(p)
                if not result.ok:
                    raise ValueError(f"Plan validation errors: {result.errors}")
                return p
            except Exception as e:
                attempt_errors.append(f"Attempt {attempt + 1}: {e}")
                last_raw = raw
                last_error = str(e)
                print(
                    f"[planner] attempt {attempt + 1} parse/validate failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )

        print(
            f"[planner] FAILED after {len(attempt_errors)} attempts",
            file=sys.stderr, flush=True,
        )
        raise StepFailed(
            "planner",
            f"Failed to produce valid Plan after {len(attempt_errors)} attempts: "
            + "; ".join(attempt_errors),
        )
