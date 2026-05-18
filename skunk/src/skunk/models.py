"""Cross-cutting runtime types.

The envelopes here are threaded between operators (`retrieve`, `extract`,
`lookup_external`, `compute`) and the orchestrator. They are not owned
by any single operator — `PageRef` is the canonical page coordinate,
`AnnotatedValue` is the data envelope every operator produces or
consumes, and `HarnessContext` is the runtime context bag carrying
config, the LLM client, prompt overrides, and the diagnostic event log.

Operator-local result types stay with their operator (e.g.
`MissingResult` / `CritiqueResult` in `compute.py`, `LookupResult` in
`lookup_external.py`). The planner's output contract (`Plan` and its
branch / computation / presentation submodels) stays in `plan.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from skunk.config import SkunkConfig
from skunk.prompt_overrides import PromptOverride

if TYPE_CHECKING:
    from skunk.common import LLMClient


# ---------------------------------------------------------------------------
# Page coordinate
# ---------------------------------------------------------------------------

@dataclass
class PageRef:
    month: str | None = None        # "YYYY-MM"
    page: int | None = None         # 1-based PDF page index (canonical)

    @property
    def year(self) -> int | None:
        return int(self.month[:4]) if self.month else None

    def __post_init__(self) -> None:
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


# ---------------------------------------------------------------------------
# Annotated value — the data envelope every operator emits / consumes
# ---------------------------------------------------------------------------

VALUE_KIND_VOCAB: frozenset[str] = frozenset({"scalar", "vector", "table"})


class AnnotatedValue(BaseModel):
    """One described, annotated datum. Carries the payload + minimal metadata.

    Fields:
      - description: free-form natural-language label that uniquely
        distinguishes this entry from siblings. Should include everything a
        reader needs to know about what this value represents — series,
        period, sub-category, unit qualifier, etc.
      - value: payload; shape determined by `kind`.
      - unit: semantic unit token (e.g. "usd_millions", "pct", "year").
      - kind: "scalar" | "vector" | "table".
      - index_name: vector only — name of the varying dim (e.g. "month").
      - row_name / col_name: table only — names of the two varying dims.
      - tag: short machine-readable key (snake_case, like
        "gross_federal_debt:fy1973-fy1980"). Used by downstream consumers to
        select entries unambiguously when description substrings overlap.
        Two entries describing the same underlying series + period MUST
        share the same tag.
      - expected_index_range: vector only — short string like
        "1969-01..1980-01" describing the FULL index range the question
        requested. Lets compute flag gaps (actual vs expected). Default
        empty (no gap analysis).

    Payload shapes by kind (enforced at construction):
      - "scalar": value is int|float|str (or list of those for
                  lookup_external multi-value replies).
      - "vector": value is dict[str, int|float|str], keyed by index_name labels.
      - "table":  value is dict[str, dict[str, int|float|str]],
                  outer key = row_name label, inner key = col_name label.
    """
    model_config = ConfigDict(frozen=True)

    description: str
    value: Any
    unit: str = ""
    kind: Literal["scalar", "vector", "table"] = "scalar"
    index_name: str | None = None
    row_name: str | None = None
    col_name: str | None = None
    tag: str = ""
    expected_index_range: str = ""

    @model_validator(mode="after")
    def _check_shape(self) -> AnnotatedValue:
        # Vector/table cells (and scalar payloads) must be non-bool int/float/str.
        # bool is an int subclass in Python — exclude it explicitly.
        def is_prim(c: Any) -> bool:
            return not isinstance(c, bool) and isinstance(c, (int, float, str))

        v = self.value
        if self.kind == "scalar":
            # scalar payload: primitive OR list of primitives (the latter for
            # lookup_external multi-value replies).
            ok = is_prim(v) or (isinstance(v, list) and all(is_prim(c) for c in v))
            if not ok:
                raise ValueError(
                    f"kind=scalar requires int|float|str (or list of those), "
                    f"got {type(v).__name__}"
                )
        elif self.kind == "vector":
            # vector payload: flat dict[str, primitive] — no nesting.
            if not (isinstance(v, dict) and all(
                isinstance(k, str) and is_prim(c) for k, c in v.items()
            )):
                raise ValueError("vector value must be flat dict[str, scalar]")
            if not self.index_name:
                raise ValueError("vector entry missing non-empty 'index_name'")
        else:  # table
            # table payload: 2-level dict[str, dict[str, primitive]]. Ragged
            # column sets allowed; no-nesting invariant holds regardless.
            if not (isinstance(v, dict) and all(
                isinstance(r, str)
                and isinstance(row, dict)
                and all(isinstance(k, str) and is_prim(c) for k, c in row.items())
                for r, row in v.items()
            )):
                raise ValueError(
                    "table value must be dict[str, dict[str, scalar]] (no nesting)"
                )
            if not self.row_name or not self.col_name:
                raise ValueError("table entry missing non-empty 'row_name'/'col_name'")
        return self


# ---------------------------------------------------------------------------
# Harness context — runtime envelope threaded through every operator
# ---------------------------------------------------------------------------

# `LLMClient` lives in `skunk.common`; the typing-only import above and the
# deferred import inside `__post_init__` together avoid an import cycle
# (common.py imports SkunkConfig from config.py, and we want config.py's
# TYPE_CHECKING import of PageRef to point at this module).

@dataclass
class HarnessContext:
    question: str
    verbose: bool = False     # live-print orchestrator + operator events to stdout
    events: list[dict] = field(default_factory=list)  # per-question diagnostic events
    config: SkunkConfig = field(default_factory=SkunkConfig.from_env)
    llm_client: LLMClient | None = None  # inject a mock for tests; auto-created otherwise
    prompt_overrides: tuple[PromptOverride, ...] = ()  # corpus/few_shot/lesson overrides; operators pick out their own entries by name

    def __post_init__(self) -> None:
        if self.llm_client is None:
            from skunk.common import LLMClient
            self.llm_client = LLMClient(self.config)

    def emit(self, source: str, message: str, **fields: Any) -> None:
        """Record a diagnostic event. Operators call this with their op name as `source`."""
        evt = {"source": source, "message": message, **fields}
        self.events.append(evt)
        if self.verbose:
            extra = ""
            if fields:
                bits = []
                for k, v in fields.items():
                    s = repr(v)
                    if len(s) > 200:
                        s = s[:200] + "..."
                    bits.append(f"{k}={s}")
                extra = " | " + ", ".join(bits)
            print(f"  [{source}] {message}{extra}")

    def with_extra_override(self, override: PromptOverride) -> HarnessContext:
        """Return a shallow copy of this ctx with one extra `PromptOverride`
        appended to `prompt_overrides`. Used by the orchestrator to scope a
        recovery lesson to a single planner re-invocation without polluting
        the ctx that operators see."""
        import dataclasses
        return dataclasses.replace(
            self, prompt_overrides=self.prompt_overrides + (override,)
        )
