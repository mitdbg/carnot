"""Cross-cutting runtime types.

The envelopes here are threaded between operators (`retrieve`, `extract`,
`lookup_external`, `compute`) and the orchestrator. They are not owned
by any single operator — `PageRef` is the canonical page coordinate,
`AnnotatedValue` is the data envelope every operator produces or
consumes, and `HarnessContext` is the runtime context bag carrying
config, the LLM client, prompt overrides, and the diagnostic event log.

Operator-local result types stay with their operator (e.g.
`CritiqueResult` in `compute.py`, `LookupResult` in
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


VALUE_KIND_VOCAB: frozenset[str] = frozenset({"scalar", "vector", "table"})


class AnnotatedValue(BaseModel):
    """One described, annotated datum. Carries the payload + minimal metadata.

    Fields:
      - description: free-form natural-language label that uniquely
        distinguishes this entry from siblings. Should include everything a
        reader needs to know about what this value represents — series,
        period, sub-category, unit qualifier, etc.
      - value: payload; shape determined by `kind`.
      - unit: natural-language unit label (e.g. "millions of dollars",
        "percent", "year"); empty string when the value is not a
        measurement (e.g. a name or other string answer).
      - kind: "scalar" | "vector" | "table".
      - index_name: vector only — name of the varying dim (e.g. "month").
      - row_name / col_name: table only — names of the two varying dims.

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

    @property
    def frame(self):
        """Uniform pandas view of the payload, regardless of `kind`.

          scalar  →  1×1 DataFrame (or N×1 for list-form scalars from
                     lookup_external). Single column named after the description.
          vector  →  N×1 DataFrame; `index.name == self.index_name`,
                     single column named after the description.
          table   →  R×C DataFrame; `index.name == self.row_name`,
                     `columns.name == self.col_name`.

        Plain property, not cached — pydantic `frozen=True` blocks the
        dict mutation `cached_property` needs, and `prev` lists are small
        enough that rebuilding is cheap.
        """
        import pandas as pd
        v = self.value
        label = (self.description or "value").strip() or "value"
        if self.kind == "scalar":
            rows = list(v) if isinstance(v, list) else [v]
            return pd.DataFrame({label: rows})
        if self.kind == "vector":
            s = pd.Series(v, name=label, dtype=object if not v else None)
            df = s.to_frame()
            df.index.name = self.index_name
            return df
        df = pd.DataFrame.from_dict(v, orient="index")
        df.index.name = self.row_name
        df.columns.name = self.col_name
        return df


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
