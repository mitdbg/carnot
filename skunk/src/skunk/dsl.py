"""DSL — Plan/Branch/ComputeNode types, text<->AST parser, validator, serializer.

Surface forms accepted:

  retrieve(concept='X', period='CY1940') --> extract() --> compute()
  retrieve(concept='X', period='1990-09') --> extract(visual_only=True) --> compute()
  lookup_external(nl='CPI-U for 1953') --> compute()
  [ retrieve(...) --> extract() ; retrieve(...) --> extract() ] --> compute()
  [ retrieve(...) --> extract() ; lookup_external(...) ] --> compute()

  Decomposed (multi-compute):
  [
    [ branches_1 ] --> compute(task='...');
    [ branches_2 ] --> compute(task='...')
  ] --> compute()

Inner `[branches] --> compute(task='...')` blocks are intermediate computes
that emit list[AnnotatedValue] for the trailing aggregator `compute()` to
consume. Depth is bounded by `SkunkConfig.max_compute_depth` (default 2).

Page number convention:
  PageRef.page = 1-based PDF page index (canonical throughout the codebase).
  The bulletin's printed-page footer is recoverable via
  skunk.subagents.extract.get_printed_page() for trace/prompt enrichment,
  but is never used as a lookup key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Inter-op data model
# ---------------------------------------------------------------------------

@dataclass
class PageRef:
    month: str | None = None        # "YYYY-MM"
    page: int | None = None         # 1-based PDF page index (canonical)
    file_path: str | None = None    # resolved by manifest

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
        if self.file_path:
            parts.append(f"file={self.file_path!r}")
        return f"PageRef({', '.join(parts)})"


@dataclass
class DocHandle:
    refs: list[PageRef] = field(default_factory=list)
    desc: str = ""


VALID_KINDS = frozenset({"scalar", "vector", "table"})


@dataclass
class AnnotatedValue:
    """One described, annotated datum. Carries the payload + minimal metadata.

    Fields:
      - description: free-form natural-language label that uniquely distinguishes
        this entry from siblings. Should include everything a reader needs to
        know about what this value represents — series, period, sub-category,
        unit hint, etc. There is no separate `dims` / `quote` field; rich
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


@dataclass
class FormattedString:
    text: str


# Union type for inter-op values
OpOutput = DocHandle | list[AnnotatedValue] | FormattedString


# ---------------------------------------------------------------------------
# Subagent dispatch envelope
# ---------------------------------------------------------------------------
# Not an AST node — just a small (name, args) bag the orchestrator passes to
# subagent.run(op, prev, ctx). Subagents read op.args[...] by key.

VALID_OPS = frozenset({"retrieve", "extract", "lookup_external", "compute"})


@dataclass
class OpNode:
    op: str
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"Unknown op: {self.op!r}. Must be one of {sorted(VALID_OPS)}")


# ---------------------------------------------------------------------------
# AST: flat Plan with branches
# ---------------------------------------------------------------------------

@dataclass
class RetrieveBranch:
    concept: str
    period: str
    visual_only: bool = False       # passes through to the implicit extract step
    period_type: str | None = None  # hint for the retriever: FY | CY | month | fiscal_quarter | calendar_quarter | specific_date | mixed
    # Advisory hints for the extract subagent (planning-time decisions about
    # the heavyweight stage):
    value_kind: str | None = None   # scalar | vector | table — expected shape of extracted values
    index_name: str | None = None   # axis label when value_kind="vector" (e.g. "fiscal_year", "bureau")


@dataclass
class LookupBranch:
    nl: str


Branch = RetrieveBranch | LookupBranch


@dataclass
class ComputeNode:
    """One compute step in a Plan's chain.

    Intermediate nodes (final=False) have non-empty `branches` and a non-empty
    `task`. They run their branches to produce a list[AnnotatedValue], then
    invoke the compute subagent in intermediate mode to derive their own
    list[AnnotatedValue] result for the downstream aggregator to consume.

    The final node (final=True) has empty `branches` when it follows
    intermediates (it aggregates their outputs), or non-empty `branches` in the
    legacy single-compute shape (it consumes data directly).
    """
    branches: list[Branch] = field(default_factory=list)
    task: str = ""
    final: bool = False
    method: str | None = None                # named method/growth key (cagr, yoy_growth, geometric_mean, ...)
    transforms: list[str] = field(default_factory=list)  # orthogonal transforms (log_of, per_capita, midpoint_normalized, ...)


@dataclass
class Plan:
    """Chain of one or more ComputeNodes terminating in a final aggregator.

    Legacy flat plans = `Plan(computes=[ComputeNode(branches=[...], final=True)])`.

    Plan-level fields describe the FINAL answer's shape — they apply to the
    terminal aggregator regardless of how many sub-computes precede it.
    """
    computes: list[ComputeNode] = field(default_factory=list)
    units_out: str | None = None             # usd, usd_millions, pct, year, count, rate, ...
    precision: int | None = None             # decimal places to round the final answer to
    answer_form: str = "scalar"              # scalar | bracketed_list | labeled_pair | string

    # ---- Back-compat read accessor ----------------------------------------
    # Older callers iterate plan.branches. For the legacy single-compute shape
    # (len(computes) == 1, final=True), expose the branches directly so tests
    # and external code that pre-date the chain change keep working without
    # a wider refactor.
    @property
    def branches(self) -> list[Branch]:
        if len(self.computes) == 1 and self.computes[0].final:
            return self.computes[0].branches
        # Otherwise: return all data branches across intermediates (the flat
        # view doesn't capture chain structure, so this is best-effort).
        out: list[Branch] = []
        for c in self.computes:
            out.extend(c.branches)
        return out


# ---------------------------------------------------------------------------
# Text DSL parser
# ---------------------------------------------------------------------------

class ParseError(ValueError):
    pass


def _split_chain(text: str) -> list[str]:
    """Split `text` on '-->' respecting bracket depth."""
    tokens: list[str] = []
    depth = 0
    current: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif depth == 0 and text[i:i+3] == "-->":
            tokens.append("".join(current).strip())
            current = []
            i += 3
            continue
        else:
            current.append(ch)
        i += 1
    if depth != 0:
        raise ParseError(f"Unbalanced brackets in DSL: depth={depth} at end of input")
    if current:
        tokens.append("".join(current).strip())
    return [t for t in tokens if t]


def _split_branches(text: str) -> list[str]:
    """Split `text` on ';' respecting bracket depth."""
    branches: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch == ";":
            branches.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if depth != 0:
        raise ParseError(f"Unbalanced brackets in parallel branches: depth={depth} at end of input")
    if current:
        branches.append("".join(current).strip())
    return [b for b in branches if b]


_OP_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)")

# Per-op positional arg keys. The first bare value at index N is assigned to this
# op's positional_keys[N]. Cached plans (data/dsl_planning_pass.csv) use bare form
# for the primary arg of retrieve and lookup_external.
_POSITIONAL_KEYS_BY_OP: dict[str, list[str]] = {
    "retrieve": ["concept"],
    "lookup_external": ["nl"],
    "extract": [],
    "compute": [],
}


def _parse_op(token: str) -> tuple[str, dict[str, Any]]:
    """Split `op_name(args)` into (op_name, args_dict)."""
    m = _OP_RE.match(token.strip())
    if not m:
        raise ParseError(f"Cannot parse op: {token!r}")
    op_name = m.group(1)
    return op_name, _parse_args(m.group(2), op_name)


def _parse_args(raw: str, op_name: str = "") -> dict[str, Any]:
    """Parse `key=val, key2=val2` (or bare positional) into a dict. Strings may be quoted."""
    raw = raw.strip()
    if not raw:
        return {}
    args: dict[str, Any] = {}
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    in_quote = False
    quote_char = ""
    for ch in raw:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
        elif ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())

    positional_keys = _POSITIONAL_KEYS_BY_OP.get(op_name, [])
    positional_idx = 0
    for part in parts:
        part = part.strip()
        m = _KV_RE.match(part)
        if m:
            args[m.group(1).strip()] = _coerce(m.group(2).strip())
        else:
            if positional_idx >= len(positional_keys):
                raise ParseError(f"Unexpected positional arg {part!r} for op {op_name!r}")
            args[positional_keys[positional_idx]] = _coerce(part)
            positional_idx += 1
    return args


def _split_top_level(raw: str, sep: str = ",") -> list[str]:
    """Split `raw` on `sep` at bracket/quote depth 0 — used for list literals
    and arg lists."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    for ch in raw:
        if in_quote:
            current.append(ch)
            if ch == quote_char:
                in_quote = False
        elif ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            current.append(ch)
        elif ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif depth == 0 and ch == sep:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _coerce(v: str) -> Any:
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_coerce(item) for item in _split_top_level(inner, ",")]
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    if v == "True":
        return True
    if v == "False":
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


_PLAN_LEVEL_KWARGS = frozenset({"units_out", "precision", "answer_form"})
_FINAL_COMPUTE_KWARGS = frozenset({"method", "transforms"}) | _PLAN_LEVEL_KWARGS


def parse(text: str) -> Plan:
    """Parse text DSL into a Plan.

    Accepts:
      Flat (single compute):
        retrieve(...) --> extract(...) --> compute()
        lookup_external(...) --> compute()
        [ branch ; branch ; ... ] --> compute()

      Decomposed (chain of computes):
        [ [branches] --> compute(task='...'); [branches] --> compute(task='...') ] --> compute()

    The final `compute(...)` may carry constraint kwargs:
      - Plan-level (answer-shape): units_out, precision, answer_form
      - Final-compute (computation): method, transforms
    Intermediate `compute(task='...')` may additionally carry method, transforms.
    """
    text = text.strip()
    parts = _split_chain(text)
    if not parts:
        raise ParseError(f"Empty plan text: {text!r}")
    last = parts[-1].strip()
    if last.startswith("[") and last.endswith("]"):
        raise ParseError(f"Plan must end with compute(), got parallel block: {last!r}")
    last_name, last_args = _parse_op(last)
    if last_name != "compute":
        raise ParseError(f"Plan must end with compute(), got {parts[-1]!r}")
    unknown = set(last_args.keys()) - _FINAL_COMPUTE_KWARGS
    if unknown:
        raise ParseError(
            f"final compute() got unexpected kwargs {sorted(unknown)}; "
            f"allowed: {sorted(_FINAL_COMPUTE_KWARGS)}"
        )
    plan_units = last_args.get("units_out")
    plan_precision = last_args.get("precision")
    plan_answer_form = last_args.get("answer_form")
    final_method, final_transforms, _ = _split_compute_kwargs(
        last_args, allowed=_FINAL_COMPUTE_KWARGS,
    )

    def _wrap(computes: list[ComputeNode]) -> Plan:
        return Plan(
            computes=computes,
            units_out=str(plan_units) if plan_units else None,
            precision=int(plan_precision) if plan_precision is not None else None,
            answer_form=str(plan_answer_form) if plan_answer_form else "scalar",
        )

    body = parts[:-1]
    if not body:
        raise ParseError(f"Plan must have at least one data-gathering step before compute(): {text!r}")

    # Detect decomposed form: a single outer bracket block whose first inner
    # member is itself bracketed (an inner [branches] --> compute(task=...)).
    if len(body) == 1 and body[0].startswith("[") and body[0].endswith("]"):
        inner = body[0][1:-1].strip()
        raw_members = _split_branches(inner)
        if len(raw_members) < 2:
            raise ParseError(
                f"Parallel block must have >=2 members, got {len(raw_members)}"
            )
        # Decomposed: every member starts with '['.
        if all(m.lstrip().startswith("[") for m in raw_members):
            intermediates = [_parse_intermediate_compute(m) for m in raw_members]
            return _wrap(
                [*intermediates, ComputeNode(
                    branches=[], task="", final=True,
                    method=final_method, transforms=final_transforms,
                )],
            )
        # Otherwise legacy parallel-of-branches form.
        if any(m.lstrip().startswith("[") for m in raw_members):
            raise ParseError(
                "Parallel block mixes plain branches with sub-compute brackets — "
                "either all members are branches, or all are `[branches] --> compute(task='...')`"
            )
        branches = [_parse_branch(b) for b in raw_members]
        return _wrap([ComputeNode(
            branches=branches, task="", final=True,
            method=final_method, transforms=final_transforms,
        )])

    branches = [_parse_branch(" --> ".join(body))]
    return _wrap([ComputeNode(
        branches=branches, task="", final=True,
        method=final_method, transforms=final_transforms,
    )])


def _parse_intermediate_compute(text: str) -> ComputeNode:
    """Parse a single `[branches] --> compute(task='...', method?='...', transforms?=[...])` block."""
    text = text.strip()
    parts = _split_chain(text)
    if len(parts) < 2:
        raise ParseError(
            f"Intermediate compute must be `[branches] --> compute(task='...')`, got: {text!r}"
        )
    last_name, last_args = _parse_op(parts[-1])
    if last_name != "compute":
        raise ParseError(f"Intermediate must end with compute(...), got {parts[-1]!r}")
    task = str(last_args.get("task", "")).strip()
    if not task:
        raise ParseError(
            f"Intermediate compute requires non-empty task='...': {text!r}"
        )
    method, transforms, unknown = _split_compute_kwargs(last_args, allowed={"task", "method", "transforms"})
    if unknown:
        raise ParseError(
            f"Intermediate compute got unexpected kwargs {sorted(unknown)}; "
            "allowed: task, method, transforms"
        )
    body = parts[:-1]
    if len(body) == 1 and body[0].startswith("[") and body[0].endswith("]"):
        inner = body[0][1:-1].strip()
        raw_branches = _split_branches(inner)
        if len(raw_branches) < 1:
            raise ParseError(f"Intermediate compute has no branches: {text!r}")
        branches = [_parse_branch(b) for b in raw_branches]
    else:
        branches = [_parse_branch(" --> ".join(body))]
    return ComputeNode(
        branches=branches, task=task, final=False,
        method=method, transforms=transforms,
    )


def _split_compute_kwargs(
    args: dict[str, Any], allowed: set[str],
) -> tuple[str | None, list[str], set[str]]:
    """Pull `method` and `transforms` out of a compute() args dict.
    Returns (method, transforms, unknown_keys_outside_allowed)."""
    method = args.get("method")
    method_val = str(method).strip() if method else None
    raw_transforms = args.get("transforms") or []
    if not isinstance(raw_transforms, list):
        raise ParseError(f"compute.transforms must be a list, got {raw_transforms!r}")
    transforms = [str(t) for t in raw_transforms]
    unknown = set(args.keys()) - allowed
    return method_val or None, transforms, unknown


def _parse_branch(text: str) -> Branch:
    """Parse a single branch: either `retrieve(...) --> extract(...)` or `lookup_external(...)`."""
    parts = _split_chain(text)
    if not parts:
        raise ParseError(f"Empty branch: {text!r}")
    head_name, head_args = _parse_op(parts[0])

    if head_name == "lookup_external":
        if len(parts) != 1:
            raise ParseError(f"lookup_external branch must be single op, got: {text!r}")
        if "nl" not in head_args:
            raise ParseError(f"lookup_external requires 'nl' arg: {text!r}")
        return LookupBranch(nl=str(head_args["nl"]))

    if head_name == "retrieve":
        if len(parts) != 2:
            raise ParseError(f"retrieve branch must be `retrieve --> extract`, got: {text!r}")
        ext_name, ext_args = _parse_op(parts[1])
        if ext_name != "extract":
            raise ParseError(f"retrieve must be followed by extract, got {ext_name!r}: {text!r}")
        if "concept" not in head_args or "period" not in head_args:
            raise ParseError(f"retrieve requires 'concept' and 'period': {text!r}")
        # Legacy plan caches may still include `source_bulletin='...'`; the
        # parser accepts it (via the generic kwarg path) and we silently drop
        # it here. v0.6 removed the field end-to-end.
        pt = head_args.get("period_type")
        vk = ext_args.get("value_kind")
        ix = ext_args.get("index_name")
        return RetrieveBranch(
            concept=str(head_args["concept"]),
            period=str(head_args["period"]),
            visual_only=bool(ext_args.get("visual_only", False)),
            period_type=str(pt) if pt else None,
            value_kind=str(vk) if vk else None,
            index_name=str(ix) if ix else None,
        )

    raise ParseError(f"Branch must start with retrieve or lookup_external, got {head_name!r}: {text!r}")


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def _fmt_string_list(items: list[str]) -> str:
    return "[" + ", ".join(f"'{x}'" for x in items) + "]"


def _serialize_branch(b: Branch) -> str:
    if isinstance(b, RetrieveBranch):
        args = [f"concept='{b.concept}'", f"period='{b.period}'"]
        if b.period_type:
            args.append(f"period_type='{b.period_type}'")
        ext_args: list[str] = []
        if b.value_kind:
            ext_args.append(f"value_kind='{b.value_kind}'")
        if b.index_name:
            ext_args.append(f"index_name='{b.index_name}'")
        if b.visual_only:
            ext_args.append("visual_only=True")
        ext = f"extract({', '.join(ext_args)})" if ext_args else "extract()"
        return f"retrieve({', '.join(args)}) --> {ext}"
    if isinstance(b, LookupBranch):
        return f"lookup_external(nl='{b.nl}')"
    raise TypeError(f"Unknown branch type: {type(b)}")


def _serialize_branches_block(branches: list[Branch]) -> str:
    if len(branches) == 1:
        return _serialize_branch(branches[0])
    return "[ " + " ; ".join(_serialize_branch(b) for b in branches) + " ]"


def _serialize_final_compute(plan: Plan, final: ComputeNode) -> str:
    """Render the trailing `compute(...)`. Plan-level constraints
    (units_out/precision/answer_form) and final-compute fields
    (method/transforms) live together on this op."""
    args: list[str] = []
    if plan.units_out:
        args.append(f"units_out='{plan.units_out}'")
    if plan.precision is not None:
        args.append(f"precision={plan.precision}")
    if plan.answer_form and plan.answer_form != "scalar":
        args.append(f"answer_form='{plan.answer_form}'")
    if final.method:
        args.append(f"method='{final.method}'")
    if final.transforms:
        args.append(f"transforms={_fmt_string_list(final.transforms)}")
    return f"compute({', '.join(args)})" if args else "compute()"


def _serialize_intermediate_compute(c: ComputeNode) -> str:
    args = [f"task='{c.task}'"]
    if c.method:
        args.append(f"method='{c.method}'")
    if c.transforms:
        args.append(f"transforms={_fmt_string_list(c.transforms)}")
    return f"compute({', '.join(args)})"


def serialize(plan: Plan) -> str:
    """Serialize Plan back to text DSL.

    Flat shape (single final compute with branches):
        [branches] --> compute([constraints])
    Decomposed (intermediates + aggregator):
        [ [branches] --> compute(task='...'[, method=, transforms=]); ... ] --> compute([constraints])
    """
    if not plan.computes:
        raise ValueError("Cannot serialize empty Plan")
    if len(plan.computes) == 1:
        c = plan.computes[0]
        if not c.final:
            raise ValueError("Single-compute Plan must have final=True")
        if not c.branches:
            raise ValueError("Single-compute Plan must have non-empty branches")
        return f"{_serialize_branches_block(c.branches)} --> {_serialize_final_compute(plan, c)}"
    # Decomposed: validate shape lightly, then emit.
    intermediates = plan.computes[:-1]
    final = plan.computes[-1]
    if not final.final or final.branches:
        raise ValueError("Decomposed Plan must end with a final aggregator with empty branches")
    if any(c.final or not c.branches or not c.task for c in intermediates):
        raise ValueError("Each intermediate compute needs non-empty branches and task, final=False")
    members = []
    for c in intermediates:
        # Always bracket the inner branches block so re-parsing distinguishes
        # decomposed members from a plain branch.
        if len(c.branches) == 1:
            inner = "[ " + _serialize_branch(c.branches[0]) + " ]"
        else:
            inner = _serialize_branches_block(c.branches)
        members.append(f"{inner} --> {_serialize_intermediate_compute(c)}")
    return "[ " + " ; ".join(members) + f" ] --> {_serialize_final_compute(plan, final)}"


# ---------------------------------------------------------------------------
# JSON serialization (for storage / LLM output)
# ---------------------------------------------------------------------------

def _branch_to_dict(b: Branch) -> dict:
    if isinstance(b, RetrieveBranch):
        d: dict = {"kind": "retrieve", "concept": b.concept, "period": b.period}
        if b.visual_only:
            d["visual_only"] = True
        if b.period_type:
            d["period_type"] = b.period_type
        if b.value_kind:
            d["value_kind"] = b.value_kind
        if b.index_name:
            d["index_name"] = b.index_name
        return d
    if isinstance(b, LookupBranch):
        return {"kind": "lookup_external", "nl": b.nl}
    raise TypeError(f"Unknown branch type: {type(b)}")


def _branch_from_dict(d: dict) -> Branch:
    kind = d.get("kind")
    if kind == "retrieve":
        # Legacy cached plans may still carry `source_bulletin`; silently
        # ignore it — v0.6 removed the field end-to-end.
        return RetrieveBranch(
            concept=d["concept"],
            period=d["period"],
            visual_only=bool(d.get("visual_only", False)),
            period_type=d.get("period_type") or None,
            value_kind=d.get("value_kind") or None,
            index_name=d.get("index_name") or None,
        )
    if kind == "lookup_external":
        return LookupBranch(nl=d["nl"])
    raise ValueError(f"Unknown branch kind: {kind!r}")


def _is_legacy_flat(plan: Plan) -> bool:
    return (
        len(plan.computes) == 1
        and plan.computes[0].final
        and plan.computes[0].task == ""
        and not plan.computes[0].method
        and not plan.computes[0].transforms
        and bool(plan.computes[0].branches)
    )


def _plan_meta(plan: Plan) -> dict:
    """Top-level constraint fields, emitted only when non-default."""
    meta: dict = {}
    if plan.units_out:
        meta["units_out"] = plan.units_out
    if plan.precision is not None:
        meta["precision"] = plan.precision
    if plan.answer_form != "scalar":
        meta["answer_form"] = plan.answer_form
    return meta


def _compute_to_dict(c: ComputeNode) -> dict:
    entry: dict = {"task": c.task, "branches": [_branch_to_dict(b) for b in c.branches]}
    if c.final:
        entry["final"] = True
    if c.method:
        entry["method"] = c.method
    if c.transforms:
        entry["transforms"] = list(c.transforms)
    return entry


def _compute_from_dict(entry: dict) -> ComputeNode:
    return ComputeNode(
        branches=[_branch_from_dict(b) for b in entry.get("branches", [])],
        task=str(entry.get("task", "")),
        final=bool(entry.get("final", False)),
        method=entry.get("method") or None,
        transforms=list(entry.get("transforms") or []),
    )


def to_dict(plan: Plan) -> dict:
    """Serialize to dict. Emits legacy `{"branches": [...]}` shape when the
    plan is a single unnamed final compute with no constraints (back-compat
    with cached CSVs); otherwise emits the explicit `{"computes": [...]}` shape.
    """
    meta = _plan_meta(plan)
    if _is_legacy_flat(plan) and not meta:
        return {"branches": [_branch_to_dict(b) for b in plan.computes[0].branches]}
    out: dict = {"computes": [_compute_to_dict(c) for c in plan.computes]}
    out.update(meta)
    return out


def from_dict(d: dict) -> Plan:
    """Build a Plan from dict. Accepts both legacy and new shapes:

    Legacy: `{"branches": [...]}` → single final compute with those branches.
    New:    `{"computes": [{"task": "...", "branches": [...], "final"?: bool,
                            "method"?: str, "transforms"?: list[str]}, ...],
             "units_out"?: str, "precision"?: int, "answer_form"?: str}`
            — if no entry has `final=True`, append an empty-branches final aggregator.
    """
    if "computes" in d:
        raw = d["computes"]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Plan 'computes' must be a non-empty list: {d!r}")
        computes = [_compute_from_dict(entry) for entry in raw]
        if not any(c.final for c in computes):
            computes.append(ComputeNode(branches=[], task="", final=True))
        return Plan(
            computes=computes,
            units_out=d.get("units_out") or None,
            precision=d.get("precision"),
            answer_form=str(d.get("answer_form") or "scalar"),
        )
    if "branches" in d:
        branches = [_branch_from_dict(b) for b in d["branches"]]
        return Plan(
            computes=[ComputeNode(
                branches=branches,
                task="",
                final=True,
                method=d.get("method") or None,
                transforms=list(d.get("transforms") or []),
            )],
            units_out=d.get("units_out") or None,
            precision=d.get("precision"),
            answer_form=str(d.get("answer_form") or "scalar"),
        )
    raise ValueError(f"Plan dict missing required field 'branches' or 'computes': {d!r}")


# ---------------------------------------------------------------------------
# Validator — type system enforces structure; only period grammar remains.
# ---------------------------------------------------------------------------

# Period grammar: point | range (point..point) | enumeration (point,point,...)
# point: CY/FY year, Qn quarter, YYYY-MM-DD, YYYY-MM, plain YYYY
_PERIOD_POINT = r"(?:CY\d{4}|FY\d{4}|Q[1-4]-\d{4}|\d{4}-\d{2}-\d{2}|\d{4}-\d{2}|\d{4})"
_PERIOD_RE = re.compile(
    r"^(?:"
    + _PERIOD_POINT + r"\.\." + _PERIOD_POINT       # range
    + r"|" + _PERIOD_POINT + r"(?:," + _PERIOD_POINT + r")+"  # enumeration
    + r"|" + _PERIOD_POINT                          # single point
    + r")$"
)

# ---------------------------------------------------------------------------
# Controlled vocabularies for the constraint fields (shared with the planner
# prompt — keep these in sync with planner.py's DSL spec section).
# ---------------------------------------------------------------------------

UNITS_OUT_VOCAB: frozenset[str] = frozenset({
    "usd", "usd_thousands", "usd_millions", "usd_billions",
    "pct", "count", "year", "rate", "fx_rate", "ratio", "text",
})

ANSWER_FORM_VOCAB: frozenset[str] = frozenset({
    "scalar", "bracketed_list", "labeled_pair", "string",
})

METHOD_VOCAB: frozenset[str] = frozenset({
    "cagr", "yoy_growth", "mom_growth",
    "geometric_mean", "arithmetic_mean",
    "ols_regression",
    "pearson_correlation", "partial_correlation", "spearman",
    "mad", "gini", "theil", "cv", "iqr", "h_spread",
    "expected_shortfall", "arc_elasticity", "point_elasticity",
    "zipf", "hp_filter", "rolling_average",
    "hazen_plotting_position",
})

TRANSFORMS_VOCAB: frozenset[str] = frozenset({
    "log_of", "ratio_of", "per_capita", "midpoint_normalized",
    "weighted", "inflation_adjusted", "absolute_value",
})

PERIOD_TYPE_VOCAB: frozenset[str] = frozenset({
    "FY", "CY", "month", "fiscal_quarter", "calendar_quarter",
    "specific_date", "mixed",
})

VALUE_KIND_VOCAB: frozenset[str] = frozenset({"scalar", "vector", "table"})


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate(plan: Plan, max_compute_depth: int = 2) -> ValidationResult:
    """Validate Plan structure + period grammar.

    Structural invariants:
      - At least one ComputeNode.
      - Exactly one final=True, and it is the last node.
      - len(plan.computes) <= max_compute_depth.
      - Single-compute (legacy) plan must have non-empty branches.
      - Multi-compute plan: intermediates need non-empty branches AND non-empty
        task; the final aggregator must have empty branches.
      - Every RetrieveBranch period matches the period grammar.
    """
    errors: list[str] = []
    if not plan.computes:
        errors.append("Plan has no computes")
        return ValidationResult(ok=False, errors=errors)

    final_idx = [i for i, c in enumerate(plan.computes) if c.final]
    if len(final_idx) != 1:
        errors.append(f"Plan must have exactly one final=True compute, got {len(final_idx)}")
    elif final_idx[0] != len(plan.computes) - 1:
        errors.append("Final compute must be the last in plan.computes")

    # Depth = number of compute LAYERS, not node count. v1 supports at most one
    # sub-compute layer feeding a final aggregator, so:
    #   - depth 1 when there's only a final compute (legacy flat),
    #   - depth 2 when there are intermediates + final.
    # (Nested intermediates aren't representable in the current AST.)
    depth = 1 if len(plan.computes) == 1 else 2
    if depth > max_compute_depth:
        errors.append(
            f"Plan depth {depth} exceeds max_compute_depth={max_compute_depth}"
        )

    # Plan-level constraint fields against controlled vocab.
    if plan.units_out and plan.units_out not in UNITS_OUT_VOCAB:
        errors.append(f"Plan.units_out {plan.units_out!r} not in controlled vocab {sorted(UNITS_OUT_VOCAB)}")
    if plan.answer_form not in ANSWER_FORM_VOCAB:
        errors.append(f"Plan.answer_form {plan.answer_form!r} not in {sorted(ANSWER_FORM_VOCAB)}")

    is_legacy = len(plan.computes) == 1
    for i, c in enumerate(plan.computes):
        if is_legacy:
            if not c.branches:
                errors.append(f"computes[{i}]: single-compute plan must have non-empty branches")
        else:
            if c.final:
                if c.branches:
                    errors.append(
                        f"computes[{i}]: final aggregator must have empty branches in a decomposed plan"
                    )
            else:
                if not c.branches:
                    errors.append(f"computes[{i}]: intermediate compute must have non-empty branches")
                if not c.task.strip():
                    errors.append(f"computes[{i}]: intermediate compute must have non-empty task")

        if c.method and c.method not in METHOD_VOCAB:
            errors.append(
                f"computes[{i}].method {c.method!r} not in controlled vocab {sorted(METHOD_VOCAB)}"
            )
        for t in c.transforms:
            if t not in TRANSFORMS_VOCAB:
                errors.append(
                    f"computes[{i}].transforms contains {t!r}, not in {sorted(TRANSFORMS_VOCAB)}"
                )
            if t == c.method:
                errors.append(
                    f"computes[{i}]: transform {t!r} duplicates method; transforms must be orthogonal to method"
                )

        for j, b in enumerate(c.branches):
            if isinstance(b, RetrieveBranch):
                if not _PERIOD_RE.match(b.period):
                    errors.append(
                        f"computes[{i}].branches[{j}]: period {b.period!r} does not match expected "
                        "format (CY/FY year, Qn-YYYY, YYYY-MM, YYYY-MM-DD, YYYY, range X..Y, "
                        "enumeration X,Y)"
                    )
                if b.period_type and b.period_type not in PERIOD_TYPE_VOCAB:
                    errors.append(
                        f"computes[{i}].branches[{j}]: period_type {b.period_type!r} "
                        f"not in {sorted(PERIOD_TYPE_VOCAB)}"
                    )
                if b.value_kind and b.value_kind not in VALUE_KIND_VOCAB:
                    errors.append(
                        f"computes[{i}].branches[{j}]: value_kind {b.value_kind!r} "
                        f"not in {sorted(VALUE_KIND_VOCAB)}"
                    )
                if b.index_name and b.value_kind != "vector":
                    errors.append(
                        f"computes[{i}].branches[{j}]: index_name is only meaningful "
                        f"when value_kind='vector' (got value_kind={b.value_kind!r})"
                    )
    return ValidationResult(ok=not errors, errors=errors)
