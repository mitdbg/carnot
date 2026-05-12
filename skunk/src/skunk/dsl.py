"""DSL — flat Plan/Branch types, text<->AST parser, validator, serializer.

Surface forms accepted (compute is implicit at the end of every plan):

  retrieve(concept='X', period='CY1940') --> extract() --> compute()
  retrieve(concept='X', period='1990-09') --> extract(visual_only=True) --> compute()
  lookup_external(nl='CPI-U for 1953') --> compute()
  [ retrieve(...) --> extract() ; retrieve(...) --> extract() ] --> compute()
  [ retrieve(...) --> extract() ; lookup_external(...) ] --> compute()

These three shapes are the only ones any real plan uses. Sub-chains and nested
parallels are not representable.

Page number convention:
  PageRef.page = 1-based PDF page index (canonical throughout the codebase).
  The bulletin's printed-page footer is recoverable via
  skunk.common.parsed_json.get_printed_page() for trace/prompt enrichment,
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
class NamedEntry:
    """Sidecar metadata for one entry in a TypedValue.

    The payload itself lives in `TypedValue.value[name]`. This record carries the
    per-entry unit, the verbatim page phrase that anchors it, the kind discriminator
    that selects the payload shape, optional axis-name labels for vector/table, and
    an optional `dims` dict of categorical labels that lets compute group / filter
    siblings (e.g. {"denomination": 1, "series": "Total"}).

    Payload shapes by kind (enforced in TypedValue.__post_init__):
      - "scalar": value is int | float | str.
      - "vector": value is dict[str, int|float|str], keyed by index_name labels.
      - "table":  value is dict[str, dict[str, int|float|str]],
                  outer key = row_name label, inner key = col_name label.
    No nesting beyond these shapes.
    """
    unit: str = ""
    quote: str = ""
    dims: dict[str, Any] = field(default_factory=dict)
    kind: str = "scalar"                  # "scalar" | "vector" | "table"
    index_name: str | None = None         # vector only — name of the varying dim
    row_name: str | None = None           # table only
    col_name: str | None = None           # table only


def _is_primitive_cell(v: Any) -> bool:
    """A vector/table cell must be a non-bool number or a string."""
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float, str))


def _validate_payload(name: str, kind: str, value: Any) -> None:
    """Walk a TypedValue.value[name] payload and reject anything that doesn't
    match the declared `kind`'s flat shape. Raises ValueError on violation."""
    if kind not in VALID_KINDS:
        raise ValueError(f"NamedEntry {name!r}: invalid kind {kind!r}; must be one of {sorted(VALID_KINDS)}")

    if kind == "scalar":
        if not _is_primitive_cell(value):
            raise ValueError(
                f"NamedEntry {name!r}: kind='scalar' requires int|float|str, got {type(value).__name__}"
            )
        return

    if kind == "vector":
        if not isinstance(value, dict):
            raise ValueError(
                f"NamedEntry {name!r}: kind='vector' requires dict[str, scalar], got {type(value).__name__}"
            )
        for k, cell in value.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"NamedEntry {name!r}: vector index key must be str, got {type(k).__name__} for {k!r}"
                )
            if not _is_primitive_cell(cell):
                raise ValueError(
                    f"NamedEntry {name!r}: vector cell at {k!r} must be int|float|str (no nesting), "
                    f"got {type(cell).__name__}"
                )
        return

    # kind == "table" — 2-level dict-of-dict of primitive scalars. Ragged column
    # sets are allowed (real bulletin tables often start/stop mid-year); the
    # no-nesting invariant is enforced regardless of which columns each row has.
    if not isinstance(value, dict):
        raise ValueError(
            f"NamedEntry {name!r}: kind='table' requires dict[str, dict[str, scalar]], "
            f"got {type(value).__name__}"
        )
    for r, row in value.items():
        if not isinstance(r, str):
            raise ValueError(
                f"NamedEntry {name!r}: table row key must be str, got {type(r).__name__} for {r!r}"
            )
        if not isinstance(row, dict):
            raise ValueError(
                f"NamedEntry {name!r}: table row {r!r} must be a dict[str, scalar], "
                f"got {type(row).__name__}"
            )
        for c, cell in row.items():
            if not isinstance(c, str):
                raise ValueError(
                    f"NamedEntry {name!r}: table col key in row {r!r} must be str, "
                    f"got {type(c).__name__} for {c!r}"
                )
            if not _is_primitive_cell(cell):
                raise ValueError(
                    f"NamedEntry {name!r}: table cell at ({r!r}, {c!r}) must be int|float|str "
                    f"(no nesting), got {type(cell).__name__}"
                )


@dataclass
class TypedValue:
    value: dict[str, Any]              # keyed by name; single unnamed results use key ""
    desc: str = ""
    meta: dict[str, NamedEntry] = field(default_factory=dict)
    # meta is keyed identically to value; unit/quote/dims/kind live per-entry in NamedEntry.

    def __post_init__(self) -> None:
        # Validate every entry against its declared kind. A construction-time failure
        # here is a bug in extract, not silent data corruption downstream.
        for name, payload in self.value.items():
            entry = self.meta.get(name)
            kind = entry.kind if entry is not None else "scalar"
            _validate_payload(name, kind, payload)


@dataclass
class FormattedString:
    text: str


# Union type for inter-op values
OpOutput = DocHandle | TypedValue | FormattedString


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
    source_bulletin: str | None = None
    visual_only: bool = False       # passes through to the implicit extract step


@dataclass
class LookupBranch:
    nl: str


Branch = RetrieveBranch | LookupBranch


@dataclass
class Plan:
    branches: list[Branch] = field(default_factory=list)   # len 1 = simple; len > 1 = parallel


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


def _coerce(v: str) -> Any:
    v = v.strip()
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


def parse(text: str) -> Plan:
    """Parse text DSL into a Plan.

    Accepts only the three real shapes:
      retrieve(...) --> extract(...) --> compute()
      lookup_external(...) --> compute()
      [ branch ; branch ; ... ] --> compute()
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
    if last_args:
        raise ParseError(f"compute() takes no args, got {last_args!r}")

    body = parts[:-1]
    if not body:
        raise ParseError(f"Plan must have at least one data-gathering step before compute(): {text!r}")

    if len(body) == 1 and body[0].startswith("[") and body[0].endswith("]"):
        inner = body[0][1:-1].strip()
        raw_branches = _split_branches(inner)
        if len(raw_branches) < 2:
            raise ParseError(f"Parallel block must have >=2 branches, got {len(raw_branches)}")
        branches = [_parse_branch(b) for b in raw_branches]
    else:
        branches = [_parse_branch(" --> ".join(body))]

    return Plan(branches=branches)


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
        return RetrieveBranch(
            concept=str(head_args["concept"]),
            period=str(head_args["period"]),
            source_bulletin=str(head_args["source_bulletin"]) if "source_bulletin" in head_args else None,
            visual_only=bool(ext_args.get("visual_only", False)),
        )

    raise ParseError(f"Branch must start with retrieve or lookup_external, got {head_name!r}: {text!r}")


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------

def _serialize_branch(b: Branch) -> str:
    if isinstance(b, RetrieveBranch):
        args = [f"concept='{b.concept}'", f"period='{b.period}'"]
        if b.source_bulletin:
            args.append(f"source_bulletin='{b.source_bulletin}'")
        ext = "extract(visual_only=True)" if b.visual_only else "extract()"
        return f"retrieve({', '.join(args)}) --> {ext}"
    if isinstance(b, LookupBranch):
        return f"lookup_external(nl='{b.nl}')"
    raise TypeError(f"Unknown branch type: {type(b)}")


def serialize(plan: Plan) -> str:
    """Serialize Plan back to text DSL."""
    if not plan.branches:
        raise ValueError("Cannot serialize empty Plan")
    if len(plan.branches) == 1:
        body = _serialize_branch(plan.branches[0])
    else:
        body = "[ " + " ; ".join(_serialize_branch(b) for b in plan.branches) + " ]"
    return f"{body} --> compute()"


# ---------------------------------------------------------------------------
# JSON serialization (for storage / LLM output)
# ---------------------------------------------------------------------------

def _branch_to_dict(b: Branch) -> dict:
    if isinstance(b, RetrieveBranch):
        d: dict = {"kind": "retrieve", "concept": b.concept, "period": b.period}
        if b.source_bulletin:
            d["source_bulletin"] = b.source_bulletin
        if b.visual_only:
            d["visual_only"] = True
        return d
    if isinstance(b, LookupBranch):
        return {"kind": "lookup_external", "nl": b.nl}
    raise TypeError(f"Unknown branch type: {type(b)}")


def _branch_from_dict(d: dict) -> Branch:
    kind = d.get("kind")
    if kind == "retrieve":
        return RetrieveBranch(
            concept=d["concept"],
            period=d["period"],
            source_bulletin=d.get("source_bulletin"),
            visual_only=bool(d.get("visual_only", False)),
        )
    if kind == "lookup_external":
        return LookupBranch(nl=d["nl"])
    raise ValueError(f"Unknown branch kind: {kind!r}")


def to_dict(plan: Plan) -> dict:
    return {
        "branches": [_branch_to_dict(b) for b in plan.branches],
    }


def from_dict(d: dict) -> Plan:
    if "branches" not in d:
        raise ValueError(f"Plan dict missing required field 'branches': {d!r}")
    return Plan(branches=[_branch_from_dict(b) for b in d["branches"]])


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


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate(plan: Plan) -> ValidationResult:
    """Validate the plan. Type system enforces shape; only period grammar is checked here."""
    errors: list[str] = []
    if not plan.branches:
        errors.append("Plan has no branches")
    for i, b in enumerate(plan.branches):
        if isinstance(b, RetrieveBranch):
            if not _PERIOD_RE.match(b.period):
                errors.append(
                    f"branches[{i}]: period {b.period!r} does not match expected format "
                    "(CY/FY year, Qn-YYYY, YYYY-MM, YYYY-MM-DD, YYYY, range X..Y, enumeration X,Y)"
                )
    return ValidationResult(ok=not errors, errors=errors)
