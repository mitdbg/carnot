"""DSL — AST types, text<->AST parser, validator, serializer.

Text surface form:
    retrieve(concept='national_defense', period='CY1940') --> extract(concept='total', mode='value')

Parallel branches via brackets + semicolons:
    [ A --> B ; C --> D ] --> compute(code='...')

Nested:
    [ [ A ; B ] --> compute(code='...') ; lookup_external(resource='cpi_u') ] --> compute(code='...')

Page number convention:
    PageRef.page      = bulletin printed page number (canonical; matches source_docs?page=N)
    PageRef.pdf_page  = 1-based PDF page index (for cache file access and rendering)
    Translation between the two lives in prep/page_map.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Inter-op data model
# ---------------------------------------------------------------------------

@dataclass
class PageRef:
    year: int | None = None
    month: str | None = None        # "YYYY-MM"
    page: int | None = None         # bulletin printed page number (canonical)
    pdf_page: int | None = None     # 1-based PDF page index (for cache/render access)
    file_path: str | None = None    # resolved by manifest

    def __repr__(self) -> str:
        parts = []
        if self.year:
            parts.append(f"year={self.year}")
        if self.month:
            parts.append(f"month={self.month}")
        if self.page is not None:
            parts.append(f"page={self.page}")
        if self.pdf_page is not None:
            parts.append(f"pdf_page={self.pdf_page}")
        if self.file_path:
            parts.append(f"file={self.file_path!r}")
        return f"PageRef({', '.join(parts)})"


@dataclass
class DocHandle:
    refs: list[PageRef] = field(default_factory=list)
    desc: str = ""

    def is_empty(self) -> bool:
        return len(self.refs) == 0


@dataclass
class TypedValue:
    value: Any              # never None; subagent must raise StepFailed instead
    dtype: str = "scalar"   # "scalar" | "list[scalar]" | "text" | "df"
    unit: str = ""          # semantic unit: "usd_millions", "pct", "fx_rate", "year", "cpi", etc.
    desc: str = ""


@dataclass
class FormattedString:
    text: str
    desc: str = ""


# Union type for inter-op values
OpOutput = DocHandle | TypedValue | FormattedString


# ---------------------------------------------------------------------------
# AST node types
# ---------------------------------------------------------------------------

VALID_OPS = frozenset({
    "retrieve", "extract", "read_visual",
    "lookup_external", "compute", "format",
})


@dataclass
class OpNode:
    op: str
    args: dict[str, Any] = field(default_factory=dict)
    concepts: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    node_type: str = field(default="op", init=False)

    def __post_init__(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"Unknown op: {self.op!r}. Must be one of {sorted(VALID_OPS)}")


@dataclass
class ParallelNode:
    branches: list[ChainNode] = field(default_factory=list)
    node_type: str = field(default="parallel", init=False)


@dataclass
class ChainNode:
    steps: list[OpNode | ParallelNode] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)
    node_type: str = field(default="chain", init=False)


# Root pipeline is always a ChainNode
Pipeline = ChainNode


# ---------------------------------------------------------------------------
# Text DSL parser
# ---------------------------------------------------------------------------

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
    if current:
        branches.append("".join(current).strip())
    return [b for b in branches if b]


_OP_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
_KV_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)")


def _parse_args(raw: str) -> dict[str, Any]:
    """Parse 'key=val, key2=val2' or positional 'val1, val2' into a dict."""
    raw = raw.strip()
    if not raw:
        return {}
    args: dict[str, Any] = {}
    # split on commas not inside brackets or parens
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in raw:
        if ch in "([":
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

    positional_idx = 0
    positional_keys = ["source", "concept", "locator", "resource", "formula",
                       "reducer", "spec", "element", "criterion"]
    for part in parts:
        part = part.strip()
        m = _KV_RE.match(part)
        if m:
            k, v = m.group(1).strip(), m.group(2).strip()
            # Try to parse value as int, then float, else keep as string
            args[k] = _coerce(v)
        else:
            # positional
            key = positional_keys[positional_idx] if positional_idx < len(positional_keys) else f"arg{positional_idx}"
            args[key] = _coerce(part)
            positional_idx += 1
    return args


def _coerce(v: str) -> Any:
    """Convert string token to int/float/str as appropriate."""
    v = v.strip()
    # Remove wrapping quotes if present
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_step(token: str) -> OpNode | ParallelNode:
    """Parse a single step token into an OpNode or ParallelNode."""
    token = token.strip()

    # Parallel block: starts and ends with [ ]
    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        raw_branches = _split_branches(inner)
        branches = [parse(b) for b in raw_branches]
        return ParallelNode(branches=branches)

    # Op: name(args...)
    m = _OP_RE.match(token)
    if not m:
        raise ParseError(f"Cannot parse step: {token!r}")
    op_name = m.group(1)
    raw_args = m.group(2)
    return OpNode(op=op_name, args=_parse_args(raw_args))


def parse(text: str) -> ChainNode:
    """Parse text DSL into a ChainNode (Pipeline)."""
    text = text.strip()
    tokens = _split_chain(text)
    if not tokens:
        raise ParseError(f"Empty pipeline text: {text!r}")
    steps: list[OpNode | ParallelNode] = []
    for token in tokens:
        steps.append(_parse_step(token))
    return ChainNode(steps=steps)


class ParseError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Serializer (AST → text DSL)
# ---------------------------------------------------------------------------

def _args_to_str(args: dict[str, Any]) -> str:
    parts = []
    for k, v in args.items():
        if k in ("source", "concept", "locator", "resource", "formula",
                 "reducer", "spec", "element", "criterion"):
            # positional
            parts.append(str(v))
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _serialize_step(step: OpNode | ParallelNode) -> str:
    if isinstance(step, OpNode):
        return f"{step.op}({_args_to_str(step.args)})"
    elif isinstance(step, ParallelNode):
        branch_strs = [" --> ".join(_serialize_step(s) for s in b.steps) for b in step.branches]
        return "[ " + " ; ".join(branch_strs) + " ]"
    raise TypeError(f"Unknown step type: {type(step)}")


def serialize(chain: ChainNode) -> str:
    """Serialize ChainNode back to text DSL."""
    return " --> ".join(_serialize_step(s) for s in chain.steps)


# ---------------------------------------------------------------------------
# JSON serialization (for storage / LLM output)
# ---------------------------------------------------------------------------

def to_dict(node: ChainNode | OpNode | ParallelNode) -> dict:
    if isinstance(node, ChainNode):
        return {
            "type": "chain",
            "global_constraints": node.global_constraints,
            "steps": [to_dict(s) for s in node.steps],
        }
    elif isinstance(node, OpNode):
        return {
            "type": "op",
            "op": node.op,
            "args": node.args,
            "concepts": node.concepts,
            "constraints": node.constraints,
        }
    elif isinstance(node, ParallelNode):
        return {
            "type": "parallel",
            "branches": [to_dict(b) for b in node.branches],
        }
    raise TypeError(f"Unknown node type: {type(node)}")


def from_dict(d: dict) -> ChainNode | OpNode | ParallelNode:
    t = d.get("type")
    if t == "chain":
        return ChainNode(
            steps=[from_dict(s) for s in d.get("steps", [])],
            global_constraints=d.get("global_constraints", []),
        )
    elif t == "op":
        return OpNode(
            op=d["op"],
            args=d.get("args", {}),
            concepts=d.get("concepts", []),
            constraints=d.get("constraints", []),
        )
    elif t == "parallel":
        return ParallelNode(branches=[from_dict(b) for b in d.get("branches", [])])
    raise ValueError(f"Unknown node type: {t!r}")


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)


def validate(chain: ChainNode) -> ValidationResult:
    errors: list[str] = []
    _validate_node(chain, errors, path="root")
    return ValidationResult(ok=len(errors) == 0, errors=errors)


def _validate_node(node: ChainNode | OpNode | ParallelNode, errors: list[str], path: str) -> None:
    if isinstance(node, ChainNode):
        if not node.steps:
            errors.append(f"{path}: empty chain")
        for i, step in enumerate(node.steps):
            _validate_node(step, errors, path=f"{path}.steps[{i}]")
    elif isinstance(node, OpNode):
        if node.op not in VALID_OPS:
            errors.append(f"{path}: unknown op {node.op!r}")
    elif isinstance(node, ParallelNode):
        if len(node.branches) < 2:
            errors.append(f"{path}: parallel must have ≥2 branches, got {len(node.branches)}")
        for i, b in enumerate(node.branches):
            _validate_node(b, errors, path=f"{path}.branches[{i}]")
    else:
        errors.append(f"{path}: unknown node type {type(node).__name__}")
