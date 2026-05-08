"""DSL round-trip tests.

Tests:
1. All templates parse without raising ParseError.
2. Serialized-then-reparsed text matches the original serialization (stable).
3. AST → JSON → AST round-trip is lossless.
4. Validator catches known-bad inputs.
"""

from __future__ import annotations

import json
import pytest

from skunk.dsl import (
    ChainNode, OpNode, ParallelNode,
    ParseError, ValidationResult,
    from_dict, parse, serialize, to_dict, validate
)

# ---------------------------------------------------------------------------
# Templates using canonical single-quoted string args
# ---------------------------------------------------------------------------

VALID_TEMPLATES = [
    # simple chain
    "retrieve(concept='national_defense', period='CY1940') --> extract(concept='total_expenditure', mode='value') --> format(precision=0)",
    # parallel retrieve → compute → format
    "[ retrieve(concept='national_defense', period='CY1940') --> extract(concept='total', mode='value') ; retrieve(concept='national_defense', period='CY1953') --> extract(concept='total', mode='value') ] --> compute(code='abs_pct_change') --> format(precision=2)",
    # read_visual
    "retrieve(concept='debt_chart', period='FY1975') --> read_visual(concept='total_debt_by_type') --> format(precision=1)",
    # lookup_external + retrieve parallel
    "[ retrieve(concept='fx_investments', period='2025-03') --> extract(concept='japanese_yen_holdings', mode='value') ; lookup_external(resource='fx_rate', pair='USD/JPY', date='2025-03-31') ] --> compute(code='currency_conversion') --> format(precision=1)",
    # nested brackets
    "[ [ retrieve(concept='national_defense', period='CY1940') --> extract(concept='monthly_total', mode='list') ; retrieve(concept='national_defense', period='CY1953') --> extract(concept='monthly_total', mode='list') ] --> compute(code='abs_diff') ; lookup_external(resource='fx_rate', pair='USD/CAD') ] --> compute(code='currency_conversion') --> format(precision=2)",
    # list extract → compute → format
    "retrieve(concept='interest_rates', period='1953..1955') --> extract(concept='91day_bill_rate', mode='list') --> compute(code='geo_mean') --> format(precision=3)",
    # source_bulletin pin
    "retrieve(concept='expenditure_table', period='FY1940', source_bulletin='1941-06') --> extract(concept='national_defense_total', mode='value') --> format(precision=0)",
]

INVALID_TEMPLATES = [
    "",                                            # empty
    "unknownop(foo) --> format(x)",                # bad op
    "filter(concept='x') --> format(x)",           # filter is not a DSL op
    "aggregate(reducer='sum') --> format(x)",      # aggregate is not a DSL op
    "[ A --> B ]",                                 # single branch parallel fails validation
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_parse_valid_templates(template: str) -> None:
    chain = parse(template)
    assert isinstance(chain, ChainNode)
    assert len(chain.steps) > 0


@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_serialize_roundtrip_stable(template: str) -> None:
    """Parsing → serializing → re-parsing → re-serializing should be idempotent."""
    chain1 = parse(template)
    s1 = serialize(chain1)
    chain2 = parse(s1)
    s2 = serialize(chain2)
    assert s1 == s2, f"Serialization not stable:\n  s1={s1}\n  s2={s2}"


@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_json_roundtrip(template: str) -> None:
    """AST → JSON → AST should be lossless (same serialization)."""
    chain = parse(template)
    d = to_dict(chain)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    chain2 = from_dict(d2)
    assert isinstance(chain2, ChainNode)
    assert serialize(chain) == serialize(chain2)


@pytest.mark.parametrize("template", INVALID_TEMPLATES)
def test_invalid_templates_rejected(template: str) -> None:
    """Templates with unknown ops or structural issues must fail parse or validate."""
    try:
        chain = parse(template)
        result = validate(chain)
        assert not result.ok, f"Expected invalid template to fail:\n{template}"
    except (ParseError, ValueError):
        pass


def test_parse_empty_raises() -> None:
    with pytest.raises(ParseError):
        parse("")


def test_unknown_op_raises() -> None:
    with pytest.raises(ValueError, match="Unknown op"):
        OpNode(op="badop")


def test_validate_single_branch_parallel() -> None:
    # A parallel node with only 1 branch should fail validation
    chain = ChainNode(steps=[
        ParallelNode(branches=[
            ChainNode(steps=[OpNode(op="retrieve", args={"source": "x"})])
        ]),
        OpNode(op="format", args={"spec": "x"}),
    ])
    result = validate(chain)
    assert not result.ok
    assert any("≥2" in e for e in result.errors)


def test_validate_empty_chain() -> None:
    result = validate(ChainNode(steps=[]))
    assert not result.ok


def test_parallel_branches_parse() -> None:
    template = "[ retrieve(x, year=1940) --> extract(y) ; retrieve(x, year=1950) --> extract(y) ] --> format(x)"
    chain = parse(template)
    assert isinstance(chain.steps[0], ParallelNode)
    assert len(chain.steps[0].branches) == 2


def test_nested_parallel_parse() -> None:
    template = "[ [ retrieve(x) --> extract(a) ; retrieve(y) --> extract(b) ] --> compute(diff) ; lookup_external(z) ] --> compute(convert)"
    chain = parse(template)
    assert isinstance(chain.steps[0], ParallelNode)
    # First branch should itself be a chain with a parallel inside
    branch0 = chain.steps[0].branches[0]
    assert isinstance(branch0, ChainNode)


@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_validate_valid_templates_ok(template: str) -> None:
    chain = parse(template)
    result = validate(chain)
    assert result.ok, f"Validation failed for template:\n{template}\nErrors: {result.errors}"


def test_opnode_concepts_constraints_preserved() -> None:
    """Check that per-op metadata survives JSON round-trip."""
    chain = parse("retrieve(treasury_bulletin, year=1940) --> format(millions_usd)")
    chain.steps[0].concepts = ["calendar year (Treasury)"]
    chain.steps[0].constraints = ["calendar year not fiscal"]

    d = to_dict(chain)
    chain2 = from_dict(d)
    assert chain2.steps[0].concepts == ["calendar year (Treasury)"]
    assert chain2.steps[0].constraints == ["calendar year not fiscal"]
