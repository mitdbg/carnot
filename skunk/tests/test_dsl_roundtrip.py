"""DSL round-trip tests for the flat Plan AST.

Verifies:
1. All valid templates parse without raising ParseError.
2. parse → serialize → parse → serialize is stable.
3. AST → JSON → AST is lossless.
4. Invalid templates (shapes the new flat AST doesn't accept) fail parse.
5. Every cached plan in data/dsl_planning_pass.csv parses cleanly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from skunk.dsl import (
    LookupBranch,
    ParseError,
    Plan,
    RetrieveBranch,
    from_dict,
    parse,
    serialize,
    to_dict,
    validate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_CACHE_CSV = REPO_ROOT / "data" / "dsl_planning_pass.csv"


VALID_TEMPLATES = [
    # simple retrieve chain
    "retrieve(concept='national_defense', period='CY1940') --> extract() --> compute()",
    # parallel of two retrieves
    "[ retrieve(concept='nd', period='CY1940') --> extract() ; retrieve(concept='nd', period='CY1953') --> extract() ] --> compute()",
    # visual_only flag
    "retrieve(concept='debt_chart', period='1990-09', source_bulletin='1990-09') --> extract(visual_only=True) --> compute()",
    # lookup_external + retrieve in parallel
    "[ retrieve(concept='fx_investments', period='2025-03') --> extract() ; lookup_external(nl='USD/JPY exchange rate on 2025-03-31') ] --> compute()",
    # range period
    "retrieve(concept='interest_rates', period='CY1953..CY1955') --> extract() --> compute()",
    # source_bulletin pin
    "retrieve(concept='expenditure_table', period='FY1940', source_bulletin='1941-06') --> extract() --> compute()",
    # single lookup_external
    "lookup_external(nl='year WWII ended') --> compute()",
    # parallel of lookup_externals only
    "[ lookup_external(nl='year WWII ended') ; lookup_external(nl='year Korean War started') ] --> compute()",
]

INVALID_TEMPLATES = [
    "",                                                              # empty
    "unknownop(foo='x') --> compute()",                              # bad op name
    "filter(concept='x') --> compute()",                             # not a DSL op
    "retrieve(concept='x', period='CY1940') --> extract()",          # missing compute terminator
    "retrieve(concept='x', period='CY1940') --> compute()",          # retrieve without extract
    "extract() --> compute()",                                       # branch without head
    "[ retrieve(concept='x', period='CY1940') --> extract() ] --> compute()",  # parallel with 1 branch
    "[ [ retrieve(concept='x', period='CY1940') --> extract() ; retrieve(concept='y', period='CY1941') --> extract() ] --> compute() ; lookup_external(nl='z') ] --> compute()",  # nested parallel
    "compute()",                                                     # only compute
]


# ---------------------------------------------------------------------------
# Parse + roundtrip on hand-written templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_parse_valid_templates(template: str) -> None:
    plan = parse(template)
    assert isinstance(plan, Plan)
    assert len(plan.branches) >= 1


@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_serialize_roundtrip_stable(template: str) -> None:
    plan1 = parse(template)
    s1 = serialize(plan1)
    plan2 = parse(s1)
    s2 = serialize(plan2)
    assert s1 == s2, f"Serialization not stable:\n  s1={s1}\n  s2={s2}"


@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_json_roundtrip(template: str) -> None:
    plan = parse(template)
    d = to_dict(plan)
    plan2 = from_dict(json.loads(json.dumps(d)))
    assert plan == plan2


@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_validate_valid_templates_ok(template: str) -> None:
    plan = parse(template)
    result = validate(plan)
    assert result.ok, f"Validation failed:\n{template}\nErrors: {result.errors}"


@pytest.mark.parametrize("template", INVALID_TEMPLATES)
def test_invalid_templates_rejected(template: str) -> None:
    with pytest.raises((ParseError, ValueError)):
        parse(template)


# ---------------------------------------------------------------------------
# Targeted shape checks
# ---------------------------------------------------------------------------

def test_simple_chain_yields_single_retrieve_branch() -> None:
    plan = parse("retrieve(concept='x', period='CY1940') --> extract() --> compute()")
    assert len(plan.branches) == 1
    assert isinstance(plan.branches[0], RetrieveBranch)
    assert plan.branches[0].concept == "x"
    assert plan.branches[0].period == "CY1940"
    assert plan.branches[0].visual_only is False


def test_visual_only_flag_threads_through() -> None:
    plan = parse(
        "retrieve(concept='chart', period='1990-09', source_bulletin='1990-09') "
        "--> extract(visual_only=True) --> compute()"
    )
    assert plan.branches[0].visual_only is True
    assert plan.branches[0].source_bulletin == "1990-09"


def test_lookup_only_chain() -> None:
    plan = parse("lookup_external(nl='CPI-U for 1953') --> compute()")
    assert len(plan.branches) == 1
    assert isinstance(plan.branches[0], LookupBranch)
    assert plan.branches[0].nl == "CPI-U for 1953"


def test_parallel_branches() -> None:
    plan = parse(
        "[ retrieve(concept='a', period='CY1940') --> extract() ; "
        "lookup_external(nl='b') ] --> compute()"
    )
    assert len(plan.branches) == 2
    assert isinstance(plan.branches[0], RetrieveBranch)
    assert isinstance(plan.branches[1], LookupBranch)


def test_bad_period_caught_by_validator() -> None:
    plan = Plan(branches=[RetrieveBranch(concept="x", period="not-a-period")])
    result = validate(plan)
    assert not result.ok
    assert any("period" in e for e in result.errors)


def test_period_range_accepted() -> None:
    plan = parse("retrieve(concept='x', period='CY1940..CY1949') --> extract() --> compute()")
    assert validate(plan).ok


# ---------------------------------------------------------------------------
# Cached plans (regression: every entry in data/dsl_planning_pass.csv must parse)
# ---------------------------------------------------------------------------

def _load_cached_plans() -> list[tuple[str, str]]:
    if not PLAN_CACHE_CSV.exists():
        return []
    rows: list[tuple[str, str]] = []
    with PLAN_CACHE_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            text = row.get("plan_text", "")
            if text:
                rows.append((row["uid"], text))
    return rows


@pytest.mark.parametrize("uid,plan_text", _load_cached_plans())
def test_cached_plan_parses_and_validates(uid: str, plan_text: str) -> None:
    plan = parse(plan_text)
    assert isinstance(plan, Plan)
    # Roundtrip
    assert parse(serialize(plan)) == plan
    assert from_dict(to_dict(plan)) == plan
    # Validate
    assert validate(plan).ok, f"{uid}: {validate(plan).errors}"
