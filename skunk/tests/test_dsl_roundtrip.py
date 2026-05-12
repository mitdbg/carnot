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
    ComputeNode,
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
    # decomposed: two intermediates -> final aggregator
    "[ [ retrieve(concept='a', period='CY1940') --> extract() ] --> compute(task='Compute 1940 total') ; "
    "[ retrieve(concept='a', period='CY1953') --> extract() ] --> compute(task='Compute 1953 total') ] --> compute()",
    # decomposed: lookup + retrieve sub-computes
    "[ [ lookup_external(nl='year WWII ended') ] --> compute(task='Find WWII end year') ; "
    "[ lookup_external(nl='year Korean War started') ] --> compute(task='Find Korean War start year') ] --> compute()",
]

INVALID_TEMPLATES = [
    "",                                                              # empty
    "unknownop(foo='x') --> compute()",                              # bad op name
    "filter(concept='x') --> compute()",                             # not a DSL op
    "retrieve(concept='x', period='CY1940') --> extract()",          # missing compute terminator
    "retrieve(concept='x', period='CY1940') --> compute()",          # retrieve without extract
    "extract() --> compute()",                                       # branch without head
    "[ retrieve(concept='x', period='CY1940') --> extract() ] --> compute()",  # parallel with 1 branch
    "compute()",                                                     # only compute
    # decomposed with mixed members (some sub-compute brackets, some plain branches)
    "[ [ retrieve(concept='a', period='CY1940') --> extract() ] --> compute(task='X') ; "
    "lookup_external(nl='y') ] --> compute()",
    # intermediate compute with missing task
    "[ [ retrieve(concept='a', period='CY1940') --> extract() ] --> compute() ; "
    "[ retrieve(concept='a', period='CY1953') --> extract() ] --> compute(task='Y') ] --> compute()",
]


# ---------------------------------------------------------------------------
# Parse + roundtrip on hand-written templates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("template", VALID_TEMPLATES)
def test_parse_valid_templates(template: str) -> None:
    plan = parse(template)
    assert isinstance(plan, Plan)
    assert len(plan.computes) >= 1
    assert plan.computes[-1].final


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
        plan = parse(template)
        # Some shapes parse but fail structural validation — surface that too.
        result = validate(plan)
        if not result.ok:
            raise ValueError(result.errors)


# ---------------------------------------------------------------------------
# Targeted shape checks
# ---------------------------------------------------------------------------

def test_simple_chain_yields_single_retrieve_branch() -> None:
    plan = parse("retrieve(concept='x', period='CY1940') --> extract() --> compute()")
    assert len(plan.computes) == 1
    final = plan.computes[0]
    assert final.final and final.task == ""
    assert len(final.branches) == 1
    assert isinstance(final.branches[0], RetrieveBranch)
    assert final.branches[0].concept == "x"
    assert final.branches[0].period == "CY1940"
    assert final.branches[0].visual_only is False


def test_visual_only_flag_threads_through() -> None:
    plan = parse(
        "retrieve(concept='chart', period='1990-09', source_bulletin='1990-09') "
        "--> extract(visual_only=True) --> compute()"
    )
    b = plan.computes[0].branches[0]
    assert b.visual_only is True
    assert b.source_bulletin == "1990-09"


def test_lookup_only_chain() -> None:
    plan = parse("lookup_external(nl='CPI-U for 1953') --> compute()")
    branches = plan.computes[0].branches
    assert len(branches) == 1
    assert isinstance(branches[0], LookupBranch)
    assert branches[0].nl == "CPI-U for 1953"


def test_parallel_branches() -> None:
    plan = parse(
        "[ retrieve(concept='a', period='CY1940') --> extract() ; "
        "lookup_external(nl='b') ] --> compute()"
    )
    branches = plan.computes[0].branches
    assert len(branches) == 2
    assert isinstance(branches[0], RetrieveBranch)
    assert isinstance(branches[1], LookupBranch)


def test_bad_period_caught_by_validator() -> None:
    plan = Plan(computes=[ComputeNode(
        branches=[RetrieveBranch(concept="x", period="not-a-period")],
        task="", final=True,
    )])
    result = validate(plan)
    assert not result.ok
    assert any("period" in e for e in result.errors)


def test_period_range_accepted() -> None:
    plan = parse("retrieve(concept='x', period='CY1940..CY1949') --> extract() --> compute()")
    assert validate(plan).ok


def test_decomposed_plan_shape() -> None:
    text = (
        "[ [ retrieve(concept='a', period='CY1940') --> extract() ] --> compute(task='one') ; "
        "[ retrieve(concept='b', period='CY1953') --> extract() ] --> compute(task='two') ] "
        "--> compute()"
    )
    plan = parse(text)
    assert len(plan.computes) == 3
    assert plan.computes[0].task == "one" and not plan.computes[0].final
    assert plan.computes[1].task == "two" and not plan.computes[1].final
    assert plan.computes[2].final and plan.computes[2].branches == []
    # Depth = 2 (one intermediate layer + final aggregator), regardless of how
    # many intermediates run in parallel.
    assert validate(plan, max_compute_depth=2).ok


def test_depth_limit_rejects_decomposed_when_max_depth_is_one() -> None:
    text = (
        "[ [ retrieve(concept='a', period='CY1940') --> extract() ] --> compute(task='one') ; "
        "[ retrieve(concept='b', period='CY1953') --> extract() ] --> compute(task='two') ] "
        "--> compute()"
    )
    plan = parse(text)
    result = validate(plan, max_compute_depth=1)
    assert not result.ok
    assert any("depth" in e.lower() for e in result.errors)


def test_legacy_branches_dict_accepted_by_from_dict() -> None:
    d = {"branches": [{"kind": "retrieve", "concept": "x", "period": "CY1940"}]}
    plan = from_dict(d)
    assert len(plan.computes) == 1
    assert plan.computes[0].final
    assert plan.computes[0].task == ""
    assert len(plan.computes[0].branches) == 1


def test_legacy_to_dict_round_trip_preserves_branches_shape() -> None:
    plan = parse("retrieve(concept='x', period='CY1940') --> extract() --> compute()")
    d = to_dict(plan)
    assert "branches" in d and "computes" not in d


def test_decomposed_to_dict_uses_computes_shape() -> None:
    text = (
        "[ [ retrieve(concept='a', period='CY1940') --> extract() ] --> compute(task='one') ; "
        "[ retrieve(concept='b', period='CY1953') --> extract() ] --> compute(task='two') ] "
        "--> compute()"
    )
    plan = parse(text)
    d = to_dict(plan)
    assert "computes" in d and "branches" not in d
    assert len(d["computes"]) == 3
    assert d["computes"][-1].get("final") is True


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
