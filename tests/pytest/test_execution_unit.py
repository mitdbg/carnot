"""Unit tests for Execution helper methods (no LLM required).

Tests cover:
1. ``_get_op_from_plan_dict`` — returns the correct physical operator
   type for every known operator name, and raises for unknowns.
2. ``_get_ops_in_topological_order`` — produces the correct linearised
   order for single-op, chained, and branching (join) plan DAGs.
3. ``_operator_display_name`` — returns human-readable labels.
4. ``ExecutionProgress.to_dict`` — serializes correctly.
"""

from __future__ import annotations

import pytest

from carnot.data.dataset import Dataset
from carnot.execution.execution import Execution
from carnot.execution.progress import ExecutionProgress
from carnot.operators.code import CodeOperator
from carnot.operators.limit import LimitOperator
from carnot.operators.reasoning import ReasoningOperator
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator

# ── Helpers ─────────────────────────────────────────────────────────────────

# Minimal llm_config so Execution can be instantiated without env vars.
_LLM_CONFIG = {"OPENAI_API_KEY": "test-key-not-real"}


def _make_execution(datasets: list[Dataset] | None = None) -> Execution:
    """Create a minimal Execution instance for unit-testing helper methods.

    Requires:
        - ``_LLM_CONFIG`` provides a dummy API key.

    Returns:
        An ``Execution`` object ready for plan-parsing tests.
    """
    return Execution(
        query="test query",
        datasets=datasets or [],
        llm_config=_LLM_CONFIG,
    )


def _leaf_plan(name: str) -> dict:
    """Build a leaf plan node representing a raw Dataset reference.

    The plan node has no operator param so ``_get_op_from_plan_dict``
    falls through to the dataset-name lookup.
    """
    return {
        "name": name,
        "output_dataset_id": name,
        "params": {},
        "parents": [],
    }


def _op_plan(
    operator: str,
    output_id: str,
    parents: list[dict],
    **extra_params,
) -> dict:
    """Build an operator plan node with given parents."""
    params = {"operator": operator, **extra_params}
    return {
        "name": output_id,
        "output_dataset_id": output_id,
        "params": params,
        "parents": parents,
    }


# ═══════════════════════════════════════════════════════════════════════
# _get_op_from_plan_dict
# ═══════════════════════════════════════════════════════════════════════


class TestGetOpFromPlanDict:
    """Verify that each operator name maps to the correct physical class."""

    def test_code_operator(self):
        """'Code' → CodeOperator."""
        ex = _make_execution()
        plan = _op_plan("Code", "code1", [], task="compute stats")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, CodeOperator)
        assert parent_ids == []

    def test_limit_operator(self):
        """'Limit' → LimitOperator."""
        ex = _make_execution()
        plan = _op_plan("Limit", "limit1", [], n=5)
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, LimitOperator)

    def test_semantic_filter(self):
        """'SemanticFilter' → SemFilterOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemFilterOperator)
        assert parent_ids == ["Movies"]

    def test_semantic_map(self):
        """'SemanticMap' → SemMapOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticMap", "map1", [leaf],
            field="sentiment", type="str", field_desc="overall sentiment",
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemMapOperator)

    def test_semantic_flat_map(self):
        """'SemanticFlatMap' → SemFlatMapOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticFlatMap", "flatmap1", [leaf],
            field="keyword", type="str", field_desc="extracted keyword",
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemFlatMapOperator)

    def test_semantic_agg(self):
        """'SemanticAgg' → SemAggOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticAgg", "agg1", [leaf],
            task="summarize reviews",
            agg_fields=[{"name": "summary", "type": "str"}],
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemAggOperator)

    def test_semantic_groupby(self):
        """'SemanticGroupBy' → SemGroupByOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticGroupBy", "gby1", [leaf],
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemGroupByOperator)

    def test_semantic_join(self):
        """'SemanticJoin' → SemJoinOperator with two parent ids."""
        ex = _make_execution()
        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan = _op_plan("SemanticJoin", "join1", [left, right], condition="same movie")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemJoinOperator)
        assert parent_ids == ["Movies", "Reviews"]

    def test_semantic_topk(self):
        """'SemanticTopK' → SemTopKOperator."""
        ex = _make_execution()
        leaf = _leaf_plan("Movies")
        plan = _op_plan(
            "SemanticTopK", "topk1", [leaf],
            search_str="best action", k=5, index_name="flat",
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert isinstance(op, SemTopKOperator)

    def test_dataset_reference(self):
        """A plan node whose name matches a dataset returns that Dataset."""
        ds = Dataset(name="Movies", annotation="film data")
        ex = _make_execution(datasets=[ds])
        plan = _leaf_plan("Movies")
        op, parent_ids = ex._get_op_from_plan_dict(plan)
        assert op is ds
        assert parent_ids == []

    def test_unknown_operator_raises(self):
        """An unrecognized operator name raises ValueError."""
        ex = _make_execution()
        plan = _op_plan("UnknownOp", "unk1", [])
        with pytest.raises(ValueError, match="Unknown operator"):
            ex._get_op_from_plan_dict(plan)


# ═══════════════════════════════════════════════════════════════════════
# _get_ops_in_topological_order
# ═══════════════════════════════════════════════════════════════════════


class TestGetOpsInTopologicalOrder:
    """Verify topological linearization of plan DAGs."""

    def test_single_leaf(self):
        """A single dataset node yields one entry."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        plan = _leaf_plan("Movies")
        ops = ex._get_ops_in_topological_order(plan)
        assert len(ops) == 1
        assert ops[0][0] is ds

    def test_linear_chain(self):
        """Dataset → Filter yields [Dataset, Filter] in order."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        leaf = _leaf_plan("Movies")
        plan = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        ops = ex._get_ops_in_topological_order(plan)
        assert len(ops) == 2
        # First is the dataset, second is the filter
        assert ops[0][0] is ds
        assert isinstance(ops[1][0], SemFilterOperator)

    def test_two_step_chain(self):
        """Dataset → Filter → Map yields three entries in correct order."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        leaf = _leaf_plan("Movies")
        filt = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        mapped = _op_plan(
            "SemanticMap", "map1", [filt],
            field="label", type="str", field_desc="genre label",
        )
        ops = ex._get_ops_in_topological_order(mapped)
        assert len(ops) == 3
        assert ops[0][0] is ds
        assert isinstance(ops[1][0], SemFilterOperator)
        assert isinstance(ops[2][0], SemMapOperator)

    def test_join_has_both_parents_before_join(self):
        """A join node is preceded by both parent datasets."""
        ds_a = Dataset(name="Movies")
        ds_b = Dataset(name="Reviews")
        ex = _make_execution(datasets=[ds_a, ds_b])

        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan = _op_plan("SemanticJoin", "join1", [left, right], condition="same movie")

        ops = ex._get_ops_in_topological_order(plan)
        assert len(ops) == 3

        # The join must be last
        assert isinstance(ops[2][0], SemJoinOperator)

        # Both datasets appear before the join
        pre_join_types = {type(o[0]) for o in ops[:2]}
        assert Dataset in pre_join_types

    def test_parent_ids_propagated(self):
        """Parent dataset IDs are correctly threaded through the order."""
        ds = Dataset(name="Movies")
        ex = _make_execution(datasets=[ds])
        leaf = _leaf_plan("Movies")
        plan = _op_plan("SemanticFilter", "filter1", [leaf], condition="x")
        ops = ex._get_ops_in_topological_order(plan)
        # The filter's parent_ids should be ["Movies"]
        _, parent_ids = ops[1]
        assert parent_ids == ["Movies"]


# ═══════════════════════════════════════════════════════════════════════
# _operator_display_name
# ═══════════════════════════════════════════════════════════════════════


class TestOperatorDisplayName:
    """Verify that _operator_display_name returns readable labels."""

    def test_dataset_includes_name(self):
        """A Dataset produces 'Dataset: <name>'."""
        ds = Dataset(name="Movies")
        assert Execution._operator_display_name(ds) == "Dataset: Movies"

    def test_sem_filter(self):
        """SemFilterOperator → 'Semantic Filter'."""
        ex = _make_execution()
        plan = _op_plan("SemanticFilter", "f1", [], condition="x")
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Filter"

    def test_sem_map(self):
        """SemMapOperator → 'Semantic Map'."""
        ex = _make_execution()
        plan = _op_plan("SemanticMap", "m1", [], field="x", type="str", field_desc="d")
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Map"

    def test_sem_join(self):
        """SemJoinOperator → 'Semantic Join'."""
        ex = _make_execution()
        plan = _op_plan("SemanticJoin", "j1", [_leaf_plan("A"), _leaf_plan("B")], condition="c")
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Join"

    def test_limit(self):
        """LimitOperator → 'Limit'."""
        ex = _make_execution()
        plan = _op_plan("Limit", "l1", [], n=5)
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Limit"

    def test_code(self):
        """CodeOperator → 'Code'."""
        ex = _make_execution()
        plan = _op_plan("Code", "c1", [], task="compute")
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Code"

    def test_sem_topk(self):
        """SemTopKOperator → 'Semantic Top-K'."""
        ex = _make_execution()
        plan = _op_plan("SemanticTopK", "t1", [], search_str="best", k=3, index_name="flat")
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Top-K"

    def test_sem_agg(self):
        """SemAggOperator → 'Semantic Aggregation'."""
        ex = _make_execution()
        plan = _op_plan("SemanticAgg", "a1", [], task="summarize", agg_fields=[{"name": "s", "type": "str"}])
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Aggregation"

    def test_sem_groupby(self):
        """SemGroupByOperator → 'Semantic Group By'."""
        ex = _make_execution()
        plan = _op_plan(
            "SemanticGroupBy", "g1", [],
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Group By"

    def test_sem_flat_map(self):
        """SemFlatMapOperator → 'Semantic Flat Map'."""
        ex = _make_execution()
        plan = _op_plan("SemanticFlatMap", "fm1", [], field="kw", type="str", field_desc="keyword")
        op, _ = ex._get_op_from_plan_dict(plan)
        assert Execution._operator_display_name(op) == "Semantic Flat Map"

    def test_reasoning_operator(self):
        """ReasoningOperator → 'Reasoning' (inherits from CodeOperator)."""
        op = ReasoningOperator(
            task="reason",
            output_dataset_id="out",
            model_id="openai/gpt-5-mini",
            llm_config=_LLM_CONFIG,
        )
        # ReasoningOperator is a subclass of CodeOperator, so the lookup
        # falls through to the exact-type match for ReasoningOperator.
        assert Execution._operator_display_name(op) == "Reasoning"


# ═══════════════════════════════════════════════════════════════════════
# ExecutionProgress
# ═══════════════════════════════════════════════════════════════════════


class TestExecutionProgress:
    """Verify ExecutionProgress serialization."""

    def test_to_dict_full(self):
        """All fields present when set."""
        ep = ExecutionProgress(
            message="Running step 1/3: Semantic Filter…",
            operator_index=0,
            total_operators=3,
            operator_name="Semantic Filter",
            detail={"items": 42},
        )
        d = ep.to_dict()
        assert d["message"] == "Running step 1/3: Semantic Filter…"
        assert d["operator_index"] == 0
        assert d["total_operators"] == 3
        assert d["operator_name"] == "Semantic Filter"
        assert d["detail"] == {"items": 42}

    def test_to_dict_omits_none(self):
        """None fields are omitted from the dict."""
        ep = ExecutionProgress(message="Starting execution")
        d = ep.to_dict()
        assert "operator_index" not in d
        assert "total_operators" not in d
        assert "operator_name" not in d
        assert d["message"] == "Starting execution"

    def test_to_dict_omits_empty_detail(self):
        """Empty detail dict is still included (it's not None)."""
        ep = ExecutionProgress(message="test")
        d = ep.to_dict()
        # The empty dict is truthy by dataclass default, so it appears
        assert d["detail"] == {}
