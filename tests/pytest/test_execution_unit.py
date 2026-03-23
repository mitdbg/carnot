"""Unit tests for Execution helper methods and PlanNode/PhysicalPlan (no LLM required).

Tests cover:
1. ``PlanNode.to_operator`` — returns the correct physical operator
   type for every known operator name, and raises for unknowns.
2. ``PhysicalPlan.topo_order`` via ``from_plan_dict`` — produces the
   correct linearised order for single-op, chained, and branching
   (join) plan DAGs.
3. ``PlanNode.display_name`` and ``OPERATOR_DISPLAY_NAMES`` —
   returns human-readable labels.
4. ``ExecutionProgress.to_dict`` — serializes correctly.
"""

from __future__ import annotations

import pytest

from carnot.data.dataset import Dataset
from carnot.execution.progress import ExecutionProgress
from carnot.operators.code import CodeOperator
from carnot.operators.limit import LimitOperator
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator
from carnot.plan import PhysicalPlan
from carnot.plan.node import PlanNode

# ── Helpers ─────────────────────────────────────────────────────────────────

# Minimal llm_config so operators can be instantiated without env vars.
_LLM_CONFIG = {"OPENAI_API_KEY": "test-key-not-real"}


def _leaf_plan(name: str) -> dict:
    """Build a leaf plan node representing a raw Dataset reference.

    Requires:
        - *name* is a non-empty string.

    Returns:
        A dict in the canonical plan-dict format with no operator.

    Raises:
        None.
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
    """Build an operator plan node with given parents.

    Requires:
        - *operator* is a valid operator type string.
        - *output_id* is a non-empty string.

    Returns:
        A dict in the canonical plan-dict format.

    Raises:
        None.
    """
    params = {"operator": operator, **extra_params}
    return {
        "name": output_id,
        "output_dataset_id": output_id,
        "params": params,
        "parents": parents,
    }


def _make_node(
    operator_type: str,
    output_id: str = "out",
    parent_ids: list[str] | None = None,
    **extra_params,
) -> PlanNode:
    """Build a PlanNode for an operator with the given type and params.

    Requires:
        - *operator_type* is a valid operator type string.

    Returns:
        A ``PlanNode`` with ``node_type="operator"``.

    Raises:
        None.
    """
    return PlanNode(
        node_id=output_id,
        node_type="operator",
        operator_type=operator_type,
        name=output_id,
        description=f"test {operator_type}",
        params=extra_params,
        parent_ids=parent_ids or [],
        output_dataset_id=output_id,
    )


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.to_operator
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeToOperator:
    """Verify that each operator type maps to the correct physical class.

    These tests exercise ``PlanNode.to_operator()`` — the Phase 2
    replacement for the deleted ``Execution._get_op_from_plan_dict()``.
    """

    def test_code_operator(self):
        """'Code' → CodeOperator."""
        node = _make_node("Code", "code1", task="compute stats")
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, CodeOperator)

    def test_limit_operator(self):
        """'Limit' → LimitOperator."""
        node = _make_node("Limit", "limit1", n=5)
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, LimitOperator)

    def test_semantic_filter(self):
        """'SemanticFilter' → SemFilterOperator."""
        node = _make_node("SemanticFilter", "filter1", ["Movies"], condition="rating > 8")
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemFilterOperator)

    def test_semantic_map(self):
        """'SemanticMap' → SemMapOperator."""
        node = _make_node(
            "SemanticMap", "map1", ["Movies"],
            field="sentiment", type="str", field_desc="overall sentiment",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemMapOperator)

    def test_semantic_flat_map(self):
        """'SemanticFlatMap' → SemFlatMapOperator."""
        node = _make_node(
            "SemanticFlatMap", "flatmap1", ["Movies"],
            field="keyword", type="str", field_desc="extracted keyword",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemFlatMapOperator)

    def test_semantic_agg(self):
        """'SemanticAgg' → SemAggOperator."""
        node = _make_node(
            "SemanticAgg", "agg1", ["Movies"],
            task="summarize reviews",
            agg_fields=[{"name": "summary", "type": "str"}],
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemAggOperator)

    def test_semantic_groupby(self):
        """'SemanticGroupBy' → SemGroupByOperator."""
        node = _make_node(
            "SemanticGroupBy", "gby1", ["Movies"],
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemGroupByOperator)

    def test_semantic_join(self):
        """'SemanticJoin' → SemJoinOperator."""
        node = _make_node(
            "SemanticJoin", "join1", ["Movies", "Reviews"],
            condition="same movie",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemJoinOperator)

    def test_semantic_topk(self):
        """'SemanticTopK' → SemTopKOperator."""
        node = _make_node(
            "SemanticTopK", "topk1", ["Movies"],
            search_str="best action", k=5, index_name="flat",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemTopKOperator)

    def test_dataset_node_raises(self):
        """A dataset-type node raises ValueError when to_operator is called."""
        node = PlanNode(
            node_id="movies", node_type="dataset", operator_type=None,
            name="Movies", description="film data",
            output_dataset_id="Movies",
        )
        with pytest.raises(ValueError, match="Cannot create an operator"):
            node.to_operator(_LLM_CONFIG)

    def test_unknown_operator_raises(self):
        """An unrecognized operator type raises ValueError."""
        node = _make_node("UnknownOp", "unk1")
        with pytest.raises(ValueError, match="Unknown operator"):
            node.to_operator(_LLM_CONFIG)


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.topo_order (via from_plan_dict)
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanTopoOrder:
    """Verify topological linearization of plan DAGs.

    These tests exercise ``PhysicalPlan.from_plan_dict()`` +
    ``topo_order()`` — the Phase 2 replacement for the deleted
    ``Execution._get_ops_in_topological_order()``.
    """

    def test_single_leaf(self):
        """A single dataset plan yields one dataset node."""
        ds = Dataset(name="Movies")
        plan_dict = _leaf_plan("Movies")
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds], include_reasoning=False)
        nodes = pp.topo_order()
        assert len(nodes) == 1
        assert nodes[0].node_type == "dataset"
        assert nodes[0].name == "Movies"

    def test_linear_chain(self):
        """Dataset → Filter yields [Dataset, Filter] in order."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds], include_reasoning=False)
        nodes = pp.topo_order()
        assert len(nodes) == 2
        assert nodes[0].node_type == "dataset"
        assert nodes[1].operator_type == "SemanticFilter"

    def test_two_step_chain(self):
        """Dataset → Filter → Map yields three entries in correct order."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        filt = _op_plan("SemanticFilter", "filter1", [leaf], condition="rating > 8")
        mapped = _op_plan(
            "SemanticMap", "map1", [filt],
            field="label", type="str", field_desc="genre label",
        )
        pp = PhysicalPlan.from_plan_dict(mapped, [ds], include_reasoning=False)
        nodes = pp.topo_order()
        assert len(nodes) == 3
        assert nodes[0].node_type == "dataset"
        assert nodes[1].operator_type == "SemanticFilter"
        assert nodes[2].operator_type == "SemanticMap"

    def test_join_has_both_parents_before_join(self):
        """A join node is preceded by both parent datasets."""
        ds_a = Dataset(name="Movies")
        ds_b = Dataset(name="Reviews")

        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan_dict = _op_plan("SemanticJoin", "join1", [left, right], condition="same movie")

        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds_a, ds_b], include_reasoning=False)
        nodes = pp.topo_order()
        assert len(nodes) == 3
        # The join must be last
        assert nodes[2].operator_type == "SemanticJoin"
        # Both datasets appear before the join
        pre_join_types = {n.node_type for n in nodes[:2]}
        assert pre_join_types == {"dataset"}

    def test_parent_ids_propagated(self):
        """Parent node IDs are correctly threaded through the plan."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan("SemanticFilter", "filter1", [leaf], condition="x")
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds], include_reasoning=False)
        nodes = pp.topo_order()
        # The filter's parent_ids should reference the dataset node
        filter_node = nodes[1]
        assert len(filter_node.parent_ids) == 1
        # Parent should be the dataset node
        parent_node = pp.get_node(filter_node.parent_ids[0])
        assert parent_node.node_type == "dataset"


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.display_name and OPERATOR_DISPLAY_NAMES
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeDisplayName:
    """Verify that PlanNode.display_name() returns readable labels.

    These tests exercise ``PlanNode.display_name()`` and the
    ``OPERATOR_DISPLAY_NAMES`` mapping — the Phase 2 replacement
    for the deleted ``Execution._operator_display_name()``.
    """

    def test_dataset_includes_name(self):
        """A dataset node produces 'Dataset: <name>'."""
        node = PlanNode(
            node_id="movies", node_type="dataset", operator_type=None,
            name="Movies", description="film data",
            output_dataset_id="Movies",
        )
        assert node.display_name() == "Dataset: Movies"

    def test_sem_filter(self):
        """SemanticFilter → 'Semantic Filter'."""
        node = _make_node("SemanticFilter", condition="x")
        assert node.display_name() == "Semantic Filter"

    def test_sem_map(self):
        """SemanticMap → 'Semantic Map'."""
        node = _make_node("SemanticMap", field="x", type="str", field_desc="d")
        assert node.display_name() == "Semantic Map"

    def test_sem_join(self):
        """SemanticJoin → 'Semantic Join'."""
        node = _make_node("SemanticJoin", condition="c")
        assert node.display_name() == "Semantic Join"

    def test_limit(self):
        """Limit → 'Limit'."""
        node = _make_node("Limit", n=5)
        assert node.display_name() == "Limit"

    def test_code(self):
        """Code → 'Code'."""
        node = _make_node("Code", task="compute")
        assert node.display_name() == "Code"

    def test_sem_topk(self):
        """SemanticTopK → 'Semantic Top-K'."""
        node = _make_node("SemanticTopK", search_str="best", k=3, index_name="flat")
        assert node.display_name() == "Semantic Top-K"

    def test_sem_agg(self):
        """SemanticAgg → 'Semantic Aggregation'."""
        node = _make_node("SemanticAgg", task="summarize", agg_fields=[{"name": "s", "type": "str"}])
        assert node.display_name() == "Semantic Aggregation"

    def test_sem_groupby(self):
        """SemanticGroupBy → 'Semantic Group By'."""
        node = _make_node(
            "SemanticGroupBy",
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        assert node.display_name() == "Semantic Group By"

    def test_sem_flat_map(self):
        """SemanticFlatMap → 'Semantic Flat Map'."""
        node = _make_node("SemanticFlatMap", field="kw", type="str", field_desc="keyword")
        assert node.display_name() == "Semantic Flat Map"

    def test_reasoning(self):
        """Reasoning → 'Reasoning'."""
        node = PlanNode(
            node_id="reasoning", node_type="reasoning",
            operator_type="Reasoning", name="reasoning",
            description="synthesize results",
            output_dataset_id="final_dataset",
        )
        assert node.display_name() == "Reasoning"


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
        )
        d = ep.to_dict()
        assert d["message"] == "Running step 1/3: Semantic Filter…"
        assert d["operator_index"] == 0
        assert d["total_operators"] == 3
        assert d["operator_name"] == "Semantic Filter"

    def test_to_dict_omits_none(self):
        """None fields are omitted from the dict."""
        ep = ExecutionProgress(message="Starting execution")
        d = ep.to_dict()
        assert "operator_index" not in d
        assert "total_operators" not in d
        assert "operator_name" not in d
        assert d["message"] == "Starting execution"

# ═══════════════════════════════════════════════════════════════════════
# Execution.reoptimize
# ═══════════════════════════════════════════════════════════════════════


class TestReoptimize:
    """Verify ``Execution.reoptimize()`` contract.

    Postconditions (from docstring):
        - Node edits update params in-place.
        - Structural deletions remove nodes and rewire edges.
        - Structural insertions add nodes and rewire edges.
        - Returns a dict mapping edit labels to invalidated IDs.
    """

    @staticmethod
    def _make_execution_with_plan(plan: PhysicalPlan):
        """Create an Execution with a pre-built PhysicalPlan."""
        from carnot.execution.execution import Execution
        ds = Dataset(name="A")
        ex = Execution(query="test", datasets=[ds], llm_config=_LLM_CONFIG)
        ex._plan = plan
        return ex

    @staticmethod
    def _linear_plan() -> PhysicalPlan:
        """A → B(filter) → C(map): three-node linear plan."""
        a = PlanNode(
            node_id="a", node_type="dataset", operator_type=None,
            name="A", description="Load A", params={},
            parent_ids=[], output_dataset_id="A",
        )
        b = PlanNode(
            node_id="b", node_type="operator", operator_type="SemanticFilter",
            name="B", description="filter", params={"operator": "SemanticFilter", "condition": "old"},
            parent_ids=["a"], output_dataset_id="B",
        )
        c = PlanNode(
            node_id="c", node_type="operator", operator_type="SemanticMap",
            name="C", description="map",
            params={"operator": "SemanticMap", "field": "f", "type": "str", "field_desc": "d"},
            parent_ids=["b"], output_dataset_id="C",
        )
        return PhysicalPlan(nodes={"a": a, "b": b, "c": c})

    def test_edit_updates_params(self):
        """Node edit merges new params in-place."""
        from carnot.plan.feedback import NodeEdit, PlanFeedback

        plan = self._linear_plan()
        ex = self._make_execution_with_plan(plan)
        fb = PlanFeedback(node_edits=[
            NodeEdit(node_id="b", new_params={"condition": "new"}),
        ])

        result = ex.reoptimize(fb)

        assert plan.get_node("b").params["condition"] == "new"
        assert "edit:b" in result
        assert result["edit:b"] == ["c"]

    def test_delete_removes_node(self):
        """Structural delete removes node and rewires edges."""
        from carnot.plan.feedback import PlanFeedback, StructuralChange

        plan = self._linear_plan()
        ex = self._make_execution_with_plan(plan)
        fb = PlanFeedback(structural_changes=[
            StructuralChange(change_type="delete", node_id="b"),
        ])

        result = ex.reoptimize(fb)

        assert "b" not in plan
        assert plan.get_node("c").parent_ids == ["a"]
        assert "delete:b" in result

    def test_insert_adds_node(self):
        """Structural insert adds a node between two existing nodes."""
        from carnot.plan.feedback import PlanFeedback, StructuralChange

        plan = self._linear_plan()
        ex = self._make_execution_with_plan(plan)
        fb = PlanFeedback(structural_changes=[
            StructuralChange(
                change_type="insert",
                after_node_id="a",
                new_node_params={"operator": "Limit", "name": "lim", "n": 10},
            ),
        ])

        result = ex.reoptimize(fb)

        # Find the insert key
        insert_keys = [k for k in result if k.startswith("insert:")]
        assert len(insert_keys) == 1
        new_node_id = insert_keys[0].split(":")[1]

        # New node is in the plan
        assert new_node_id in plan
        # New node's parent is "a"
        assert plan.get_node(new_node_id).parent_ids == ["a"]
        # b now points to new node instead of a
        assert plan.get_node("b").parent_ids == [new_node_id]

    def test_combined_edit_and_delete(self):
        """Multiple feedback items applied in order."""
        from carnot.plan.feedback import NodeEdit, PlanFeedback, StructuralChange

        plan = self._linear_plan()
        ex = self._make_execution_with_plan(plan)
        fb = PlanFeedback(
            node_edits=[
                NodeEdit(node_id="c", new_params={"field": "new_f"}),
            ],
            structural_changes=[
                StructuralChange(change_type="delete", node_id="b"),
            ],
        )

        result = ex.reoptimize(fb)

        # Edit was applied
        assert plan.get_node("c").params["field"] == "new_f"
        assert "edit:c" in result
        # Delete was applied
        assert "b" not in plan
        assert "delete:b" in result
        # c now points to a
        assert plan.get_node("c").parent_ids == ["a"]

    def test_empty_feedback_returns_empty(self):
        """PlanFeedback with no edits → empty result dict."""
        from carnot.plan.feedback import PlanFeedback

        plan = self._linear_plan()
        ex = self._make_execution_with_plan(plan)
        fb = PlanFeedback()

        result = ex.reoptimize(fb)

        assert result == {}

    def test_node_from_params_generates_unique_id(self):
        """_node_from_params avoids collision with existing IDs."""
        from carnot.execution.execution import Execution

        # Build a plan where node-0 and node-1 already exist.
        n0 = PlanNode(
            node_id="node-0", node_type="dataset", operator_type=None,
            name="A", description="A", params={}, parent_ids=[],
            output_dataset_id="A",
        )
        n1 = PlanNode(
            node_id="node-1", node_type="operator",
            operator_type="SemanticFilter", name="B", description="B",
            params={"operator": "SemanticFilter", "condition": "x"},
            parent_ids=["node-0"], output_dataset_id="B",
        )
        plan = PhysicalPlan(nodes={"node-0": n0, "node-1": n1})

        new_node = Execution._node_from_params(
            {"operator": "Limit", "name": "lim"}, plan,
        )

        assert new_node.node_id == "node-2"  # node-0, node-1 taken
