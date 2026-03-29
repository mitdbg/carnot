"""Unit tests for :class:`PlanNode` and :class:`PhysicalPlan`.

Phase 1 of the Carnot library refactor (see
``docs/CARNOT_LIBRARY_REFACTOR.md``).  These are Tier 1 (no LLM) tests
that validate:

1. ``PlanNode`` — display names, code generation, cell serialisation,
   operator instantiation, and unified ``execute()``.
2. ``PhysicalPlan`` — construction from plan dicts, topological ordering,
   node lookup, cell serialisation, round-trip ``to_dict`` / ``from_dict``,
   and downstream invalidation.
"""

from __future__ import annotations

import pytest

from carnot.data.dataset import Dataset
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
from carnot.plan.node import OPERATOR_DISPLAY_NAMES, PlanNode
from carnot.plan.physical_plan import PhysicalPlan

# ── Shared helpers ───────────────────────────────────────────────────────────

_LLM_CONFIG = {"OPENAI_API_KEY": "test-key-not-real"}


def _leaf_plan(name: str) -> dict:
    """Build a leaf plan-dict node representing a raw Dataset."""
    return {
        "name": name,
        "dataset_id": name,
        "params": {},
        "parents": [],
    }


def _op_plan(
    operator: str,
    output_id: str,
    parents: list[dict],
    **extra_params,
) -> dict:
    """Build an operator plan-dict node with the given parents."""
    params = {"operator": operator, **extra_params}
    return {
        "name": output_id,
        "dataset_id": output_id,
        "params": params,
        "parents": parents,
    }


def _dataset_node(name: str, node_id: str = "node-0") -> PlanNode:
    """Create a minimal dataset PlanNode."""
    return PlanNode(
        node_id=node_id,
        node_type="dataset",
        operator_type=None,
        name=name,
        description=f"Load dataset: {name}",
        params={},
        parent_ids=[],
        dataset_id=name,
    )


def _operator_node(
    operator_type: str,
    output_id: str,
    parent_ids: list[str],
    node_id: str = "node-1",
    **extra_params,
) -> PlanNode:
    """Create a minimal operator PlanNode."""
    return PlanNode(
        node_id=node_id,
        node_type="operator",
        operator_type=operator_type,
        name=output_id,
        description=f"{operator_type} operation",
        params={"operator": operator_type, **extra_params},
        parent_ids=parent_ids,
        dataset_id=output_id,
    )


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.display_name
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeDisplayName:
    """Verify display_name() for every node type."""

    def test_dataset_node(self):
        """Dataset nodes produce 'Dataset: <name>'."""
        node = _dataset_node("Movies")
        assert node.display_name() == "Dataset: Movies"

    @pytest.mark.parametrize(
        "op_type,expected",
        list(OPERATOR_DISPLAY_NAMES.items()),
    )
    def test_known_operator_types(self, op_type: str, expected: str):
        """Every known operator type maps to its display name."""
        node = _operator_node(op_type, "out", ["p"], condition="x", task="t",
                              field="f", type="str", field_desc="d",
                              gby_fields=[], agg_fields=[], search_str="s",
                              k=5, n=10, index_name="flat")
        assert node.display_name() == expected

    def test_reasoning_node(self):
        """Reasoning nodes produce 'Reasoning'."""
        node = PlanNode(
            node_id="r", node_type="reasoning", operator_type="Reasoning",
            name="Reasoning", description="Generate final answer",
            parent_ids=["p"], dataset_id="final_dataset",
        )
        assert node.display_name() == "Reasoning"

    def test_unknown_operator_falls_back(self):
        """Unknown operator_type falls back to the raw string."""
        node = _operator_node("CustomOp", "out", ["p"])
        assert node.display_name() == "CustomOp"


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.to_operator
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeToOperator:
    """Verify to_operator() returns the correct operator class."""

    def test_code_operator(self):
        node = _operator_node("Code", "code1", [], task="compute stats")
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, CodeOperator)

    def test_limit_operator(self):
        node = _operator_node("Limit", "limit1", [], n=5)
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, LimitOperator)

    def test_semantic_filter(self):
        node = _operator_node("SemanticFilter", "f1", ["p"], condition="rating > 8")
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemFilterOperator)

    def test_semantic_map(self):
        node = _operator_node(
            "SemanticMap", "m1", ["p"],
            field="sentiment", type="str", field_desc="overall sentiment",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemMapOperator)

    def test_semantic_flat_map(self):
        node = _operator_node(
            "SemanticFlatMap", "fm1", ["p"],
            field="keyword", type="str", field_desc="extracted keyword",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemFlatMapOperator)

    def test_semantic_agg(self):
        node = _operator_node(
            "SemanticAgg", "agg1", ["p"],
            task="summarize reviews",
            agg_fields=[{"name": "summary", "type": "str"}],
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemAggOperator)

    def test_semantic_groupby(self):
        node = _operator_node(
            "SemanticGroupBy", "gby1", ["p"],
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemGroupByOperator)

    def test_semantic_join(self):
        node = _operator_node(
            "SemanticJoin", "j1", ["left", "right"],
            condition="same movie",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemJoinOperator)

    def test_semantic_topk(self):
        node = _operator_node(
            "SemanticTopK", "tk1", ["p"],
            search_str="best action", k=5, index_name="flat",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, SemTopKOperator)

    def test_reasoning_operator(self):
        node = PlanNode(
            node_id="r", node_type="reasoning", operator_type="Reasoning",
            name="Reasoning", description="Generate final answer",
            params={"task": "Summarize"},
            parent_ids=["p"], dataset_id="final_dataset",
        )
        op = node.to_operator(_LLM_CONFIG)
        assert isinstance(op, ReasoningOperator)

    def test_dataset_node_raises(self):
        """to_operator() raises for dataset nodes."""
        node = _dataset_node("Movies")
        with pytest.raises(ValueError, match="Cannot create an operator"):
            node.to_operator(_LLM_CONFIG)

    def test_unknown_operator_raises(self):
        """to_operator() raises for unknown operator types."""
        node = _operator_node("UnknownOp", "u1", [])
        with pytest.raises(ValueError, match="Unknown operator type"):
            node.to_operator(_LLM_CONFIG)


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.to_code
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeToCode:
    """Verify to_code() produces readable pseudocode."""

    def test_dataset_code(self):
        node = _dataset_node("Movies")
        code = node.to_code()
        assert "Load dataset: Movies" in code

    def test_reasoning_code(self):
        node = PlanNode(
            node_id="r", node_type="reasoning", operator_type="Reasoning",
            name="Reasoning", description="Generate final answer",
            params={"task": "Find mammals"},
            parent_ids=["p"], dataset_id="final_dataset",
        )
        code = node.to_code()
        assert "Final Reasoning" in code
        assert "Find mammals" in code

    def test_filter_code(self):
        node = _operator_node(
            "SemanticFilter", "filter1", ["Movies"],
            condition="rating > 8",
        )
        code = node.to_code()
        assert "sem_filter" in code
        assert "rating > 8" in code
        assert "filter1" in code

    def test_map_code(self):
        node = _operator_node(
            "SemanticMap", "map1", ["Movies"],
            field="sentiment", type="str", field_desc="overall sentiment",
        )
        code = node.to_code()
        assert "sem_map" in code
        assert "sentiment" in code

    def test_join_code(self):
        node = _operator_node(
            "SemanticJoin", "join1", ["Movies", "Reviews"],
            condition="same movie",
        )
        code = node.to_code()
        assert "sem_join" in code
        assert "Movies" in code
        assert "Reviews" in code
        assert "same movie" in code

    def test_limit_code(self):
        node = _operator_node("Limit", "limit1", ["Movies"], n=5)
        code = node.to_code()
        assert "limit" in code
        assert "n=5" in code

    def test_code_operator_code(self):
        node = _operator_node("Code", "code1", [], task="compute stats")
        code = node.to_code()
        assert "code_operator" in code
        assert "compute stats" in code

    def test_topk_code(self):
        node = _operator_node(
            "SemanticTopK", "topk1", ["Movies"],
            search_str="best action", k=5, index_name="flat",
        )
        code = node.to_code()
        assert "sem_topk" in code
        assert "best action" in code
        assert "k=5" in code

    def test_agg_code(self):
        node = _operator_node(
            "SemanticAgg", "agg1", ["Movies"],
            task="summarize reviews",
            agg_fields=[{"name": "summary", "type": "str"}],
        )
        code = node.to_code()
        assert "sem_agg" in code
        assert "summarize reviews" in code

    def test_flat_map_code(self):
        node = _operator_node(
            "SemanticFlatMap", "fm1", ["Movies"],
            field="keyword", type="str", field_desc="extracted keyword",
        )
        code = node.to_code()
        assert "sem_flat_map" in code
        assert "keyword" in code

    def test_groupby_code(self):
        node = _operator_node(
            "SemanticGroupBy", "gby1", ["Movies"],
            gby_fields=[{"name": "genre", "type": "str"}],
            agg_fields=[{"name": "count", "type": "int", "func": "count"}],
        )
        code = node.to_code()
        assert "sem_group_by" in code
        assert "genre" in code


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.to_dict
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeToDict:
    """Verify to_dict() produces a valid node descriptor."""

    def test_dict_has_required_keys(self):
        node = _operator_node("SemanticFilter", "f1", ["p"], condition="x")
        d = node.to_dict()
        required_keys = {
            "node_id", "node_type", "operator_name", "operator_type",
            "description", "code", "original_code", "params",
            "parent_dataset_ids", "dataset_id",
        }
        assert required_keys == set(d.keys())

    def test_dict_values_match_node(self):
        node = _operator_node("SemanticFilter", "f1", ["p"], condition="x")
        d = node.to_dict()
        assert d["node_id"] == "node-1"
        assert d["node_type"] == "operator"
        assert d["operator_type"] == "SemanticFilter"
        assert d["dataset_id"] == "f1"
        assert d["parent_dataset_ids"] == ["p"]

    def test_code_equals_original_code(self):
        """On fresh nodes, code and original_code are identical."""
        node = _dataset_node("Movies")
        d = node.to_dict()
        assert d["code"] == d["original_code"]
        assert "Load dataset: Movies" in d["code"]


# ═══════════════════════════════════════════════════════════════════════
# PlanNode.execute
# ═══════════════════════════════════════════════════════════════════════


class TestPlanNodeExecute:
    """Verify the unified execute() dispatch."""

    def test_dataset_node_materializes(self):
        """Dataset node adds the dataset to the store."""
        ds = Dataset(name="Animals", annotation="test", items=[{"a": 1}])
        node = _dataset_node("Animals")
        store, stats = node.execute(
            {}, _LLM_CONFIG, leaf_datasets={"Animals": ds},
        )
        assert "Animals" in store
        assert store["Animals"].items == [{"a": 1}]
        assert stats is None

    def test_dataset_node_missing_raises(self):
        """KeyError when the leaf dataset isn't provided."""
        node = _dataset_node("Missing")
        with pytest.raises(KeyError, match="Missing"):
            node.execute({}, _LLM_CONFIG, leaf_datasets={})

    def test_limit_operator_via_execute(self):
        """Limit node truncates items correctly via execute()."""
        ds = Dataset(
            name="Animals", annotation="test",
            items=[{"a": 1}, {"a": 2}, {"a": 3}],
        )
        node = _operator_node("Limit", "limited", ["ds-node"], n=2)
        store, stats = node.execute(
            {"Animals": ds}, _LLM_CONFIG,
            parent_output_ids=["Animals"],
        )
        assert "limited" in store
        assert len(store["limited"].items) == 2
        assert stats is not None
        assert stats.operator_name == "Limit"

    def test_execute_preserves_existing_store(self):
        """Execute merges new output into the existing store."""
        ds = Dataset(
            name="Animals", annotation="test",
            items=[{"a": 1}, {"a": 2}],
        )
        existing = {"other": Dataset(name="other", items=[{"b": 1}])}
        node = _operator_node("Limit", "limited", ["ds-node"], n=1)
        store, _ = node.execute(
            {**existing, "Animals": ds}, _LLM_CONFIG,
            parent_output_ids=["Animals"],
        )
        assert "other" in store
        assert "limited" in store
        assert "Animals" in store


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.from_plan_dict
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanFromPlanDict:
    """Verify construction from recursive plan dicts."""

    def test_single_leaf(self):
        """A single-dataset plan produces two nodes (dataset + reasoning)."""
        ds = Dataset(name="Movies")
        plan_dict = _leaf_plan("Movies")
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds], query="test")

        assert len(pp) == 2
        assert len(pp.leaf_nodes) == 1
        assert pp.leaf_nodes[0].name == "Movies"
        assert pp.reasoning_node is not None
        assert pp.query == "test"

    def test_single_leaf_no_reasoning(self):
        """With include_reasoning=False, no reasoning node is appended."""
        ds = Dataset(name="Movies")
        plan_dict = _leaf_plan("Movies")
        pp = PhysicalPlan.from_plan_dict(
            plan_dict, [ds], include_reasoning=False,
        )

        assert len(pp) == 1
        assert pp.reasoning_node is None

    def test_linear_chain(self):
        """Dataset → Filter produces 3 nodes with reasoning."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds])

        # dataset + filter + reasoning
        assert len(pp) == 3
        topo = pp.topo_order()
        assert topo[0].node_type == "dataset"
        assert topo[1].operator_type == "SemanticFilter"
        assert topo[2].node_type == "reasoning"

    def test_two_step_chain(self):
        """Dataset → Filter → Map produces 4 nodes with reasoning."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        filt = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )
        mapped = _op_plan(
            "SemanticMap", "map1", [filt],
            field="label", type="str", field_desc="genre label",
        )
        pp = PhysicalPlan.from_plan_dict(mapped, [ds])

        assert len(pp) == 4
        topo = pp.topo_order()
        types = [n.node_type for n in topo]
        assert types == ["dataset", "operator", "operator", "reasoning"]

    def test_join_plan(self):
        """A join plan has both parent datasets before the join."""
        ds_a = Dataset(name="Movies")
        ds_b = Dataset(name="Reviews")
        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan_dict = _op_plan(
            "SemanticJoin", "join1", [left, right], condition="same movie",
        )
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds_a, ds_b])

        # 2 datasets + join + reasoning
        assert len(pp) == 4
        topo = pp.topo_order()
        # Both datasets come before the join
        dataset_indices = [
            i for i, n in enumerate(topo) if n.node_type == "dataset"
        ]
        join_index = next(
            i for i, n in enumerate(topo) if n.operator_type == "SemanticJoin"
        )
        assert all(di < join_index for di in dataset_indices)

    def test_parent_ids_are_node_ids(self):
        """Parent IDs in PlanNodes reference node_ids, not dataset_ids."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="x",
        )
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds], include_reasoning=False)

        topo = pp.topo_order()
        dataset_node = topo[0]
        filter_node = topo[1]
        assert filter_node.parent_ids == [dataset_node.node_id]


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.topo_order
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanTopoOrder:
    """Verify topological ordering."""

    def test_single_node(self):
        """A plan with one node returns that node."""
        node = _dataset_node("Movies")
        pp = PhysicalPlan(nodes={node.node_id: node})
        topo = pp.topo_order()
        assert len(topo) == 1
        assert topo[0] is node

    def test_linear_order(self):
        """Linear chain is returned in dependency order."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1", condition="x")
        n2 = _operator_node("SemanticMap", "C", ["n1"], node_id="n2",
                            field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})
        topo = pp.topo_order()
        ids = [n.node_id for n in topo]
        assert ids == ["n0", "n1", "n2"]

    def test_diamond_dag(self):
        """Diamond DAG: A→B, A→C, B→D, C→D yields valid topo order."""
        a = _dataset_node("A", node_id="a")
        b = _operator_node("SemanticFilter", "B", ["a"], node_id="b", condition="x")
        c = _operator_node("SemanticMap", "C", ["a"], node_id="c",
                           field="f", type="str", field_desc="d")
        d = _operator_node("SemanticJoin", "D", ["b", "c"], node_id="d", condition="j")
        pp = PhysicalPlan(nodes={"a": a, "b": b, "c": c, "d": d})
        topo = pp.topo_order()

        ids = [n.node_id for n in topo]
        # A must come first, D must come last
        assert ids[0] == "a"
        assert ids[-1] == "d"
        # B and C must both be before D
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.get_node and related accessors
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanAccessors:
    """Verify node lookup, contains, len, children_of."""

    def test_get_node_success(self):
        node = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": node})
        assert pp.get_node("n0") is node

    def test_get_node_missing_raises(self):
        pp = PhysicalPlan()
        with pytest.raises(KeyError, match="not found"):
            pp.get_node("missing")

    def test_contains(self):
        node = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": node})
        assert "n0" in pp
        assert "n1" not in pp

    def test_len(self):
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "B", ["n0"], node_id="n1", n=5)
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})
        assert len(pp) == 2

    def test_children_of(self):
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "B", ["n0"], node_id="n1", n=5)
        n2 = _operator_node("SemanticFilter", "C", ["n0"], node_id="n2", condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})
        children = pp.children_of("n0")
        child_ids = {c.node_id for c in children}
        assert child_ids == {"n1", "n2"}

    def test_children_of_leaf_is_empty(self):
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "B", ["n0"], node_id="n1", n=5)
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})
        assert pp.children_of("n1") == []


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.invalidated_downstream
# ═══════════════════════════════════════════════════════════════════════


class TestInvalidatedDownstream:
    """Verify transitive downstream closure."""

    def test_leaf_has_all_downstream(self):
        """Invalidating the leaf invalidates every other node."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "B", ["n0"], node_id="n1", n=5)
        n2 = _operator_node("SemanticFilter", "C", ["n1"], node_id="n2", condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})
        invalid = pp.invalidated_downstream("n0")
        assert set(invalid) == {"n1", "n2"}

    def test_middle_node(self):
        """Invalidating a middle node only includes downstream."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "B", ["n0"], node_id="n1", n=5)
        n2 = _operator_node("SemanticFilter", "C", ["n1"], node_id="n2", condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})
        invalid = pp.invalidated_downstream("n1")
        assert set(invalid) == {"n2"}

    def test_terminal_node_has_no_downstream(self):
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "B", ["n0"], node_id="n1", n=5)
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})
        assert pp.invalidated_downstream("n1") == []

    def test_diamond_invalidation(self):
        """Invalidating root of diamond → all downstream."""
        a = _dataset_node("A", node_id="a")
        b = _operator_node("SemanticFilter", "B", ["a"], node_id="b", condition="x")
        c = _operator_node("SemanticMap", "C", ["a"], node_id="c",
                           field="f", type="str", field_desc="d")
        d = _operator_node("SemanticJoin", "D", ["b", "c"], node_id="d", condition="j")
        pp = PhysicalPlan(nodes={"a": a, "b": b, "c": c, "d": d})

        assert set(pp.invalidated_downstream("a")) == {"b", "c", "d"}
        assert set(pp.invalidated_downstream("b")) == {"d"}
        assert set(pp.invalidated_downstream("c")) == {"d"}
        assert pp.invalidated_downstream("d") == []


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.to_node_dicts
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanToNodeDicts:
    """Verify to_node_dicts() serialisation."""

    def test_to_node_dicts_count(self):
        """to_node_dicts returns one dict per node in topo order."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds])
        dicts = pp.to_node_dicts()
        assert len(dicts) == len(pp)

    def test_to_node_dicts_order_matches_topo(self):
        """Dict order matches topological order."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="x",
        )
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds])
        dicts = pp.to_node_dicts()
        topo = pp.topo_order()
        for d, node in zip(dicts, topo, strict=True):
            assert d["node_id"] == node.node_id

    def test_node_types(self):
        """First is dataset, middle is operator, last is reasoning."""
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="x",
        )
        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds])
        dicts = pp.to_node_dicts()
        assert dicts[0]["node_type"] == "dataset"
        assert dicts[1]["node_type"] == "operator"
        assert dicts[2]["node_type"] == "reasoning"


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan round-trip: to_dict / from_dict
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanRoundTrip:
    """Verify to_dict() → from_dict() round-trip."""

    def test_round_trip_preserves_structure(self):
        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )
        original = PhysicalPlan.from_plan_dict(plan_dict, [ds], query="test query")

        serialized = original.to_dict()
        restored = PhysicalPlan.from_dict(serialized)

        assert len(restored) == len(original)
        assert restored.query == original.query

        for orig_node, rest_node in zip(
            original.topo_order(), restored.topo_order(), strict=True,
        ):
            assert orig_node.node_id == rest_node.node_id
            assert orig_node.node_type == rest_node.node_type
            assert orig_node.operator_type == rest_node.operator_type
            assert orig_node.name == rest_node.name
            assert orig_node.params == rest_node.params
            assert orig_node.parent_ids == rest_node.parent_ids
            assert orig_node.dataset_id == rest_node.dataset_id

    def test_round_trip_join_plan(self):
        """Round-trip preserves a join plan with two parents."""
        ds_a = Dataset(name="Movies")
        ds_b = Dataset(name="Reviews")
        left = _leaf_plan("Movies")
        right = _leaf_plan("Reviews")
        plan_dict = _op_plan(
            "SemanticJoin", "join1", [left, right], condition="same movie",
        )
        original = PhysicalPlan.from_plan_dict(plan_dict, [ds_a, ds_b])

        restored = PhysicalPlan.from_dict(original.to_dict())
        assert len(restored) == len(original)

        # The join node should still have two parent_ids
        join_nodes = [
            n for n in restored.topo_order()
            if n.operator_type == "SemanticJoin"
        ]
        assert len(join_nodes) == 1
        assert len(join_nodes[0].parent_ids) == 2

    def test_round_trip_preserves_all_fields(self):
        """All node fields survive the round-trip."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("Limit", "out", ["n0"], node_id="n1", n=5)
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})
        restored = PhysicalPlan.from_dict(pp.to_dict())
        orig = pp.get_node("n1")
        rest = restored.get_node("n1")
        assert rest.node_id == orig.node_id
        assert rest.operator_type == orig.operator_type
        assert rest.params == orig.params


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.from_plan_dict vs. Execution._get_ops_in_topological_order
# compatibility check
# ═══════════════════════════════════════════════════════════════════════


class TestPhysicalPlanConsistencyWithExecution:
    """Verify that Execution correctly delegates to PhysicalPlan after
    the Phase 2 rewiring.

    These tests confirm that:
    - Setting ``execution._plan`` with a dict auto-converts to PhysicalPlan.
    - ``execution.get_physical_plan()`` delegates to plan.to_cells().
    - The plan property round-trips correctly.
    """

    def test_plan_property_converts_dict_to_physical_plan(self):
        """Assigning a dict to ``_plan`` creates a PhysicalPlan."""
        from carnot.execution.execution import Execution

        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )

        ex = Execution(query="test", datasets=[ds], llm_config=_LLM_CONFIG)
        ex._plan = plan_dict

        assert isinstance(ex._physical_plan, PhysicalPlan)
        nodes = ex._physical_plan.topo_order()
        # Dataset + Filter + Reasoning
        assert len(nodes) == 3
        assert nodes[0].node_type == "dataset"
        assert nodes[1].operator_type == "SemanticFilter"
        assert nodes[2].node_type == "reasoning"

    def test_plan_property_accepts_physical_plan(self):
        """Assigning a PhysicalPlan to ``_plan`` stores it directly."""
        from carnot.execution.execution import Execution

        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )

        pp = PhysicalPlan.from_plan_dict(plan_dict, [ds], query="test")
        ex = Execution(query="test", datasets=[ds], llm_config=_LLM_CONFIG)
        ex._plan = pp

        assert ex._physical_plan is pp

    def test_get_physical_plan_delegates_to_to_node_dicts(self):
        """``get_physical_plan()`` returns node descriptors from the plan."""
        from carnot.execution.execution import Execution

        ds = Dataset(name="Movies")
        leaf = _leaf_plan("Movies")
        plan_dict = _op_plan(
            "SemanticFilter", "filter1", [leaf], condition="rating > 8",
        )

        ex = Execution(
            query="test", datasets=[ds], plan=plan_dict,
            llm_config=_LLM_CONFIG,
        )
        dicts = ex.get_physical_plan()

        assert isinstance(dicts, list)
        assert len(dicts) == 3  # Dataset + Filter + Reasoning
        # Dicts should have node_id, node_type keys
        for d in dicts:
            assert "node_id" in d
            assert "node_type" in d
            assert "code" in d


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.edit_node
# ═══════════════════════════════════════════════════════════════════════


class TestEditNode:
    """Verify ``PhysicalPlan.edit_node()`` contract.

    Postconditions (from docstring):
        - Merges *new_params* into existing ``params`` dict.
        - Returns sorted downstream IDs (not including *node_id*).
        - Raises ``KeyError`` if *node_id* not found.
        - Raises ``ValueError`` if *new_params* is empty.
    """

    def test_updates_params_in_place(self):
        """Existing params are merged with new ones."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node(
            "SemanticFilter", "B", ["n0"], node_id="n1",
            condition="old condition",
        )
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        pp.edit_node("n1", {"condition": "new condition"})

        assert pp.get_node("n1").params["condition"] == "new condition"
        # operator key should be preserved
        assert pp.get_node("n1").params["operator"] == "SemanticFilter"

    def test_adds_new_keys(self):
        """Keys not in the original params are added."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        pp.edit_node("n1", {"new_key": "new_value"})

        assert pp.get_node("n1").params["new_key"] == "new_value"
        assert pp.get_node("n1").params["condition"] == "x"

    def test_returns_downstream_ids(self):
        """Returns sorted list of downstream node IDs."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        n2 = _operator_node("SemanticMap", "C", ["n1"], node_id="n2",
                            field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})

        result = pp.edit_node("n1", {"condition": "y"})

        assert result == ["n2"]

    def test_terminal_node_returns_empty(self):
        """Editing a terminal node has no downstream."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        result = pp.edit_node("n1", {"condition": "y"})

        assert result == []

    def test_raises_key_error_for_missing_node(self):
        n0 = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": n0})

        with pytest.raises(KeyError):
            pp.edit_node("nonexistent", {"x": 1})

    def test_raises_value_error_for_empty_params(self):
        n0 = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": n0})

        with pytest.raises(ValueError, match="non-empty"):
            pp.edit_node("n0", {})

    def test_diamond_edit_returns_all_downstream(self):
        """Editing the root of a diamond returns all downstream."""
        a = _dataset_node("A", node_id="a")
        b = _operator_node("SemanticFilter", "B", ["a"], node_id="b",
                           condition="x")
        c = _operator_node("SemanticMap", "C", ["a"], node_id="c",
                           field="f", type="str", field_desc="d")
        d = _operator_node("SemanticJoin", "D", ["b", "c"], node_id="d",
                           condition="j")
        pp = PhysicalPlan(nodes={"a": a, "b": b, "c": c, "d": d})

        result = pp.edit_node("a", {"extra": "data"})

        assert set(result) == {"b", "c", "d"}


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.insert_node
# ═══════════════════════════════════════════════════════════════════════


class TestInsertNode:
    """Verify ``PhysicalPlan.insert_node()`` contract.

    Postconditions (from docstring):
        - New node is in the plan.
        - ``new_node.parent_ids == [after_node_id]``.
        - Former children of after_node now have new_node.node_id in
          their parent_ids in place of after_node_id.
        - Returns sorted downstream IDs of the new node.
        - Raises ``KeyError`` if after_node_id not found.
        - Raises ``ValueError`` if new_node.node_id already exists.
    """

    def test_simple_insert_between_two_nodes(self):
        """Insert a node between parent and child."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        pp.insert_node("n0", new)

        # new_node parents should be set to [after_node_id]
        assert new.parent_ids == ["n0"]
        # Former child n1 should now point to new node
        assert pp.get_node("n1").parent_ids == ["n-new"]
        # New node is in the plan
        assert "n-new" in pp

    def test_insert_returns_downstream(self):
        """Returns downstream of the newly inserted node."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        n2 = _operator_node("SemanticMap", "C", ["n1"], node_id="n2",
                            field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        result = pp.insert_node("n0", new)

        # n1 and n2 are downstream of new
        assert set(result) == {"n1", "n2"}

    def test_insert_at_terminal_no_downstream(self):
        """Insert after the last node → no downstream."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        result = pp.insert_node("n1", new)

        assert result == []
        assert new.parent_ids == ["n1"]
        # Original edges unchanged for n1
        assert pp.get_node("n1").parent_ids == ["n0"]

    def test_insert_in_diamond_rewires_multiple_children(self):
        """Insert after node with multiple children rewires all."""
        a = _dataset_node("A", node_id="a")
        b = _operator_node("SemanticFilter", "B", ["a"], node_id="b",
                           condition="x")
        c = _operator_node("SemanticMap", "C", ["a"], node_id="c",
                           field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"a": a, "b": b, "c": c})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        pp.insert_node("a", new)

        # Both b and c should now point to n-new
        assert pp.get_node("b").parent_ids == ["n-new"]
        assert pp.get_node("c").parent_ids == ["n-new"]
        assert new.parent_ids == ["a"]

    def test_topo_order_valid_after_insert(self):
        """Topological order is valid after insertion."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        pp.insert_node("n0", new)

        order = [n.node_id for n in pp.topo_order()]
        assert order.index("n0") < order.index("n-new")
        assert order.index("n-new") < order.index("n1")

    def test_plan_size_increases(self):
        """Plan has one more node after insertion."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        pp.insert_node("n0", new)

        assert len(pp) == 3

    def test_raises_key_error_for_missing_after_node(self):
        n0 = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": n0})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        with pytest.raises(KeyError):
            pp.insert_node("nonexistent", new)

    def test_raises_value_error_for_duplicate_node_id(self):
        n0 = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": n0})

        new = _operator_node("Limit", "lim", [], node_id="n0", n=10)
        with pytest.raises(ValueError, match="already exists"):
            pp.insert_node("n0", new)

    def test_insert_only_rewires_after_node_children(self):
        """Insert after b does not rewire c (child of a only)."""
        a = _dataset_node("A", node_id="a")
        b = _operator_node("SemanticFilter", "B", ["a"], node_id="b",
                           condition="x")
        c = _operator_node("SemanticMap", "C", ["a"], node_id="c",
                           field="f", type="str", field_desc="d")
        d = _operator_node("SemanticJoin", "D", ["b", "c"], node_id="d",
                           condition="j")
        pp = PhysicalPlan(nodes={"a": a, "b": b, "c": c, "d": d})

        new = _operator_node("Limit", "lim", [], node_id="n-new", n=10)
        pp.insert_node("b", new)

        # d was the only child of b; it should now point to n-new for
        # the b slot, and still point to c for the c slot.
        assert pp.get_node("d").parent_ids == ["n-new", "c"]
        # c is unaffected
        assert pp.get_node("c").parent_ids == ["a"]


# ═══════════════════════════════════════════════════════════════════════
# PhysicalPlan.delete_node
# ═══════════════════════════════════════════════════════════════════════


class TestDeleteNode:
    """Verify ``PhysicalPlan.delete_node()`` contract.

    Postconditions (from docstring):
        - The node is removed from the plan.
        - Former children have their parent_ids rewired to the deleted
          node's parents.
        - Returns sorted downstream IDs (computed before deletion).
        - Raises ``KeyError`` if *node_id* not found.
        - Raises ``ValueError`` if node is a dataset node.
    """

    def test_simple_delete_rewires_child(self):
        """Delete middle node; child now points to grandparent."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        n2 = _operator_node("SemanticMap", "C", ["n1"], node_id="n2",
                            field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})

        pp.delete_node("n1")

        # n1 removed
        assert "n1" not in pp
        # n2 now points to n0
        assert pp.get_node("n2").parent_ids == ["n0"]

    def test_delete_returns_downstream(self):
        """Returns downstream IDs computed before deletion."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        n2 = _operator_node("SemanticMap", "C", ["n1"], node_id="n2",
                            field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})

        result = pp.delete_node("n1")

        assert result == ["n2"]

    def test_delete_terminal_operator(self):
        """Delete the last operator; no rewiring needed."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        result = pp.delete_node("n1")

        assert "n1" not in pp
        assert result == []
        assert len(pp) == 1

    def test_delete_with_multiple_children(self):
        """Delete a node that has two children; both rewire."""
        a = _dataset_node("A", node_id="a")
        b = _operator_node("SemanticFilter", "B", ["a"], node_id="b",
                           condition="x")
        c = _operator_node("SemanticMap", "C", ["b"], node_id="c",
                           field="f", type="str", field_desc="d")
        d = _operator_node("Limit", "D", ["b"], node_id="d", n=5)
        pp = PhysicalPlan(nodes={"a": a, "b": b, "c": c, "d": d})

        pp.delete_node("b")

        assert "b" not in pp
        assert pp.get_node("c").parent_ids == ["a"]
        assert pp.get_node("d").parent_ids == ["a"]

    def test_delete_node_with_multiple_parents(self):
        """Delete a join-like node that has two parents."""
        a = _dataset_node("A", node_id="a")
        b = _dataset_node("B", node_id="b")
        j = _operator_node("SemanticJoin", "J", ["a", "b"], node_id="j",
                           condition="x")
        r = _operator_node("Reasoning", "R", ["j"], node_id="r")
        pp = PhysicalPlan(nodes={"a": a, "b": b, "j": j, "r": r})

        pp.delete_node("j")

        # r should now point to both a and b
        assert set(pp.get_node("r").parent_ids) == {"a", "b"}
        assert "j" not in pp

    def test_topo_order_valid_after_delete(self):
        """Topological order is still valid after deletion."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        n2 = _operator_node("SemanticMap", "C", ["n1"], node_id="n2",
                            field="f", type="str", field_desc="d")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1, "n2": n2})

        pp.delete_node("n1")

        order = [n.node_id for n in pp.topo_order()]
        assert order == ["n0", "n2"]

    def test_plan_size_decreases(self):
        """Plan has one fewer node after deletion."""
        n0 = _dataset_node("A", node_id="n0")
        n1 = _operator_node("SemanticFilter", "B", ["n0"], node_id="n1",
                            condition="x")
        pp = PhysicalPlan(nodes={"n0": n0, "n1": n1})

        pp.delete_node("n1")

        assert len(pp) == 1

    def test_raises_key_error_for_missing_node(self):
        n0 = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": n0})

        with pytest.raises(KeyError):
            pp.delete_node("nonexistent")

    def test_raises_value_error_for_dataset_node(self):
        n0 = _dataset_node("A", node_id="n0")
        pp = PhysicalPlan(nodes={"n0": n0})

        with pytest.raises(ValueError, match="dataset"):
            pp.delete_node("n0")
