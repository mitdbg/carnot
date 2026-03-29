"""Tests for the CostEntry-based optimizer pipeline: pareto bookkeeping,
``OptimizePhysicalExpression.perform``, and ``Optimizer._select_best_plan``.

These are **Tier 1** (pure unit) tests — no network calls or LLM mocks needed.
All optimizer primitives are constructed in-memory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.core.models import PlanCost
from carnot.data.dataset import Dataset
from carnot.data.item import DataItem
from carnot.operators.scan import ScanOp
from carnot.optimizer.optimizer import (
    _SCORERS,
    Optimizer,
    _expr_to_plan_node,
    _get_scorer,
)
from carnot.optimizer.primitives import (
    CostEntry,
    Group,
    PhysicalExpression,
)
from carnot.optimizer.tasks import (
    OptimizePhysicalExpression,
    _dominates,
    _pareto_filter,
    _update_group_frontier,
)

# ── Helpers ───────────────────────────────────────────────────────────


def _pc(cost: float = 0.0, time: float = 0.0, quality: float = 1.0) -> PlanCost:
    """Convenience: build a PlanCost with explicit quality via token trick.

    ``quality = scanned / total``.  We set ``total_input_tokens = 1``
    so ``total_scanned_input_tokens = quality``.
    """
    return PlanCost(
        cost=cost,
        time=time,
        total_input_tokens=1.0,
        total_scanned_input_tokens=quality,
    )


def _make_scan_expr(dataset_id: str = "ds", group_id: int = 0, num_items: int = 10) -> PhysicalExpression:
    """Create a PhysicalExpression wrapping a ScanOp."""
    op = ScanOp(dataset_id=dataset_id, num_items=num_items, est_tokens_per_item=5.0)
    return PhysicalExpression(operator=op, input_group_ids=[], group_id=group_id)


def _make_optimizer() -> Optimizer:
    """Minimal optimizer with mock model."""
    model = MagicMock()
    return Optimizer(model=model, available_model_ids=["test-model"])


# ── _dominates ────────────────────────────────────────────────────────


class TestDominates:
    """Tests for the module-level ``_dominates`` helper."""

    def test_identical_does_not_dominate(self):
        """A cost does not dominate itself (must be *strictly* better on at least one)."""
        pc = _pc(cost=1.0, time=1.0, quality=0.5)
        assert not _dominates(pc, pc)

    def test_strictly_better_on_cost(self):
        """Lower cost with same time and quality → dominates."""
        a = _pc(cost=0.5, time=1.0, quality=0.5)
        b = _pc(cost=1.0, time=1.0, quality=0.5)
        assert _dominates(a, b)

    def test_strictly_better_on_time(self):
        """Lower time with same cost and quality → dominates."""
        a = _pc(cost=1.0, time=0.5, quality=0.5)
        b = _pc(cost=1.0, time=1.0, quality=0.5)
        assert _dominates(a, b)

    def test_strictly_better_on_quality(self):
        """Higher quality with same cost and time → dominates."""
        a = _pc(cost=1.0, time=1.0, quality=0.8)
        b = _pc(cost=1.0, time=1.0, quality=0.5)
        assert _dominates(a, b)

    def test_trade_off_does_not_dominate(self):
        """Better cost but worse time → no domination."""
        a = _pc(cost=0.5, time=2.0, quality=0.5)
        b = _pc(cost=1.0, time=1.0, quality=0.5)
        assert not _dominates(a, b)

    def test_worse_on_all_does_not_dominate(self):
        """Worse on every dimension → does not dominate."""
        a = _pc(cost=2.0, time=2.0, quality=0.3)
        b = _pc(cost=1.0, time=1.0, quality=0.5)
        assert not _dominates(a, b)


# ── _pareto_filter ────────────────────────────────────────────────────


class TestParetoFilter:
    """Tests for the module-level ``_pareto_filter`` helper."""

    def test_single_entry_returned(self):
        """A single entry is always on the frontier."""
        e = CostEntry(entry_id=0, plan_cost=_pc(cost=1.0, time=1.0), input_entry_ids=())
        result = _pareto_filter([e])
        assert result == [e]

    def test_dominated_entry_removed(self):
        """An entry dominated on all dimensions is removed."""
        good = CostEntry(entry_id=0, plan_cost=_pc(cost=0.5, time=0.5, quality=0.8), input_entry_ids=())
        bad = CostEntry(entry_id=1, plan_cost=_pc(cost=1.0, time=1.0, quality=0.5), input_entry_ids=())
        result = _pareto_filter([good, bad])
        assert result == [good]

    def test_trade_off_both_kept(self):
        """Two entries with trade-offs are both on the frontier."""
        fast = CostEntry(entry_id=0, plan_cost=_pc(cost=2.0, time=0.5), input_entry_ids=())
        cheap = CostEntry(entry_id=1, plan_cost=_pc(cost=0.5, time=2.0), input_entry_ids=())
        result = _pareto_filter([fast, cheap])
        assert len(result) == 2
        assert set(e.entry_id for e in result) == {0, 1}

    def test_multiple_dominated_removed(self):
        """Only non-dominated entries survive."""
        best = CostEntry(entry_id=0, plan_cost=_pc(cost=0.1, time=0.1, quality=0.9), input_entry_ids=())
        mid = CostEntry(entry_id=1, plan_cost=_pc(cost=0.5, time=0.5, quality=0.5), input_entry_ids=())
        worst = CostEntry(entry_id=2, plan_cost=_pc(cost=1.0, time=1.0, quality=0.3), input_entry_ids=())
        result = _pareto_filter([best, mid, worst])
        assert [e.entry_id for e in result] == [0]


# ── _update_group_frontier ────────────────────────────────────────────


class TestUpdateGroupFrontier:
    """Tests for the module-level ``_update_group_frontier`` helper."""

    def _make_parts(self):
        """Return (group, expr, optimizer_mock) for frontier tests."""
        optimizer = MagicMock()
        optimizer.cost_entries = {}

        group = Group(logical_expressions=[], group_id=0)
        expr = _make_scan_expr(group_id=0)
        return group, expr, optimizer

    def test_empty_frontier_initialized(self):
        """When pareto_frontier is None, the new entries become the frontier."""
        group, expr, opt = self._make_parts()
        entry = CostEntry(entry_id=0, plan_cost=_pc(cost=1.0, time=1.0), input_entry_ids=())
        opt.cost_entries[0] = entry

        _update_group_frontier(group, expr, [entry], opt)
        assert group.pareto_frontier == [(expr.expr_id, 0)]

    def test_merge_keeps_non_dominated(self):
        """Merging a cheaper entry with an existing one keeps both if they trade off."""
        group, expr, opt = self._make_parts()

        old_entry = CostEntry(entry_id=0, plan_cost=_pc(cost=0.5, time=2.0), input_entry_ids=())
        opt.cost_entries[0] = old_entry
        group.pareto_frontier = [(expr.expr_id, 0)]

        new_entry = CostEntry(entry_id=1, plan_cost=_pc(cost=2.0, time=0.5), input_entry_ids=())
        opt.cost_entries[1] = new_entry

        new_expr = _make_scan_expr(dataset_id="ds2", group_id=0)
        _update_group_frontier(group, new_expr, [new_entry], opt)

        # Both should survive (trade-off).
        assert len(group.pareto_frontier) == 2
        entry_ids = {eid for _, eid in group.pareto_frontier}
        assert entry_ids == {0, 1}

    def test_merge_removes_dominated(self):
        """A dominated entry on the old frontier is removed when a better entry arrives."""
        group, expr, opt = self._make_parts()

        old_entry = CostEntry(entry_id=0, plan_cost=_pc(cost=1.0, time=1.0, quality=0.5), input_entry_ids=())
        opt.cost_entries[0] = old_entry
        group.pareto_frontier = [(expr.expr_id, 0)]

        better = CostEntry(entry_id=1, plan_cost=_pc(cost=0.5, time=0.5, quality=0.8), input_entry_ids=())
        opt.cost_entries[1] = better

        new_expr = _make_scan_expr(dataset_id="ds2", group_id=0)
        _update_group_frontier(group, new_expr, [better], opt)

        assert len(group.pareto_frontier) == 1
        assert group.pareto_frontier[0][1] == 1  # better entry


# ── OptimizePhysicalExpression.perform ────────────────────────────────


class TestOptimizePhysicalExpressionPerform:
    """Tests for the CostEntry-based ``perform()`` on scan-only plans."""

    @pytest.fixture
    def optimizer(self):
        return _make_optimizer()

    def test_scan_creates_cost_entry(self, optimizer):
        """Performing on a ScanOp creates exactly one CostEntry."""
        expr = _make_scan_expr(group_id=0)
        group = Group(logical_expressions=[], group_id=0)
        group.physical_expressions.add(expr)
        optimizer.groups[0] = group
        optimizer.expressions[expr.expr_id] = expr

        task = OptimizePhysicalExpression(expr)
        new_tasks = task.perform(optimizer.cost_model, optimizer.groups, optimizer)

        # No further tasks needed for a leaf scan.
        assert new_tasks == []
        # Exactly one cost entry was created.
        assert len(optimizer.cost_entries) == 1
        # Expression has one pareto entry.
        assert len(expr.pareto_entry_ids) == 1
        # Group frontier is set.
        assert group.pareto_frontier is not None
        assert len(group.pareto_frontier) == 1

    def test_scan_cost_entry_values(self, optimizer):
        """The CostEntry for a ScanOp has cost=0, time=0, and correct token counts."""
        op = ScanOp(dataset_id="ds", num_items=20, est_tokens_per_item=10.0)
        expr = PhysicalExpression(operator=op, input_group_ids=[], group_id=0)
        group = Group(logical_expressions=[], group_id=0)
        group.physical_expressions.add(expr)
        optimizer.groups[0] = group
        optimizer.expressions[expr.expr_id] = expr

        task = OptimizePhysicalExpression(expr)
        task.perform(optimizer.cost_model, optimizer.groups, optimizer)

        entry = optimizer.cost_entries[expr.pareto_entry_ids[0]]
        assert entry.plan_cost.cost == 0.0
        assert entry.plan_cost.time == 0.0
        assert entry.plan_cost.total_input_tokens == 200.0  # 20 * 10

    def test_already_costed_is_noop(self, optimizer):
        """If the expression is already costed, perform() is a no-op."""
        expr = _make_scan_expr(group_id=0)
        expr.pareto_entry_ids = [42]  # mark as already costed
        group = Group(logical_expressions=[], group_id=0)
        optimizer.groups[0] = group

        task = OptimizePhysicalExpression(expr)
        new_tasks = task.perform(optimizer.cost_model, optimizer.groups, optimizer)
        assert new_tasks == []
        assert len(optimizer.cost_entries) == 0  # nothing added

    def test_sets_group_optimized(self, optimizer):
        """After perform(), the owning group is marked optimized."""
        expr = _make_scan_expr(group_id=0)
        group = Group(logical_expressions=[], group_id=0)
        group.physical_expressions.add(expr)
        optimizer.groups[0] = group
        optimizer.expressions[expr.expr_id] = expr

        task = OptimizePhysicalExpression(expr)
        task.perform(optimizer.cost_model, optimizer.groups, optimizer)
        assert optimizer.groups[0].optimized is True


# ── Scorers ───────────────────────────────────────────────────────────


class TestScorers:
    """Tests for ``_SCORERS`` and ``_get_scorer``."""

    def test_available_policies(self):
        """All three standard policies are registered."""
        assert set(_SCORERS) == {"min_cost", "min_time", "max_quality"}

    def test_get_scorer_valid(self):
        """_get_scorer returns a callable for each valid policy."""
        for policy in ("min_cost", "min_time", "max_quality"):
            assert callable(_get_scorer(policy))

    def test_get_scorer_invalid_raises(self):
        """Unknown policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            _get_scorer("fastest_cheapest")

    def test_min_cost_selects_cheapest(self):
        """min_cost scorer picks the lowest-cost PlanCost."""
        scorer = _get_scorer("min_cost")
        cheap = _pc(cost=0.1, time=5.0)
        expensive = _pc(cost=10.0, time=0.1)
        assert scorer(cheap) < scorer(expensive)

    def test_min_time_selects_fastest(self):
        """min_time scorer picks the lowest-time PlanCost."""
        scorer = _get_scorer("min_time")
        fast = _pc(cost=5.0, time=0.1)
        slow = _pc(cost=0.1, time=10.0)
        assert scorer(fast) < scorer(slow)

    def test_max_quality_selects_highest(self):
        """max_quality scorer (negated) picks highest quality."""
        scorer = _get_scorer("max_quality")
        high_q = _pc(cost=5.0, time=5.0, quality=0.9)
        low_q = _pc(cost=0.1, time=0.1, quality=0.1)
        assert scorer(high_q) < scorer(low_q)


# ── _expr_to_plan_node ────────────────────────────────────────────────


class TestExprToPlanNode:
    """Tests for the module-level ``_expr_to_plan_node`` helper."""

    def test_scan_creates_dataset_node(self):
        """A ScanOp expression → node_type='dataset'."""
        expr = _make_scan_expr(dataset_id="movies")
        node = _expr_to_plan_node(expr, "node-0", [])
        assert node.node_type == "dataset"
        assert node.dataset_id == "movies"
        assert node.operator_type is None
        assert node.parent_ids == []

    def test_scan_params_include_tokens(self):
        """Dataset node's params include est_total_tokens."""
        expr = _make_scan_expr(dataset_id="movies", num_items=100)
        node = _expr_to_plan_node(expr, "node-0", [])
        assert "est_total_tokens" in node.params

    def test_non_scan_creates_operator_node(self):
        """A non-ScanOp expression → node_type='operator'."""
        # Use a bare PhysicalOperator with a mocked op_name to avoid operator-specific init issues.
        from carnot.operators.physical import PhysicalOperator

        phys_op = PhysicalOperator(logical_op_id="filter-001", logical_op_class_name="Filter")
        phys_op.task = "keep mammals"
        expr = PhysicalExpression(operator=phys_op, input_group_ids=[0], group_id=1)
        node = _expr_to_plan_node(expr, "node-1", ["node-0"])
        assert node.node_type == "operator"
        assert node.parent_ids == ["node-0"]


# ── _select_best_plan (end-to-end scan-only) ─────────────────────────


class TestSelectBestPlanScanOnly:
    """Integration test: run the full optimizer on a leaf dataset and verify plan reconstruction."""

    @pytest.fixture
    def leaf_dataset(self):
        items = [
            {"title": "Inception", "genre": "Sci-Fi"},
            {"title": "The Matrix", "genre": "Sci-Fi"},
        ]
        return Dataset(
            name="Movies",
            annotation="Two movies",
            items=[DataItem.from_dict(d) for d in items],
        )

    @pytest.fixture
    def optimizer(self):
        return _make_optimizer()

    def test_scan_only_plan_structure(self, optimizer, leaf_dataset):
        """A single-dataset plan produces a PhysicalPlan with one dataset node."""
        plan_dict = optimizer.optimize(leaf_dataset)

        # Should have exactly one node.
        nodes = plan_dict["nodes"]
        assert len(nodes) == 1

        node = nodes[0]
        assert node["node_type"] == "dataset"
        assert node["dataset_id"] == "Movies"  # matches dataset name (default id)

    def test_scan_only_plan_min_cost(self, optimizer, leaf_dataset):
        """min_cost policy on a single scan → same result (only one option)."""
        plan = optimizer.optimize(leaf_dataset, policy="min_cost")
        assert len(plan["nodes"]) == 1

    def test_scan_only_plan_max_quality(self, optimizer, leaf_dataset):
        """max_quality policy on a single scan → same result."""
        plan = optimizer.optimize(leaf_dataset, policy="max_quality")
        assert len(plan["nodes"]) == 1

    def test_invalid_policy_raises(self, optimizer, leaf_dataset):
        """An unknown policy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown policy"):
            optimizer.optimize(leaf_dataset, policy="invalid")
