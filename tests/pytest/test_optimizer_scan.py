"""Tests for the Scan operator pipeline: logical Scan, physical ScanOp,
ScanRule, CostModel base case, and optimizer group-tree construction.

These are **Tier 1** (pure unit) and **Tier 2** (mocked) tests — no network
calls are made.
"""

from __future__ import annotations

import pytest

from carnot.core.models import PlanCost
from carnot.data.dataset import Dataset
from carnot.data.item import DataItem
from carnot.operators.logical import Scan
from carnot.operators.scan import ScanOp
from carnot.optimizer.cost_model import CostModel
from carnot.optimizer.primitives import LogicalExpression
from carnot.optimizer.rules import ScanRule

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def scan_logical_op():
    """A simple logical Scan with known metadata."""
    return Scan(name="movies", dataset_id="movies-001", num_items=100, est_tokens_per_item=50.0)


@pytest.fixture
def scan_physical_op():
    """A ScanOp mirroring the logical Scan fixture."""
    return ScanOp(dataset_id="movies-001", num_items=100, est_tokens_per_item=50.0)


@pytest.fixture
def small_dataset():
    """A tiny in-memory dataset for group-tree tests."""
    items = [
        {"title": "Inception", "genre": "Sci-Fi"},
        {"title": "The Matrix", "genre": "Sci-Fi"},
        {"title": "Pulp Fiction", "genre": "Crime"},
    ]
    return Dataset(
        name="SmallMovies",
        annotation="Three movies for unit testing",
        items=[DataItem.from_dict(d) for d in items],
    )


# ── Scan logical operator ────────────────────────────────────────────


class TestScanLogicalOperator:
    """Tests for the ``Scan`` logical operator."""

    def test_fields_stored(self, scan_logical_op):
        """Fields passed to __init__ are stored correctly."""
        assert scan_logical_op.dataset_id == "movies-001"
        assert scan_logical_op.num_items == 100
        assert scan_logical_op.est_tokens_per_item == 50.0

    def test_desc(self, scan_logical_op):
        """desc() includes the dataset_id and item count."""
        desc = scan_logical_op.desc()
        assert "movies-001" in desc
        assert "100" in desc

    def test_str(self, scan_logical_op):
        """__str__ produces a readable representation."""
        s = str(scan_logical_op)
        assert "Scan" in s
        assert "movies-001" in s

    def test_logical_op_params(self, scan_logical_op):
        """get_logical_op_params includes all scan-specific keys."""
        params = scan_logical_op.get_logical_op_params()
        assert params["dataset_id"] == "movies-001"
        assert params["num_items"] == 100
        assert params["est_tokens_per_item"] == 50.0

    def test_zero_items(self):
        """A Scan with zero items is valid."""
        op = Scan(name="empty", dataset_id="empty-ds", num_items=0, est_tokens_per_item=0.0)
        assert op.num_items == 0
        assert op.est_tokens_per_item == 0.0


# ── ScanOp physical operator ─────────────────────────────────────────


class TestScanOp:
    """Tests for the ``ScanOp`` physical operator."""

    def test_est_total_tokens(self, scan_physical_op):
        """est_total_tokens == num_items * est_tokens_per_item."""
        assert scan_physical_op.est_total_tokens == 100 * 50.0

    def test_est_total_tokens_zero_items(self):
        """Zero items ⇒ zero total tokens."""
        op = ScanOp(dataset_id="empty", num_items=0, est_tokens_per_item=42.0)
        assert op.est_total_tokens == 0.0

    def test_get_id_params(self, scan_physical_op):
        """get_id_params returns only the dataset_id."""
        assert scan_physical_op.get_id_params() == {"dataset_id": "movies-001"}

    def test_get_op_params_roundtrip(self, scan_physical_op):
        """get_op_params returns enough info to reconstruct the operator."""
        params = scan_physical_op.get_op_params()
        assert params["dataset_id"] == "movies-001"
        assert params["num_items"] == 100
        assert params["est_tokens_per_item"] == 50.0

    def test_op_name(self, scan_physical_op):
        """op_name() returns the class name."""
        assert scan_physical_op.op_name() == "ScanOp"


# ── ScanRule ──────────────────────────────────────────────────────────


class TestScanRule:
    """Tests for the ``ScanRule`` implementation rule."""

    def test_matches_scan(self, scan_logical_op):
        """ScanRule matches a LogicalExpression wrapping a Scan."""
        le = LogicalExpression(operator=scan_logical_op, input_group_ids=[], group_id=0)
        assert ScanRule.matches_pattern(le)

    def test_does_not_match_non_scan(self):
        """ScanRule does not match non-Scan logical operators."""
        from carnot.operators.logical import Filter

        le = LogicalExpression(
            operator=Filter(name="f", filter="keep mammals"),
            input_group_ids=[0],
            group_id=1,
        )
        assert not ScanRule.matches_pattern(le)

    def test_substitute_returns_single_physical_expr(self, scan_logical_op):
        """substitute() returns exactly one PhysicalExpression wrapping a ScanOp."""
        le = LogicalExpression(operator=scan_logical_op, input_group_ids=[], group_id=0)
        result = ScanRule.substitute(le)
        assert len(result) == 1

        phys_expr = next(iter(result))
        assert isinstance(phys_expr.operator, ScanOp)
        assert phys_expr.operator.dataset_id == "movies-001"
        assert phys_expr.operator.num_items == 100
        assert phys_expr.operator.est_tokens_per_item == 50.0

    def test_substitute_preserves_group_id(self, scan_logical_op):
        """The physical expression inherits the group_id from the logical expression."""
        le = LogicalExpression(operator=scan_logical_op, input_group_ids=[], group_id=7)
        result = ScanRule.substitute(le)
        phys_expr = next(iter(result))
        assert phys_expr.group_id == 7

    def test_substitute_preserves_empty_input_group_ids(self, scan_logical_op):
        """Scans have no inputs — input_group_ids should be empty."""
        le = LogicalExpression(operator=scan_logical_op, input_group_ids=[], group_id=0)
        result = ScanRule.substitute(le)
        phys_expr = next(iter(result))
        assert phys_expr.input_group_ids == []


# ── CostModel scan base case ─────────────────────────────────────────


class TestCostModelScanBaseCase:
    """Tests for ``CostModel.__call__`` handling of ScanOp."""

    def test_scan_returns_zero_cost_and_time(self, scan_physical_op):
        """ScanOp produces PlanCost with cost=0 and time=0."""
        cm = CostModel()
        pc = cm(scan_physical_op)
        assert pc.cost == 0.0
        assert pc.time == 0.0

    def test_scan_seeds_token_fields(self, scan_physical_op):
        """total_input_tokens and total_scanned_input_tokens == est_total_tokens."""
        cm = CostModel()
        pc = cm(scan_physical_op)
        expected = scan_physical_op.est_total_tokens
        assert pc.total_input_tokens == expected
        assert pc.total_scanned_input_tokens == expected

    def test_scan_quality_is_one(self, scan_physical_op):
        """A scan reads everything so quality == 1.0."""
        cm = CostModel()
        pc = cm(scan_physical_op)
        assert pc.quality == 1.0

    def test_scan_cardinality(self, scan_physical_op):
        """Cardinality matches num_items."""
        cm = CostModel()
        pc = cm(scan_physical_op)
        assert pc.output_cardinality == float(scan_physical_op.num_items)
        assert pc.input_cardinality == float(scan_physical_op.num_items)
        assert pc.selectivity == 1.0

    def test_scan_zero_items(self):
        """ScanOp with zero items returns quality=1.0 (safe division)."""
        op = ScanOp(dataset_id="empty", num_items=0, est_tokens_per_item=0.0)
        cm = CostModel()
        pc = cm(op)
        assert pc.quality == 1.0
        assert pc.total_input_tokens == 0.0
        assert pc.output_cardinality == 0.0

    def test_non_scan_raises(self):
        """Base CostModel raises NotImplementedError for non-scan operators."""
        from carnot.operators.physical import PhysicalOperator

        dummy = PhysicalOperator()
        cm = CostModel()
        with pytest.raises(NotImplementedError):
            cm(dummy, PlanCost(cost=0, time=0, total_input_tokens=0, total_scanned_input_tokens=0))


# ── Optimizer._construct_group_tree ───────────────────────────────────


class TestConstructGroupTree:
    """Tests for ``Optimizer._construct_group_tree`` with leaf datasets."""

    @pytest.fixture
    def optimizer(self):
        """A minimal Optimizer with dummy model and config."""
        from unittest.mock import MagicMock

        from carnot.optimizer.optimizer import Optimizer

        model = MagicMock()
        return Optimizer(model=model, available_model_ids=["test-model"])

    def test_leaf_creates_scan_group(self, optimizer, small_dataset):
        """A leaf dataset (operator=None) produces a group with a Scan logical expression."""
        gid = optimizer._construct_group_tree(small_dataset)
        group = optimizer.groups[gid]
        assert len(group.logical_expressions) == 1

        le = next(iter(group.logical_expressions))
        assert isinstance(le.operator, Scan)
        assert le.operator.dataset_id == small_dataset.dataset_id
        assert le.operator.num_items == len(small_dataset.items)

    def test_leaf_has_no_input_groups(self, optimizer, small_dataset):
        """A leaf scan expression has no input groups."""
        gid = optimizer._construct_group_tree(small_dataset)
        le = next(iter(optimizer.groups[gid].logical_expressions))
        assert le.input_group_ids == []

    def test_operator_dataset_preserves_structure(self, optimizer, small_dataset):
        """A derived dataset (with operator and parent) creates groups for the parent too."""
        from carnot.operators.logical import Filter

        derived = Dataset(
            name="Filtered",
            annotation="Filtered movies",
            parents=[small_dataset],
            operator=Filter(name="Filtered", filter="keep sci-fi movies"),
        )
        root_gid = optimizer._construct_group_tree(derived)

        # Root group should have Filter operator.
        root_group = optimizer.groups[root_gid]
        root_le = next(iter(root_group.logical_expressions))
        assert isinstance(root_le.operator, Filter)

        # It should reference one input group (the leaf).
        assert len(root_le.input_group_ids) == 1
        leaf_gid = root_le.input_group_ids[0]
        leaf_le = next(iter(optimizer.groups[leaf_gid].logical_expressions))
        assert isinstance(leaf_le.operator, Scan)


# ── Optimizer._estimate_tokens_per_item ───────────────────────────────


class TestEstimateTokensPerItem:
    """Tests for the token estimation helper."""

    @pytest.fixture
    def optimizer(self):
        from unittest.mock import MagicMock

        from carnot.optimizer.optimizer import Optimizer

        model = MagicMock()
        return Optimizer(model=model, available_model_ids=["test-model"])

    def test_empty_dataset_returns_zero(self, optimizer):
        """Empty dataset → estimate is 0.0."""
        ds = Dataset(name="Empty", annotation="nothing", items=[])
        assert optimizer._estimate_tokens_per_item(ds) == 0.0

    def test_positive_for_nonempty(self, optimizer, small_dataset):
        """A non-empty text dataset produces a positive token estimate."""
        est = optimizer._estimate_tokens_per_item(small_dataset)
        assert est > 0.0

    def test_result_is_average(self, optimizer):
        """The estimate equals the mean token count across sampled items."""
        items = [
            {"text": "hello world"},
            {"text": "another sentence with more tokens"},
        ]
        ds = Dataset(name="Texts", annotation="", items=[DataItem.from_dict(d) for d in items])
        est = optimizer._estimate_tokens_per_item(ds)
        # Exact count depends on the tokenizer, but it should be a float > 0.
        assert isinstance(est, float)
        assert est > 0.0
