"""Tests for CostModel per-operator estimation and ModelPricingLookup.

**Tier 1** (pure unit) tests — no network calls or LLM mocks needed.
Pricing is either faked or uses litellm.model_cost directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from carnot.core.models import PlanCost
from carnot.operators.scan import ScanOp
from carnot.optimizer.cost_model import (
    DEFAULT_AGG_OUTPUT_TOKENS,
    DEFAULT_EMBEDDING_TIME_PER_RECORD,
    DEFAULT_FILTER_OUTPUT_TOKENS,
    DEFAULT_FILTER_SELECTIVITY,
    DEFAULT_FLATMAP_FAN_OUT,
    DEFAULT_FLATMAP_OUTPUT_TOKENS,
    DEFAULT_GROUPBY_NUM_GROUPS,
    DEFAULT_GROUPBY_OUTPUT_TOKENS,
    DEFAULT_JOIN_OUTPUT_TOKENS,
    DEFAULT_JOIN_SELECTIVITY,
    DEFAULT_MAP_OUTPUT_TOKENS,
    DEFAULT_TIME_PER_RECORD,
    CostModel,
)
from carnot.optimizer.pricing import ModelPricing, ModelPricingLookup

# ── Fixtures ──────────────────────────────────────────────────────────

TEST_PRICING = ModelPricing(
    input_cost_per_token=1e-06,
    output_cost_per_token=2e-06,
    max_input_tokens=128_000,
    max_output_tokens=16_384,
)

EMBEDDING_PRICING = ModelPricing(
    input_cost_per_token=5e-07,
    output_cost_per_token=0.0,
    max_input_tokens=8_191,
    max_output_tokens=0,
)


@pytest.fixture
def pricing_lookup() -> ModelPricingLookup:
    """A pricing lookup that returns TEST_PRICING for any model."""
    lookup = ModelPricingLookup(default_pricing=TEST_PRICING)
    return lookup


@pytest.fixture
def cost_model(pricing_lookup) -> CostModel:
    """CostModel wired to the test pricing lookup."""
    return CostModel(pricing=pricing_lookup)


def _scan_plan_cost(num_items: int = 100, tokens_per_item: float = 50.0) -> PlanCost:
    """Build a PlanCost as if produced by _estimate_scan."""
    n = float(num_items)
    total = n * tokens_per_item
    return PlanCost(
        cost=0.0,
        time=0.0,
        total_input_tokens=total,
        total_scanned_input_tokens=total,
        input_cardinality=n,
        output_cardinality=n,
        selectivity=1.0,
        avg_tokens_per_record=tokens_per_item,
        cost_per_record=0.0,
        time_per_record=0.0,
    )


# ══════════════════════════════════════════════════════════════════════
#  ModelPricingLookup
# ══════════════════════════════════════════════════════════════════════


class TestModelPricingLookup:
    """Tests for ``ModelPricingLookup``."""

    def test_known_model_returns_pricing(self):
        """A well-known model like 'gpt-4o' resolves from litellm.model_cost."""
        lookup = ModelPricingLookup()
        pricing = lookup.get("gpt-4o")
        assert pricing.input_cost_per_token > 0
        assert pricing.output_cost_per_token > 0
        assert pricing.max_input_tokens > 0

    def test_unknown_model_with_default_returns_default(self):
        """Unknown model falls back to default pricing when provided."""
        default = ModelPricing(
            input_cost_per_token=1e-06,
            output_cost_per_token=2e-06,
            max_input_tokens=100,
            max_output_tokens=50,
        )
        lookup = ModelPricingLookup(default_pricing=default)
        pricing = lookup.get("nonexistent-model-xyz")
        assert pricing is default

    def test_unknown_model_without_default_raises(self):
        """Unknown model without default raises ValueError."""
        lookup = ModelPricingLookup()
        with pytest.raises(ValueError, match="not found"):
            lookup.get("nonexistent-model-xyz")

    def test_caching(self):
        """Second call for same model returns cached result."""
        lookup = ModelPricingLookup()
        a = lookup.get("gpt-4o")
        b = lookup.get("gpt-4o")
        assert a is b

    def test_untrusted_provider_falls_back(self):
        """A model whose provider is not in _TRUSTED_PROVIDERS falls back to default."""
        default = ModelPricing(
            input_cost_per_token=0.0,
            output_cost_per_token=0.0,
            max_input_tokens=0,
            max_output_tokens=0,
        )
        fake_model_cost = {
            "fake-model": {
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 1.0,
                "litellm_provider": "vertex_ai",
                "max_input_tokens": 100,
                "max_output_tokens": 100,
            }
        }
        with patch("carnot.optimizer.pricing.litellm") as mock_litellm:
            mock_litellm.model_cost = fake_model_cost
            lookup = ModelPricingLookup(default_pricing=default)
            pricing = lookup.get("fake-model")
            assert pricing is default


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_scan
# ══════════════════════════════════════════════════════════════════════


class TestEstimateScan:
    """Tests for ``CostModel._estimate_scan``."""

    def test_scan_cost_is_zero(self, cost_model):
        """Scan has zero dollar cost and zero time."""
        op = ScanOp(dataset_id="ds", num_items=10, est_tokens_per_item=50.0)
        pc = cost_model(op)
        assert pc.cost == 0.0
        assert pc.time == 0.0

    def test_scan_token_fields(self, cost_model):
        """Token fields are seeded from num_items × est_tokens_per_item."""
        op = ScanOp(dataset_id="ds", num_items=20, est_tokens_per_item=10.0)
        pc = cost_model(op)
        assert pc.total_input_tokens == 200.0
        assert pc.total_scanned_input_tokens == 200.0

    def test_scan_cardinality_fields(self, cost_model):
        """Cardinality fields match num_items; selectivity is 1.0."""
        op = ScanOp(dataset_id="ds", num_items=50, est_tokens_per_item=5.0)
        pc = cost_model(op)
        assert pc.input_cardinality == 50.0
        assert pc.output_cardinality == 50.0
        assert pc.selectivity == 1.0
        assert pc.avg_tokens_per_record == 5.0

    def test_scan_quality_is_one(self, cost_model):
        """A scan reads everything so quality == 1.0."""
        op = ScanOp(dataset_id="ds", num_items=10, est_tokens_per_item=10.0)
        pc = cost_model(op)
        assert pc.quality == 1.0

    def test_scan_zero_items(self, cost_model):
        """ScanOp with zero items returns quality=1.0 (safe division)."""
        op = ScanOp(dataset_id="empty", num_items=0, est_tokens_per_item=0.0)
        pc = cost_model(op)
        assert pc.quality == 1.0
        assert pc.total_input_tokens == 0.0
        assert pc.output_cardinality == 0.0


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_filter
# ══════════════════════════════════════════════════════════════════════


class TestEstimateFilter:
    """Tests for ``CostModel._estimate_filter``."""

    def _make_filter_op(self) -> MagicMock:
        from carnot.operators.sem_filter import SemFilterOperator

        op = MagicMock(spec=SemFilterOperator)
        op.model_id = "test-model"
        return op

    def test_filter_selectivity(self, cost_model):
        """Filter selectivity is DEFAULT_FILTER_SELECTIVITY."""
        op = self._make_filter_op()
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_filter(op, ipc)
        assert pc.selectivity == DEFAULT_FILTER_SELECTIVITY
        assert pc.output_cardinality == 100.0 * DEFAULT_FILTER_SELECTIVITY

    def test_filter_preserves_avg_tokens(self, cost_model):
        """Filter does not add fields, so avg_tokens_per_record == T_in."""
        op = self._make_filter_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=30.0)
        pc = cost_model._estimate_filter(op, ipc)
        assert pc.avg_tokens_per_record == 30.0

    def test_filter_cost_formula(self, cost_model):
        """cost = cost_per_record × N + upstream_cost."""
        op = self._make_filter_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_filter(op, ipc)

        expected_cpr = 50.0 * TEST_PRICING.input_cost_per_token + DEFAULT_FILTER_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        assert pc.cost_per_record == pytest.approx(expected_cpr)
        assert pc.cost == pytest.approx(expected_cpr * 10.0 + ipc.cost)

    def test_filter_time_formula(self, cost_model):
        """time = time_per_record × N + upstream_time."""
        op = self._make_filter_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_filter(op, ipc)
        assert pc.time == pytest.approx(DEFAULT_TIME_PER_RECORD * 10.0 + ipc.time)

    def test_filter_token_cumulation(self, cost_model):
        """total_input_tokens = N × T_in + upstream.total_input_tokens."""
        op = self._make_filter_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_filter(op, ipc)
        assert pc.total_input_tokens == pytest.approx(10.0 * 50.0 + ipc.total_input_tokens)
        assert pc.total_scanned_input_tokens == pytest.approx(10.0 * 50.0 + ipc.total_scanned_input_tokens)

    def test_filter_quality_stays_one(self, cost_model):
        """Naive filter scans everything → quality remains 1.0."""
        op = self._make_filter_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_filter(op, ipc)
        assert pc.quality == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_map
# ══════════════════════════════════════════════════════════════════════


class TestEstimateMap:
    """Tests for ``CostModel._estimate_map``."""

    def _make_map_op(self) -> MagicMock:
        from carnot.operators.sem_map import SemMapOperator

        op = MagicMock(spec=SemMapOperator)
        op.model_id = "test-model"
        return op

    def test_map_selectivity_is_one(self, cost_model):
        """Map preserves cardinality → selectivity == 1.0."""
        op = self._make_map_op()
        ipc = _scan_plan_cost(num_items=20, tokens_per_item=40.0)
        pc = cost_model._estimate_map(op, ipc)
        assert pc.selectivity == 1.0
        assert pc.output_cardinality == 20.0

    def test_map_enriches_avg_tokens(self, cost_model):
        """Map adds output tokens to per-record size."""
        op = self._make_map_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_map(op, ipc)
        assert pc.avg_tokens_per_record == 50.0 + DEFAULT_MAP_OUTPUT_TOKENS

    def test_map_cost_formula(self, cost_model):
        """cost = cost_per_record × N + upstream_cost."""
        op = self._make_map_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_map(op, ipc)

        expected_cpr = 50.0 * TEST_PRICING.input_cost_per_token + DEFAULT_MAP_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        assert pc.cost_per_record == pytest.approx(expected_cpr)
        assert pc.cost == pytest.approx(expected_cpr * 10.0 + ipc.cost)

    def test_map_quality_stays_one(self, cost_model):
        """Naive map scans everything → quality remains 1.0."""
        op = self._make_map_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_map(op, ipc)
        assert pc.quality == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_flatmap
# ══════════════════════════════════════════════════════════════════════


class TestEstimateFlatMap:
    """Tests for ``CostModel._estimate_flatmap``."""

    def _make_flatmap_op(self) -> MagicMock:
        from carnot.operators.sem_flat_map import SemFlatMapOperator

        op = MagicMock(spec=SemFlatMapOperator)
        op.model_id = "test-model"
        return op

    def test_flatmap_fan_out(self, cost_model):
        """FlatMap output cardinality is N × fan_out."""
        op = self._make_flatmap_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_flatmap(op, ipc)
        assert pc.selectivity == DEFAULT_FLATMAP_FAN_OUT
        assert pc.output_cardinality == 10.0 * DEFAULT_FLATMAP_FAN_OUT

    def test_flatmap_avg_tokens(self, cost_model):
        """FlatMap output records use DEFAULT_FLATMAP_OUTPUT_TOKENS."""
        op = self._make_flatmap_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_flatmap(op, ipc)
        assert pc.avg_tokens_per_record == DEFAULT_FLATMAP_OUTPUT_TOKENS

    def test_flatmap_cost_formula(self, cost_model):
        """cost = cost_per_record × N + upstream_cost (N calls, not N×fan_out)."""
        op = self._make_flatmap_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_flatmap(op, ipc)

        expected_cpr = (
            50.0 * TEST_PRICING.input_cost_per_token + DEFAULT_FLATMAP_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        )
        assert pc.cost == pytest.approx(expected_cpr * 10.0 + ipc.cost)


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_join
# ══════════════════════════════════════════════════════════════════════


class TestEstimateJoin:
    """Tests for ``CostModel._estimate_join``."""

    def _make_join_op(self) -> MagicMock:
        from carnot.operators.sem_join import SemJoinOperator

        op = MagicMock(spec=SemJoinOperator)
        op.model_id = "test-model"
        return op

    def test_join_cross_product(self, cost_model):
        """Input cardinality is N_L × N_R."""
        op = self._make_join_op()
        left = _scan_plan_cost(num_items=10, tokens_per_item=30.0)
        right = _scan_plan_cost(num_items=5, tokens_per_item=20.0)
        pc = cost_model._estimate_join(op, left, right)
        assert pc.input_cardinality == 50.0
        assert pc.output_cardinality == 50.0 * DEFAULT_JOIN_SELECTIVITY

    def test_join_avg_tokens(self, cost_model):
        """Merged record has T_L + T_R tokens."""
        op = self._make_join_op()
        left = _scan_plan_cost(num_items=10, tokens_per_item=30.0)
        right = _scan_plan_cost(num_items=5, tokens_per_item=20.0)
        pc = cost_model._estimate_join(op, left, right)
        assert pc.avg_tokens_per_record == 50.0

    def test_join_cost_formula(self, cost_model):
        """cost = cpr × cross + left.cost + right.cost."""
        op = self._make_join_op()
        left = _scan_plan_cost(num_items=10, tokens_per_item=30.0)
        right = _scan_plan_cost(num_items=5, tokens_per_item=20.0)
        pc = cost_model._estimate_join(op, left, right)

        t_pair = 50.0
        expected_cpr = t_pair * TEST_PRICING.input_cost_per_token + DEFAULT_JOIN_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        assert pc.cost == pytest.approx(expected_cpr * 50.0 + left.cost + right.cost)

    def test_join_token_cumulation(self, cost_model):
        """total_input_tokens sums cross-product contribution + both upstreams."""
        op = self._make_join_op()
        left = _scan_plan_cost(num_items=10, tokens_per_item=30.0)
        right = _scan_plan_cost(num_items=5, tokens_per_item=20.0)
        pc = cost_model._estimate_join(op, left, right)

        expected = 50.0 * 50.0 + left.total_input_tokens + right.total_input_tokens
        assert pc.total_input_tokens == pytest.approx(expected)

    def test_join_quality_stays_one(self, cost_model):
        """Naive join scans everything → quality remains 1.0."""
        op = self._make_join_op()
        left = _scan_plan_cost(num_items=2, tokens_per_item=10.0)
        right = _scan_plan_cost(num_items=3, tokens_per_item=10.0)
        pc = cost_model._estimate_join(op, left, right)
        assert pc.quality == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_agg
# ══════════════════════════════════════════════════════════════════════


class TestEstimateAgg:
    """Tests for ``CostModel._estimate_agg``."""

    def _make_agg_op(self) -> MagicMock:
        from carnot.operators.sem_agg import SemAggOperator

        op = MagicMock(spec=SemAggOperator)
        op.model_id = "test-model"
        return op

    def test_agg_output_cardinality_is_one(self, cost_model):
        """Aggregation produces a single output row."""
        op = self._make_agg_op()
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_agg(op, ipc)
        assert pc.output_cardinality == 1.0

    def test_agg_selectivity(self, cost_model):
        """Selectivity is 1/N."""
        op = self._make_agg_op()
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_agg(op, ipc)
        assert pc.selectivity == pytest.approx(0.01)

    def test_agg_single_call_cost(self, cost_model):
        """Cost reflects a single LLM call reading all N records."""
        op = self._make_agg_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_agg(op, ipc)

        # Single call: input = N × T_in, output = DEFAULT_AGG_OUTPUT_TOKENS
        expected_cpr = (
            10.0 * 50.0 * TEST_PRICING.input_cost_per_token
            + DEFAULT_AGG_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        )
        assert pc.cost_per_record == pytest.approx(expected_cpr)
        # Total cost = single-call cost + upstream
        assert pc.cost == pytest.approx(expected_cpr + ipc.cost)

    def test_agg_avg_tokens_is_output(self, cost_model):
        """Aggregation output record uses DEFAULT_AGG_OUTPUT_TOKENS."""
        op = self._make_agg_op()
        ipc = _scan_plan_cost(num_items=10, tokens_per_item=50.0)
        pc = cost_model._estimate_agg(op, ipc)
        assert pc.avg_tokens_per_record == DEFAULT_AGG_OUTPUT_TOKENS


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_groupby
# ══════════════════════════════════════════════════════════════════════


class TestEstimateGroupBy:
    """Tests for ``CostModel._estimate_groupby``."""

    def _make_groupby_op(self) -> MagicMock:
        from carnot.operators.sem_groupby import SemGroupByOperator

        op = MagicMock(spec=SemGroupByOperator)
        op.model_id = "test-model"
        return op

    def test_groupby_output_cardinality(self, cost_model):
        """Output cardinality is DEFAULT_GROUPBY_NUM_GROUPS."""
        op = self._make_groupby_op()
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_groupby(op, ipc)
        assert pc.output_cardinality == float(DEFAULT_GROUPBY_NUM_GROUPS)

    def test_groupby_selectivity(self, cost_model):
        """Selectivity is G / N."""
        op = self._make_groupby_op()
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_groupby(op, ipc)
        assert pc.selectivity == pytest.approx(DEFAULT_GROUPBY_NUM_GROUPS / 100.0)

    def test_groupby_time_formula(self, cost_model):
        """time = (N + G) × DEFAULT_TIME_PER_RECORD + upstream_time."""
        op = self._make_groupby_op()
        n = 20
        ipc = _scan_plan_cost(num_items=n, tokens_per_item=50.0)
        pc = cost_model._estimate_groupby(op, ipc)
        g = float(DEFAULT_GROUPBY_NUM_GROUPS)
        assert pc.time == pytest.approx((n + g) * DEFAULT_TIME_PER_RECORD + ipc.time)

    def test_groupby_two_phase_cost(self, cost_model):
        """Cost has grouping phase (N calls) + aggregation phase (G calls)."""
        op = self._make_groupby_op()
        n = 20
        t_in = 50.0
        ipc = _scan_plan_cost(num_items=n, tokens_per_item=t_in)
        pc = cost_model._estimate_groupby(op, ipc)
        g = float(DEFAULT_GROUPBY_NUM_GROUPS)
        records_per_group = n / g

        grouping_cost = n * (
            t_in * TEST_PRICING.input_cost_per_token + DEFAULT_GROUPBY_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        )
        agg_cost = g * (
            records_per_group * t_in * TEST_PRICING.input_cost_per_token
            + DEFAULT_GROUPBY_OUTPUT_TOKENS * TEST_PRICING.output_cost_per_token
        )
        assert pc.cost == pytest.approx(grouping_cost + agg_cost + ipc.cost)


# ══════════════════════════════════════════════════════════════════════
#  CostModel._estimate_topk
# ══════════════════════════════════════════════════════════════════════


class TestEstimateTopK:
    """Tests for ``CostModel._estimate_topk``."""

    def _make_topk_op(self, k: int = 5) -> MagicMock:
        from carnot.operators.sem_topk import SemTopKOperator

        op = MagicMock(spec=SemTopKOperator)
        op.model_id = "openai/text-embedding-3-small"
        op.k = k
        return op

    def test_topk_output_cardinality(self, cost_model):
        """Output cardinality equals k."""
        op = self._make_topk_op(k=5)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.output_cardinality == 5.0

    def test_topk_selectivity(self, cost_model):
        """Selectivity is k / N."""
        op = self._make_topk_op(k=10)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.selectivity == pytest.approx(0.1)

    def test_topk_preserves_avg_tokens(self, cost_model):
        """TopK does not modify records, so avg_tokens_per_record == T_in."""
        op = self._make_topk_op(k=5)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.avg_tokens_per_record == 50.0

    def test_topk_embedding_time(self, cost_model):
        """Time uses DEFAULT_EMBEDDING_TIME_PER_RECORD, not completion time."""
        op = self._make_topk_op(k=5)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.time_per_record == DEFAULT_EMBEDDING_TIME_PER_RECORD
        assert pc.time == pytest.approx(DEFAULT_EMBEDDING_TIME_PER_RECORD * 100.0 + ipc.time)

    def test_topk_scanned_tokens_use_k(self, cost_model):
        """total_scanned_input_tokens = k × T_in + upstream (not N × T_in)."""
        op = self._make_topk_op(k=5)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.total_scanned_input_tokens == pytest.approx(
            5.0 * 50.0 + ipc.total_scanned_input_tokens
        )

    def test_topk_total_tokens_use_n(self, cost_model):
        """total_input_tokens = N × T_in + upstream (full dataset)."""
        op = self._make_topk_op(k=5)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.total_input_tokens == pytest.approx(
            100.0 * 50.0 + ipc.total_input_tokens
        )

    def test_topk_quality_drops(self, cost_model):
        """TopK with k < N reduces quality below 1.0."""
        op = self._make_topk_op(k=10)
        ipc = _scan_plan_cost(num_items=100, tokens_per_item=50.0)
        pc = cost_model._estimate_topk(op, ipc)
        assert pc.quality < 1.0


# ══════════════════════════════════════════════════════════════════════
#  Pipeline integration (Scan → Filter → Map)
# ══════════════════════════════════════════════════════════════════════


class TestPipelineCostEstimation:
    """Integration tests: cost propagation through multi-operator pipelines."""

    def test_scan_filter_map_pipeline(self, cost_model):
        """Scan → Filter → Map: cardinality and tokens propagate correctly."""
        from carnot.operators.sem_filter import SemFilterOperator
        from carnot.operators.sem_map import SemMapOperator

        # Scan
        scan_op = ScanOp(dataset_id="ds", num_items=100, est_tokens_per_item=50.0)
        scan_pc = cost_model(scan_op)

        # Filter
        filter_op = MagicMock(spec=SemFilterOperator)
        filter_op.model_id = "test-model"
        filter_pc = cost_model(filter_op, scan_pc)

        # Map (receives filter's output)
        map_op = MagicMock(spec=SemMapOperator)
        map_op.model_id = "test-model"
        map_pc = cost_model(map_op, filter_pc)

        # Cardinality chain: 100 → 50 (filter) → 50 (map preserves)
        assert scan_pc.output_cardinality == 100.0
        assert filter_pc.output_cardinality == 50.0
        assert map_pc.output_cardinality == 50.0

        # Map enriches records
        assert map_pc.avg_tokens_per_record == 50.0 + DEFAULT_MAP_OUTPUT_TOKENS

        # Costs are cumulative
        assert map_pc.cost > filter_pc.cost > scan_pc.cost
        assert map_pc.time > filter_pc.time > scan_pc.time

        # Quality remains 1.0 (all naive operators)
        assert map_pc.quality == pytest.approx(1.0)

    def test_scan_topk_reduces_quality(self, cost_model):
        """Scan → TopK: quality drops because k < N."""
        from carnot.operators.sem_topk import SemTopKOperator

        scan_op = ScanOp(dataset_id="ds", num_items=100, est_tokens_per_item=50.0)
        scan_pc = cost_model(scan_op)

        topk_op = MagicMock(spec=SemTopKOperator)
        topk_op.model_id = "openai/text-embedding-3-small"
        topk_op.k = 10
        topk_pc = cost_model(topk_op, scan_pc)

        assert topk_pc.quality < scan_pc.quality

    def test_unrecognized_operator_raises(self, cost_model):
        """An unrecognized operator type raises NotImplementedError."""
        from carnot.operators.physical import PhysicalOperator

        dummy = PhysicalOperator()
        ipc = _scan_plan_cost()
        with pytest.raises(NotImplementedError, match="does not handle"):
            cost_model(dummy, ipc)
