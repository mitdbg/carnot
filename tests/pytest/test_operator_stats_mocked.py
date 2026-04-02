"""Tier 2 mocked tests for operator stats collection.

These tests validate that every operator's ``__call__`` returns a
well-formed ``OperatorStats`` alongside the output datasets.  They use
the same ``mock_litellm`` fixture as other Tier 2 tests so that no real
LLM calls are made.

Operators covered:

* SemFilterOperator
* SemMapOperator
* SemFlatMapOperator
* SemAggOperator
* SemJoinOperator
* SemGroupByOperator
* SemTopKOperator
* LimitOperator
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from helpers.mock_utils import msg_text

from carnot.core.models import LLMCallStats, OperatorStats
from carnot.data.dataset import Dataset
from carnot.operators.limit import LimitOperator
from carnot.operators.sem_agg import SemAggOperator
from carnot.operators.sem_filter import SemFilterOperator
from carnot.operators.sem_flat_map import SemFlatMapOperator
from carnot.operators.sem_groupby import SemGroupByOperator
from carnot.operators.sem_join import SemJoinOperator
from carnot.operators.sem_map import SemMapOperator
from carnot.operators.sem_topk import SemTopKOperator

# ── Shared data ─────────────────────────────────────────────────────────────

_ANIMALS = [
    {"animal": "giraffe"},
    {"animal": "anaconda"},
    {"animal": "salmon"},
    {"animal": "elephant"},
    {"animal": "tucan"},
]

_MAMMALS = {"giraffe", "elephant"}

_GROUPS = {
    "giraffe": "mammal",
    "anaconda": "reptile",
    "salmon": "fish",
    "elephant": "mammal",
    "tucan": "bird",
}


# ── Handlers ────────────────────────────────────────────────────────────────


def _filter_handler(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    text = msg_text(messages)
    is_mammal = any(m in text for m in _MAMMALS)
    answer = "TRUE" if is_mammal else "FALSE"
    return _make_completion_response(f"```text\n{answer}\n```")


def _map_handler(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    text = msg_text(messages)
    for animal, group in _GROUPS.items():
        if animal in text:
            blob = json.dumps({"animal_group": group})
            return _make_completion_response(f"```json\n{blob}\n```")
    return _make_completion_response('```json\n{"animal_group": "unknown"}\n```')


def _flatmap_handler(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response(
        '```json\n[{"fruit": "apple"}, {"fruit": "banana"}]\n```'
    )


def _agg_handler(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    blob = json.dumps({"largest": "elephant"})
    return _make_completion_response(f"```json\n{blob}\n```")


def _join_handler(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    last_msg = messages[-1]
    text = msg_text([last_msg])
    matches = {("cow", "moo"), ("dog", "woof")}
    for animal, sound in matches:
        if animal in text and sound in text:
            return _make_completion_response("```text\nTRUE\n```")
    return _make_completion_response("```text\nFALSE\n```")


def _groupby_handler(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    text = msg_text(messages)
    for animal, group in _GROUPS.items():
        if animal in text:
            blob = json.dumps({"animal_group": group})
            return _make_completion_response(f"```json\n{blob}\n```")
    return _make_completion_response('```json\n{"animal_group": "unknown"}\n```')


# ── Fake index for SemTopK ──────────────────────────────────────────────────


class _FakeIndex:
    def __init__(self, name, items, model, api_key, index=None):
        self.name = name
        self._items = list(items) if items else []
        self._llm_call_stats: list = []

    def search(self, query: str, k: int) -> list[dict]:
        return self._items[:k]


# ── Helper assertions ───────────────────────────────────────────────────────


def _assert_valid_stats(
    stats: OperatorStats,
    *,
    operator_name: str,
    items_in: int,
    min_llm_calls: int = 0,
    max_llm_calls: int | None = None,
) -> None:
    """Assert invariants that every OperatorStats must satisfy.

    Requires:
        - *stats* is an instance of :class:`OperatorStats`.
        - *operator_name* matches the expected operator name.
        - *items_in* matches the expected input item count.
        - *min_llm_calls* and *max_llm_calls* bound ``total_llm_calls``.

    Returns:
        None (assertion-based).

    Raises:
        AssertionError if any invariant is violated.
    """
    assert isinstance(stats, OperatorStats)
    assert stats.operator_name == operator_name
    assert stats.items_in == items_in
    assert stats.items_out >= 0
    assert stats.wall_clock_secs >= 0.0
    assert len(stats.llm_calls) >= min_llm_calls

    if max_llm_calls is not None:
        assert len(stats.llm_calls) <= max_llm_calls

    # Every call stat must be well-formed
    for call in stats.llm_calls:
        assert isinstance(call, LLMCallStats)
        assert call.model_id != ""
        assert call.duration_secs >= 0.0

    # Derived properties must be consistent
    assert stats.total_cost_usd >= 0.0
    assert stats.total_llm_duration_secs >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSemFilterStats:
    """Verify that SemFilterOperator populates OperatorStats correctly."""

    def test_stats_populated(self, mock_litellm, mock_llm_config):
        """Stats include one LLM call per item, correct items_in/items_out."""
        mock_litellm.set_completion_handler(_filter_handler)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="is mammal",
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemFilter",
            items_in=len(_ANIMALS),
            min_llm_calls=len(_ANIMALS),
            max_llm_calls=len(_ANIMALS),
        )
        assert stats.items_out == 2  # giraffe + elephant

    def test_batched_stats_populated(self, mock_litellm, mock_llm_config):
        """Batched filtering tracks one LLM call per successful batch."""
        from fixtures.mocks import _make_completion_response

        def _batch_handler(model, messages, **kwargs):
            payload = msg_text(messages)
            if "Inputs:\n" in payload:
                return _make_completion_response('```json\n{"matching_indices": [0]}\n```')
            return _make_completion_response("```text\nFALSE\n```")

        mock_litellm.set_completion_handler(_batch_handler)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="is first item in each batch",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
            batch_size=2,
        )
        _, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemFilter",
            items_in=len(_ANIMALS),
            min_llm_calls=3,
            max_llm_calls=3,
        )
        assert stats.items_out == 2

    def test_empty_dataset_zero_calls(self, mock_litellm, mock_llm_config):
        """An empty dataset produces zero LLM calls and zero items out."""
        mock_litellm.set_completion_handler(_filter_handler)

        ds = Dataset(name="empty", annotation="test", items=[])
        op = SemFilterOperator(
            task="pass",
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("empty", {"empty": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemFilter",
            items_in=0,
            min_llm_calls=0,
            max_llm_calls=0,
        )
        assert stats.items_out == 0
        assert stats.total_cost_usd == 0.0


class TestSemMapStats:
    """Verify that SemMapOperator populates OperatorStats correctly."""

    def test_stats_populated(self, mock_litellm, mock_llm_config):
        """Stats include one LLM call per item; items_out equals items_in."""
        mock_litellm.set_completion_handler(_map_handler)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemMapOperator(
            task="Classify",
            output_fields=[{"name": "animal_group", "type": str, "description": "group"}],
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemMap",
            items_in=len(_ANIMALS),
            min_llm_calls=len(_ANIMALS),
            max_llm_calls=len(_ANIMALS),
        )
        assert stats.items_out == len(_ANIMALS)


class TestSemFlatMapStats:
    """Verify that SemFlatMapOperator populates OperatorStats correctly."""

    def test_stats_populated(self, mock_litellm, mock_llm_config):
        """One LLM call per input item; items_out reflects flattened output."""
        mock_litellm.set_completion_handler(_flatmap_handler)

        ds = Dataset(name="docs", annotation="test", items=[{"text": "a"}, {"text": "b"}])
        op = SemFlatMapOperator(
            task="Extract",
            output_fields=[{"name": "fruit", "type": str, "description": "fruit"}],
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("docs", {"docs": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemFlatMap",
            items_in=2,
            min_llm_calls=2,
            max_llm_calls=2,
        )
        # 2 items × 2 fruits each = 4
        assert stats.items_out == 4


class TestSemAggStats:
    """Verify that SemAggOperator populates OperatorStats correctly."""

    def test_stats_single_llm_call(self, mock_litellm, mock_llm_config):
        """Aggregation makes exactly one LLM call; items_out is 1."""
        mock_litellm.set_completion_handler(_agg_handler)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemAggOperator(
            task="Find largest",
            agg_fields=[{"name": "largest", "type": str, "description": "largest"}],
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemAgg",
            items_in=len(_ANIMALS),
            min_llm_calls=1,
            max_llm_calls=1,
        )
        assert stats.items_out == 1


class TestSemJoinStats:
    """Verify that SemJoinOperator populates OperatorStats correctly."""

    def test_stats_cross_product(self, mock_litellm, mock_llm_config):
        """LLM calls equal left × right; items_in equals the same product."""
        mock_litellm.set_completion_handler(_join_handler)

        left = [{"animal": "cow"}, {"animal": "dog"}]
        right = [{"sound": "moo"}, {"sound": "woof"}]

        left_ds = Dataset(name="left", annotation="test", items=list(left))
        right_ds = Dataset(name="right", annotation="test", items=list(right))
        op = SemJoinOperator(
            task="animal makes sound",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        _, stats = op("left", "right", {"left": left_ds, "right": right_ds})

        _assert_valid_stats(
            stats,
            operator_name="SemJoin",
            items_in=len(left) * len(right),
            min_llm_calls=len(left) * len(right),
            max_llm_calls=len(left) * len(right),
        )
        # cow-moo and dog-woof match
        assert stats.items_out == 2


class TestSemGroupByStats:
    """Verify that SemGroupByOperator populates OperatorStats correctly."""

    def test_stats_group_phase_calls(self, mock_litellm, mock_llm_config):
        """With built-in count agg, LLM calls equal item count (group phase only)."""
        mock_litellm.set_completion_handler(_groupby_handler)

        ds = Dataset(name="animals", annotation="test", items=[dict(a) for a in _ANIMALS])
        op = SemGroupByOperator(
            task="Count per group",
            group_by_fields=[{"name": "animal_group", "type": str, "description": "class"}],
            agg_fields=[{"name": "count", "type": int, "description": "count", "func": "count"}],
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemGroupBy",
            items_in=len(_ANIMALS),
            min_llm_calls=len(_ANIMALS),  # group phase: 1 per item
        )
        # 4 groups: mammal, reptile, fish, bird
        assert stats.items_out == 4


class TestSemTopKStats:
    """Verify that SemTopKOperator populates OperatorStats correctly."""

    def test_stats_from_fake_index(self, mock_llm_config):
        """Stats are returned even when no LLM calls are made (index-only)."""
        with patch.object(SemTopKOperator, "__init__", lambda self, **kw: None):
            op = SemTopKOperator.__new__(SemTopKOperator)
            op.task = "find mammals"
            op.dataset_id = "out"
            op.k = 2
            op.model_id = "mock-embedding"
            op.api_key = "fake"
            op.index_name = "flat"
            op.catalog = None
            op.index_cls = _FakeIndex

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        _, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="SemTopK",
            items_in=len(_ANIMALS),
            min_llm_calls=0,
        )
        assert stats.items_out == 2


class TestLimitStats:
    """Verify that LimitOperator populates OperatorStats correctly."""

    def test_stats_no_llm_calls(self, mock_llm_config):
        """Limit is a non-LLM operator; stats have zero LLM calls."""
        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = LimitOperator(n=3, dataset_id="out")
        _, stats = op("animals", {"animals": ds})

        _assert_valid_stats(
            stats,
            operator_name="Limit",
            items_in=len(_ANIMALS),
            min_llm_calls=0,
            max_llm_calls=0,
        )
        assert stats.items_out == 3
        assert stats.total_cost_usd == 0.0

    def test_limit_greater_than_items(self, mock_llm_config):
        """When n > len(items), items_out equals the actual item count."""
        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = LimitOperator(n=100, dataset_id="out")
        _, stats = op("animals", {"animals": ds})

        assert stats.items_out == len(_ANIMALS)


class TestStatsHaveCostData:
    """Cross-operator: verify that mock costs flow through to stats."""

    def test_filter_cost_positive(self, mock_litellm, mock_llm_config):
        """Mock fixture sets response_cost=0.001; aggregated cost > 0."""
        mock_litellm.set_completion_handler(_filter_handler)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="is mammal",
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("animals", {"animals": ds})

        # 5 calls × $0.001 each = $0.005
        assert stats.total_cost_usd == pytest.approx(0.005, abs=1e-6)
        assert stats.total_llm_calls == 5
        assert stats.total_embedding_calls == 0
        assert stats.total_llm_duration_secs > 0.0

    def test_filter_tokens_tracked(self, mock_litellm, mock_llm_config):
        """Mock fixture sets prompt_tokens=10, completion_tokens=5 per call."""
        mock_litellm.set_completion_handler(_filter_handler)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="is mammal",
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        _, stats = op("animals", {"animals": ds})

        # 5 calls × 10 prompt tokens = 50 input tokens
        assert stats.total_input_tokens == 50
        # 5 calls × 5 completion tokens = 25 output tokens
        assert stats.total_output_tokens == 25
