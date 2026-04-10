"""Tier 2 mocked tests for :class:`SemAggOperator`.

These tests patch ``litellm.completion`` so that **no network calls** are
made.  They validate:

* JSON output parsing for aggregation,
* single-item output dataset,
* missing aggregation fields default to ``None``,
* dataset threading (input → aggregated output).
"""

from __future__ import annotations

import json

from carnot.data.dataset import Dataset
from carnot.operators.sem_agg import SemAggOperator

# ── Helpers ──────────────────────────────────────────────────────────────────

_ANIMALS = [
    {"animal": "giraffe"},
    {"animal": "anaconda"},
    {"animal": "salmon"},
    {"animal": "elephant"},
    {"animal": "tucan"},
]


def _agg_handler_largest(model, messages, **kwargs):
    """Always return elephant as the largest animal."""
    from fixtures.mocks import _make_completion_response

    blob = json.dumps({"largest_animal": "elephant"})
    return _make_completion_response(f"```json\n{blob}\n```")


def _agg_handler_multiple_fields(model, messages, **kwargs):
    """Return multiple aggregation fields."""
    from fixtures.mocks import _make_completion_response

    blob = json.dumps({"largest": "elephant", "smallest": "tucan"})
    return _make_completion_response(f"```json\n{blob}\n```")


def _agg_handler_partial(model, messages, **kwargs):
    """Return only one of the expected fields."""
    from fixtures.mocks import _make_completion_response

    blob = json.dumps({"largest": "elephant"})
    return _make_completion_response(f"```json\n{blob}\n```")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSemAggMocked:
    """Mocked-LLM tests for SemAggOperator."""

    def test_aggregates_to_single_item(self, mock_litellm, mock_llm_config):
        """Aggregation always produces a dataset with exactly one item."""
        mock_litellm.set_completion_handler(_agg_handler_largest)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        agg_fields = [{"name": "largest_animal", "type": str, "description": "largest"}]
        op = SemAggOperator(
            task="The largest animal",
            agg_fields=agg_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert "out" in result
        out_items = result["out"].items
        assert len(out_items) == 1
        assert out_items[0]["largest_animal"] == "elephant"

    def test_multiple_agg_fields(self, mock_litellm, mock_llm_config):
        """Multiple aggregation fields are all present in the single output item."""
        mock_litellm.set_completion_handler(_agg_handler_multiple_fields)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        agg_fields = [
            {"name": "largest", "type": str, "description": "largest"},
            {"name": "smallest", "type": str, "description": "smallest"},
        ]
        op = SemAggOperator(
            task="extremes",
            agg_fields=agg_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        item = result["out"].items[0]
        assert item["largest"] == "elephant"
        assert item["smallest"] == "tucan"

    def test_missing_field_defaults_to_none(self, mock_litellm, mock_llm_config):
        """If the LLM omits an agg field, it defaults to None."""
        mock_litellm.set_completion_handler(_agg_handler_partial)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        agg_fields = [
            {"name": "largest", "type": str, "description": "largest"},
            {"name": "smallest", "type": str, "description": "smallest"},
        ]
        op = SemAggOperator(
            task="extremes",
            agg_fields=agg_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        item = result["out"].items[0]
        assert item["largest"] == "elephant"
        assert item["smallest"] is None

    def test_input_dataset_passed_through(self, mock_litellm, mock_llm_config):
        """The original input dataset is present in the returned dict."""
        mock_litellm.set_completion_handler(_agg_handler_largest)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        agg_fields = [{"name": "largest_animal", "type": str, "description": "largest"}]
        op = SemAggOperator(
            task="agg",
            agg_fields=agg_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert "animals" in result
        assert "out" in result
        assert len(result) == 2

    def test_single_llm_call_for_all_items(self, mock_litellm, mock_llm_config):
        """Aggregation makes exactly one LLM call regardless of item count."""
        mock_litellm.set_completion_handler(_agg_handler_largest)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        agg_fields = [{"name": "largest_animal", "type": str, "description": "largest"}]
        op = SemAggOperator(
            task="agg",
            agg_fields=agg_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        op("animals", {"animals": ds})

        assert len(mock_litellm.completion_calls) == 1
