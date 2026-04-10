"""Tier 2 mocked tests for :class:`SemFlatMapOperator`.

These tests patch ``litellm.completion`` so that **no network calls** are
made.  They validate:

* JSON list output parsing and flattening,
* one-to-many expansion (flat map semantics),
* missing output fields default to ``None``,
* dataset threading.
"""

from __future__ import annotations

import json

from carnot.data.dataset import Dataset
from carnot.operators.sem_flat_map import SemFlatMapOperator

# ── Helpers ──────────────────────────────────────────────────────────────────


def _flatmap_handler_fruits(model, messages, **kwargs):
    """Return a list of extracted fruit dicts."""
    from fixtures.mocks import _make_completion_response

    fruits = [
        {"fruit": "apple", "color": "red"},
        {"fruit": "banana", "color": "yellow"},
        {"fruit": "cherry", "color": "red"},
    ]
    blob = json.dumps(fruits)
    return _make_completion_response(f"```json\n{blob}\n```")


def _flatmap_handler_single(model, messages, **kwargs):
    """Return a single-element list."""
    from fixtures.mocks import _make_completion_response

    blob = json.dumps([{"key": "value"}])
    return _make_completion_response(f"```json\n{blob}\n```")


def _flatmap_handler_empty_list(model, messages, **kwargs):
    """Return an empty list — item produces zero output rows."""
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```json\n[]\n```")


def _flatmap_handler_partial_fields(model, messages, **kwargs):
    """Return items missing one of the expected fields."""
    from fixtures.mocks import _make_completion_response

    blob = json.dumps([{"present": "yes"}])
    return _make_completion_response(f"```json\n{blob}\n```")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSemFlatMapMocked:
    """Mocked-LLM tests for SemFlatMapOperator."""

    def test_expands_items(self, mock_litellm, mock_llm_config):
        """A single input item can produce multiple output items."""
        mock_litellm.set_completion_handler(_flatmap_handler_fruits)

        ds = Dataset(name="docs", annotation="test", items=[{"text": "Apple Banana Cherry"}])
        output_fields = [
            {"name": "fruit", "type": str, "description": "fruit name"},
            {"name": "color", "type": str, "description": "fruit color"},
        ]
        op = SemFlatMapOperator(
            task="Extract fruits",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("docs", {"docs": ds})

        assert "out" in result
        out_items = result["out"].items
        assert len(out_items) == 3
        names = {item["fruit"] for item in out_items}
        assert names == {"apple", "banana", "cherry"}

    def test_multiple_input_items_flattened(self, mock_litellm, mock_llm_config):
        """Outputs from multiple input items are flattened into one list."""
        mock_litellm.set_completion_handler(_flatmap_handler_fruits)

        ds = Dataset(name="docs", annotation="test", items=[{"text": "a"}, {"text": "b"}])
        output_fields = [
            {"name": "fruit", "type": str, "description": "fruit name"},
            {"name": "color", "type": str, "description": "fruit color"},
        ]
        op = SemFlatMapOperator(
            task="Extract fruits",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("docs", {"docs": ds})

        # 2 input items × 3 fruits each = 6
        assert len(result["out"].items) == 6

    def test_single_element_list(self, mock_litellm, mock_llm_config):
        """When the LLM returns a single-element list, exactly one output item appears."""
        mock_litellm.set_completion_handler(_flatmap_handler_single)

        ds = Dataset(name="d", annotation="test", items=[{"x": 1}])
        output_fields = [{"name": "key", "type": str, "description": "k"}]
        op = SemFlatMapOperator(
            task="extract",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("d", {"d": ds})

        assert len(result["out"].items) == 1
        assert result["out"].items[0]["key"] == "value"

    def test_empty_list_produces_no_items(self, mock_litellm, mock_llm_config):
        """An empty list from the LLM means the input item generates zero output rows."""
        mock_litellm.set_completion_handler(_flatmap_handler_empty_list)

        ds = Dataset(name="d", annotation="test", items=[{"x": 1}])
        output_fields = [{"name": "key", "type": str, "description": "k"}]
        op = SemFlatMapOperator(
            task="extract",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("d", {"d": ds})

        assert len(result["out"].items) == 0

    def test_missing_field_defaults_to_none(self, mock_litellm, mock_llm_config):
        """If the LLM omits an output field, it defaults to None."""
        mock_litellm.set_completion_handler(_flatmap_handler_partial_fields)

        ds = Dataset(name="d", annotation="test", items=[{"x": 1}])
        output_fields = [
            {"name": "present", "type": str, "description": "here"},
            {"name": "absent", "type": str, "description": "missing"},
        ]
        op = SemFlatMapOperator(
            task="extract",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("d", {"d": ds})

        item = result["out"].items[0]
        assert item["present"] == "yes"
        assert item["absent"] is None

    def test_input_dataset_passed_through(self, mock_litellm, mock_llm_config):
        """The original input dataset is present in the returned dict."""
        mock_litellm.set_completion_handler(_flatmap_handler_fruits)

        ds = Dataset(name="docs", annotation="test", items=[{"text": "a"}])
        output_fields = [
            {"name": "fruit", "type": str, "description": "fruit"},
            {"name": "color", "type": str, "description": "color"},
        ]
        op = SemFlatMapOperator(
            task="extract",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("docs", {"docs": ds})

        assert "docs" in result
        assert "out" in result

    def test_empty_input_dataset(self, mock_litellm, mock_llm_config):
        """An empty input dataset produces an empty output dataset."""
        mock_litellm.set_completion_handler(_flatmap_handler_fruits)

        ds = Dataset(name="empty", annotation="test", items=[])
        output_fields = [{"name": "fruit", "type": str, "description": "f"}]
        op = SemFlatMapOperator(
            task="extract",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("empty", {"empty": ds})

        assert len(result["out"].items) == 0
        assert len(mock_litellm.completion_calls) == 0
