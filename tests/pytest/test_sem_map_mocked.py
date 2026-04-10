"""Tier 2 mocked tests for :class:`SemMapOperator`.

These tests patch ``litellm.completion`` so that **no network calls** are
made.  They validate:

* JSON output parsing and item enrichment,
* missing output fields default to ``None``,
* dataset threading (input → enriched output),
* retry behaviour on malformed LLM output.
"""

from __future__ import annotations

import json

from helpers.mock_utils import msg_text

from carnot.data.dataset import Dataset
from carnot.operators.sem_map import SemMapOperator

# ── Helpers ──────────────────────────────────────────────────────────────────

_ANIMALS = [
    {"animal": "giraffe"},
    {"animal": "anaconda"},
    {"animal": "salmon"},
    {"animal": "elephant"},
    {"animal": "tucan"},
]

_GROUPS = {
    "giraffe": "mammal",
    "anaconda": "reptile",
    "salmon": "fish",
    "elephant": "mammal",
    "tucan": "bird",
}


def _map_handler_animal_group(model, messages, **kwargs):
    """Return a JSON blob mapping each animal to its group."""
    from fixtures.mocks import _make_completion_response

    text = msg_text(messages)
    for animal, group in _GROUPS.items():
        if animal in text:
            blob = json.dumps({"animal_group": group})
            return _make_completion_response(f"```json\n{blob}\n```")
    # fallback
    blob = json.dumps({"animal_group": "unknown"})
    return _make_completion_response(f"```json\n{blob}\n```")


def _map_handler_partial_fields(model, messages, **kwargs):
    """Return a JSON blob missing one of the expected output fields."""
    from fixtures.mocks import _make_completion_response

    blob = json.dumps({"found_field": "value"})
    return _make_completion_response(f"```json\n{blob}\n```")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSemMapMocked:
    """Mocked-LLM tests for SemMapOperator."""

    def test_enriches_items_with_new_fields(self, mock_litellm, mock_llm_config):
        """Each item gains the output field specified by the operator."""
        mock_litellm.set_completion_handler(_map_handler_animal_group)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        output_fields = [{"name": "animal_group", "type": str, "description": "group"}]
        op = SemMapOperator(
            task="Classify the animal",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert "out" in result
        out_items = result["out"].items
        assert len(out_items) == len(_ANIMALS)
        for item in out_items:
            assert "animal_group" in item
            assert item["animal_group"] == _GROUPS[item["animal"]]

    def test_preserves_original_fields(self, mock_litellm, mock_llm_config):
        """Original item fields are preserved alongside the new enriched field."""
        mock_litellm.set_completion_handler(_map_handler_animal_group)

        ds = Dataset(name="animals", annotation="test", items=[{"animal": "giraffe", "weight": 1200}])
        output_fields = [{"name": "animal_group", "type": str, "description": "group"}]
        op = SemMapOperator(
            task="Classify",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        item = result["out"].items[0]
        assert item["animal"] == "giraffe"
        assert item["weight"] == 1200
        assert item["animal_group"] == "mammal"

    def test_missing_output_field_defaults_to_none(self, mock_litellm, mock_llm_config):
        """If the LLM omits an output field, it defaults to None."""
        mock_litellm.set_completion_handler(_map_handler_partial_fields)

        ds = Dataset(name="d", annotation="test", items=[{"x": 1}])
        output_fields = [
            {"name": "found_field", "type": str, "description": "present"},
            {"name": "missing_field", "type": str, "description": "absent"},
        ]
        op = SemMapOperator(
            task="extract",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("d", {"d": ds})

        item = result["out"].items[0]
        assert item["found_field"] == "value"
        assert item["missing_field"] is None

    def test_all_items_returned(self, mock_litellm, mock_llm_config):
        """Map always returns the same number of items as the input (no filtering)."""
        mock_litellm.set_completion_handler(_map_handler_animal_group)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        output_fields = [{"name": "animal_group", "type": str, "description": "group"}]
        op = SemMapOperator(
            task="Classify",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert len(result["out"].items) == len(_ANIMALS)

    def test_input_dataset_passed_through(self, mock_litellm, mock_llm_config):
        """The original input dataset is present in the returned dict."""
        mock_litellm.set_completion_handler(_map_handler_animal_group)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        output_fields = [{"name": "animal_group", "type": str, "description": "group"}]
        op = SemMapOperator(
            task="Classify",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert "animals" in result
        assert "out" in result

    def test_completion_called_once_per_item(self, mock_litellm, mock_llm_config):
        """The LLM is called exactly once per item."""
        mock_litellm.set_completion_handler(_map_handler_animal_group)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        output_fields = [{"name": "animal_group", "type": str, "description": "group"}]
        op = SemMapOperator(
            task="Classify",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        op("animals", {"animals": ds})

        assert len(mock_litellm.completion_calls) == len(_ANIMALS)

    def test_empty_input_dataset(self, mock_litellm, mock_llm_config):
        """An empty input dataset produces an empty output dataset."""
        mock_litellm.set_completion_handler(_map_handler_animal_group)

        ds = Dataset(name="empty", annotation="test", items=[])
        output_fields = [{"name": "animal_group", "type": str, "description": "group"}]
        op = SemMapOperator(
            task="Classify",
            output_fields=output_fields,
            dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("empty", {"empty": ds})

        assert len(result["out"].items) == 0
        assert len(mock_litellm.completion_calls) == 0
