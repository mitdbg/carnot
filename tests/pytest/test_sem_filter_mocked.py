"""Tier 2 mocked tests for :class:`SemFilterOperator`.

These tests patch ``litellm.completion`` so that **no network calls** are
made.  They validate:

* prompt construction and message flow,
* boolean output parsing (TRUE / FALSE),
* dataset threading (input → filtered output),
* retry behaviour on malformed LLM output.
"""

from __future__ import annotations

import json

from helpers.mock_utils import msg_text

from carnot.data.dataset import Dataset
from carnot.operators.sem_filter import SemFilterOperator

# ── Helpers ──────────────────────────────────────────────────────────────────

_ANIMALS = [
    {"animal": "giraffe"},
    {"animal": "anaconda"},
    {"animal": "salmon"},
    {"animal": "elephant"},
    {"animal": "tucan"},
]

_MAMMALS = {"giraffe", "elephant"}


def _latest_payload(messages, marker: str):
    text = msg_text(messages)
    payload_text = text.rsplit(f"{marker}\n", 1)[-1].strip()
    return json.loads(payload_text)


def _filter_handler_mammals(model, messages, **kwargs):
    """Return TRUE for mammals, FALSE otherwise."""
    from fixtures.mocks import _make_completion_response

    item = _latest_payload(messages, "Input:")
    is_mammal = item.get("animal") in _MAMMALS
    answer = "TRUE" if is_mammal else "FALSE"
    return _make_completion_response(f"```text\n{answer}\n```")


def _filter_handler_all_true(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```text\nTRUE\n```")


def _filter_handler_all_false(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```text\nFALSE\n```")


def _filter_handler_batch_mammals(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    text = msg_text(messages)
    if "Inputs:\n" in text:
        items = _latest_payload(messages, "Inputs:")
        matching_indices = [
            item["batch_index"]
            for item in items
            if item.get("animal") in _MAMMALS
        ]
        return _make_completion_response(
            "```json\n"
            + json.dumps({"matching_indices": matching_indices})
            + "\n```"
        )

    item = _latest_payload(messages, "Input:")
    answer = "TRUE" if item.get("animal") in _MAMMALS else "FALSE"
    return _make_completion_response(f"```text\n{answer}\n```")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSemFilterMocked:
    """Mocked-LLM tests for SemFilterOperator."""

    def test_filters_items_by_boolean_response(self, mock_litellm, mock_llm_config):
        """Items for which the LLM returns TRUE are kept; FALSE items are dropped."""
        mock_litellm.set_completion_handler(_filter_handler_mammals)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="The animal is a mammal",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert "out" in result
        out_items = result["out"].items
        assert len(out_items) == 2
        out_names = {item["animal"] for item in out_items}
        assert out_names == _MAMMALS

    def test_all_pass_preserves_every_item(self, mock_litellm, mock_llm_config):
        """When the LLM returns TRUE for every item, all items appear in the output."""
        mock_litellm.set_completion_handler(_filter_handler_all_true)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="always true",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert len(result["out"].items) == len(_ANIMALS)

    def test_none_pass_returns_empty_dataset(self, mock_litellm, mock_llm_config):
        """When the LLM returns FALSE for every item, the output dataset is empty."""
        mock_litellm.set_completion_handler(_filter_handler_all_false)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="always false",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert len(result["out"].items) == 0

    def test_input_dataset_passed_through(self, mock_litellm, mock_llm_config):
        """The original input dataset is present in the output dict alongside the new dataset."""
        mock_litellm.set_completion_handler(_filter_handler_all_true)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="pass",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("animals", {"animals": ds})

        assert "animals" in result
        assert "out" in result
        assert len(result) == 2

    def test_completion_called_once_per_item(self, mock_litellm, mock_llm_config):
        """The LLM is called exactly once per input item (no retries when output parses correctly)."""
        mock_litellm.set_completion_handler(_filter_handler_all_true)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="pass",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        op("animals", {"animals": ds})

        assert len(mock_litellm.completion_calls) == len(_ANIMALS)

    def test_retry_on_malformed_output(self, mock_litellm, mock_llm_config):
        """When the first response is unparseable, the operator retries up to max_steps."""
        call_count = {"n": 0}

        def _bad_then_good(model, messages, **kwargs):
            from fixtures.mocks import _make_completion_response

            call_count["n"] += 1
            if call_count["n"] % 2 == 1:
                # First call: malformed (no tags)
                return _make_completion_response("I think yes")
            # Second call: proper
            return _make_completion_response("```text\nTRUE\n```")

        mock_litellm.set_completion_handler(_bad_then_good)

        ds = Dataset(name="d", annotation="test", items=[{"x": 1}])
        op = SemFilterOperator(
            task="pass",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
            max_steps=3,
        )
        result, _stats = op("d", {"d": ds})

        # Should succeed after retry
        assert len(result["out"].items) == 1
        # At least 2 calls (first bad, second good)
        assert len(mock_litellm.completion_calls) >= 2

    def test_batched_filters_items_by_json_indices(self, mock_litellm, mock_llm_config):
        """When batch_size > 1, matching items are selected from JSON indices."""
        mock_litellm.set_completion_handler(_filter_handler_batch_mammals)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="The animal is a mammal",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
            batch_size=2,
        )
        result, _stats = op("animals", {"animals": ds})

        out_items = result["out"].items
        assert len(out_items) == 2
        assert {item["animal"] for item in out_items} == _MAMMALS

    def test_batched_parse_failure_falls_back_to_smaller_batches(self, mock_litellm, mock_llm_config):
        """Malformed batch output triggers recursive fallback to smaller batches."""
        from fixtures.mocks import _make_completion_response

        def _bad_batch_good_single(model, messages, **kwargs):
            text = msg_text(messages)
            if "Inputs:\n" in text:
                return _make_completion_response("```json\nnot valid json\n```")

            item = _latest_payload(messages, "Input:")
            answer = "TRUE" if item.get("animal") in _MAMMALS else "FALSE"
            return _make_completion_response(f"```text\n{answer}\n```")

        mock_litellm.set_completion_handler(_bad_batch_good_single)

        ds = Dataset(name="animals", annotation="test", items=list(_ANIMALS))
        op = SemFilterOperator(
            task="The animal is a mammal",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
            max_steps=2,
            batch_size=4,
        )
        result, _stats = op("animals", {"animals": ds})

        out_items = result["out"].items
        assert {item["animal"] for item in out_items} == _MAMMALS
        assert any("Inputs:\n" in msg_text(call["messages"]) for call in mock_litellm.completion_calls)
        assert any("Input:\n" in msg_text(call["messages"]) for call in mock_litellm.completion_calls)

    def test_oversized_batch_falls_back_before_llm_call(self, mock_litellm, mock_llm_config):
        """Oversized batches are split until a smaller execution path is safe."""
        mock_litellm.set_completion_handler(_filter_handler_all_true)

        ds = Dataset(
            name="docs",
            annotation="test",
            items=[{"text": "x" * 1000} for _ in range(4)],
        )
        op = SemFilterOperator(
            task="always true",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
            batch_size=4,
        )
        op._MAX_BATCH_INPUT_CHARS = 10

        result, _stats = op("docs", {"docs": ds})

        assert len(result["out"].items) == 4
        assert all("Inputs:\n" not in msg_text(call["messages"]) for call in mock_litellm.completion_calls)
        assert len(mock_litellm.completion_calls) == 4

    def test_empty_input_dataset(self, mock_litellm, mock_llm_config):
        """An empty input dataset produces an empty output dataset with no LLM calls."""
        mock_litellm.set_completion_handler(_filter_handler_all_true)

        ds = Dataset(name="empty", annotation="test", items=[])
        op = SemFilterOperator(
            task="irrelevant",
            output_dataset_id="out",
            model_id="mock-model",
            llm_config=mock_llm_config,
            max_workers=1,
        )
        result, _stats = op("empty", {"empty": ds})

        assert len(result["out"].items) == 0
        assert len(mock_litellm.completion_calls) == 0
