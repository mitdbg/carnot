"""Tier 2 mocked tests for :class:`SemFilterOperator`.

These tests patch ``litellm.completion`` so that **no network calls** are
made.  They validate:

* prompt construction and message flow,
* boolean output parsing (TRUE / FALSE),
* dataset threading (input → filtered output),
* retry behaviour on malformed LLM output.
"""

from __future__ import annotations

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


def _filter_handler_mammals(model, messages, **kwargs):
    """Return TRUE for mammals, FALSE otherwise."""
    from fixtures.mocks import _make_completion_response

    text = msg_text(messages)
    is_mammal = any(m in text for m in _MAMMALS)
    answer = "TRUE" if is_mammal else "FALSE"
    return _make_completion_response(f"```text\n{answer}\n```")


def _filter_handler_all_true(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```text\nTRUE\n```")


def _filter_handler_all_false(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```text\nFALSE\n```")


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
        result = op("animals", {"animals": ds})

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
        result = op("animals", {"animals": ds})

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
        result = op("animals", {"animals": ds})

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
        result = op("animals", {"animals": ds})

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
        result = op("d", {"d": ds})

        # Should succeed after retry
        assert len(result["out"].items) == 1
        # At least 2 calls (first bad, second good)
        assert len(mock_litellm.completion_calls) >= 2

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
        result = op("empty", {"empty": ds})

        assert len(result["out"].items) == 0
        assert len(mock_litellm.completion_calls) == 0
