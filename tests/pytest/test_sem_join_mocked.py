"""Tier 2 mocked tests for :class:`SemJoinOperator`.

These tests patch ``litellm.completion`` so that **no network calls** are
made.  They validate:

* cross-product evaluation (left × right),
* boolean join predicate parsing,
* key-conflict resolution (``left_`` / ``right_`` prefixes),
* dataset threading (two inputs → joined output).
"""

from __future__ import annotations

from helpers.mock_utils import msg_text

from carnot.data.dataset import Dataset
from carnot.operators.sem_join import SemJoinOperator

# ── Helpers ──────────────────────────────────────────────────────────────────

_LEFT = [
    {"animal": "cow"},
    {"animal": "dog"},
    {"animal": "cat"},
]

_RIGHT = [
    {"sound": "moo"},
    {"sound": "woof"},
    {"sound": "meow"},
]

_MATCHES = {
    ("cow", "moo"),
    ("dog", "woof"),
    ("cat", "meow"),
}


def _join_handler_animal_sound(model, messages, **kwargs):
    """Return TRUE if the animal–sound pair is a known match."""
    from fixtures.mocks import _make_completion_response

    # Only look at the last (user) message which contains the items
    last_msg = messages[-1]
    text = msg_text([last_msg])
    for animal, sound in _MATCHES:
        if animal in text and sound in text:
            return _make_completion_response("```text\nTRUE\n```")
    return _make_completion_response("```text\nFALSE\n```")


def _join_handler_all_true(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```text\nTRUE\n```")


def _join_handler_all_false(model, messages, **kwargs):
    from fixtures.mocks import _make_completion_response

    return _make_completion_response("```text\nFALSE\n```")


# ── Tests ────────────────────────────────────────────────────────────────────


class TestSemJoinMocked:
    """Mocked-LLM tests for SemJoinOperator."""

    def test_joins_matching_pairs(self, mock_litellm, mock_llm_config):
        """Only pairs for which the LLM returns TRUE appear in the output."""
        mock_litellm.set_completion_handler(_join_handler_animal_sound)

        left_ds = Dataset(name="left", annotation="test", items=list(_LEFT))
        right_ds = Dataset(name="right", annotation="test", items=list(_RIGHT))
        op = SemJoinOperator(
            task="The animal makes the sound",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        result, _stats = op("left", "right", {"left": left_ds, "right": right_ds})

        assert "out" in result
        out_items = result["out"].items
        assert len(out_items) == 3
        joined_pairs = {(item["animal"], item["sound"]) for item in out_items}
        assert joined_pairs == _MATCHES

    def test_cartesian_join_all_true(self, mock_litellm, mock_llm_config):
        """When the predicate is always TRUE, the output is the full cross product."""
        mock_litellm.set_completion_handler(_join_handler_all_true)

        left_ds = Dataset(name="left", annotation="test", items=list(_LEFT))
        right_ds = Dataset(name="right", annotation="test", items=list(_RIGHT))
        op = SemJoinOperator(
            task="always",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        result, _stats = op("left", "right", {"left": left_ds, "right": right_ds})

        assert len(result["out"].items) == len(_LEFT) * len(_RIGHT)

    def test_no_matches_returns_empty(self, mock_litellm, mock_llm_config):
        """When the predicate is always FALSE, the output is empty."""
        mock_litellm.set_completion_handler(_join_handler_all_false)

        left_ds = Dataset(name="left", annotation="test", items=list(_LEFT))
        right_ds = Dataset(name="right", annotation="test", items=list(_RIGHT))
        op = SemJoinOperator(
            task="never",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        result, _stats = op("left", "right", {"left": left_ds, "right": right_ds})

        assert len(result["out"].items) == 0

    def test_shared_keys_get_prefix(self, mock_litellm, mock_llm_config):
        """When left and right items share a key, the output uses left_/right_ prefixes."""
        mock_litellm.set_completion_handler(_join_handler_all_true)

        left_ds = Dataset(name="left", annotation="test", items=[{"name": "alice"}])
        right_ds = Dataset(name="right", annotation="test", items=[{"name": "bob"}])
        op = SemJoinOperator(
            task="join",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        result, _stats = op("left", "right", {"left": left_ds, "right": right_ds})

        item = result["out"].items[0]
        assert "left_name" in item
        assert "right_name" in item
        assert item["left_name"] == "alice"
        assert item["right_name"] == "bob"

    def test_disjoint_keys_no_prefix(self, mock_litellm, mock_llm_config):
        """When keys do not overlap, no prefix is added."""
        mock_litellm.set_completion_handler(_join_handler_all_true)

        left_ds = Dataset(name="left", annotation="test", items=[{"animal": "cow"}])
        right_ds = Dataset(name="right", annotation="test", items=[{"sound": "moo"}])
        op = SemJoinOperator(
            task="join",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        result, _stats = op("left", "right", {"left": left_ds, "right": right_ds})

        item = result["out"].items[0]
        assert "animal" in item
        assert "sound" in item

    def test_both_input_datasets_passed_through(self, mock_litellm, mock_llm_config):
        """Both original input datasets appear in the returned dict."""
        mock_litellm.set_completion_handler(_join_handler_all_false)

        left_ds = Dataset(name="left", annotation="test", items=list(_LEFT))
        right_ds = Dataset(name="right", annotation="test", items=list(_RIGHT))
        op = SemJoinOperator(
            task="join",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        result, _stats = op("left", "right", {"left": left_ds, "right": right_ds})

        assert "left" in result
        assert "right" in result
        assert "out" in result
        assert len(result) == 3

    def test_completion_calls_equal_cross_product(self, mock_litellm, mock_llm_config):
        """The LLM is called once for each (left, right) pair."""
        mock_litellm.set_completion_handler(_join_handler_all_false)

        left_ds = Dataset(name="left", annotation="test", items=list(_LEFT))
        right_ds = Dataset(name="right", annotation="test", items=list(_RIGHT))
        op = SemJoinOperator(
            task="join",
            model_id="mock-model",
            llm_config=mock_llm_config,
            dataset_id="out",
            max_workers=1,
        )
        op("left", "right", {"left": left_ds, "right": right_ds})

        assert len(mock_litellm.completion_calls) == len(_LEFT) * len(_RIGHT)
