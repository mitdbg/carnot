"""Unit tests for `match_model_entry` — the shared exact-then-substring model-key matcher used by
both the price table and the per-model context-limit map."""

from __future__ import annotations

from skunk.usage import _match_price, match_model_entry


def test_exact_match_case_insensitive():
    m = {"Qwen/Qwen3.6-35B-A3B": 262144}
    assert match_model_entry("qwen/qwen3.6-35b-a3b", m) == 262144


def test_substring_match():
    # A key that is a substring of the model id matches (e.g. "qwen3" in "qwen/qwen3.6-...").
    assert match_model_entry("qwen/qwen3.6-35b-a3b", {"qwen3": 131072}) == 131072
    assert match_model_entry("google/gemini-3.5-flash", {"gemini-3.5-flash": 1048576}) == 1048576


def test_no_match_returns_none():
    assert match_model_entry("google/gemini-3.5-flash", {"qwen3": 131072}) is None
    assert match_model_entry("anything", {}) is None


def test_match_price_delegates_unchanged():
    prices = {"gemini-3.5-flash": {"in": 1.5, "out": 9.0}}
    assert _match_price("google/gemini-3.5-flash", prices) == {"in": 1.5, "out": 9.0}
    assert _match_price("no-such-model", prices) is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
