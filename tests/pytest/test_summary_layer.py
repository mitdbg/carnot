"""Unit tests for SummaryLayer.

Tests cover:
1. Cache hit path -- summaries already cached are returned without LLM calls.
2. Cache miss path -- new summaries are generated, cached, and returned.
3. Mixed path -- some cached, some new.
4. Skipping binary files and items without paths.
5. Graceful handling of LLM/embedding failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.agents.models import ChatMessage, LiteLLMModel
from carnot.core.models import LLMCallStats
from carnot.index.models import FileSummaryEntry, HierarchicalIndexConfig
from carnot.index.summary_layer import SummaryLayer

# -- Helpers ------------------------------------------------------------------


def _make_dict_item(path: str, contents: str = "some text") -> dict:
    """Create a plain dict item compatible with SummaryLayer."""
    return {"path": path, "contents": contents}


def _make_mock_model() -> MagicMock:
    """Create a mock LiteLLMModel with default generate/embed stubs."""
    model = MagicMock(spec=LiteLLMModel)
    model.model_id = "test-model"

    generate_response = MagicMock(spec=ChatMessage)
    generate_response.content = "A summary of the file."
    model.generate.return_value = generate_response

    _dummy_embed_stats = LLMCallStats(
        model_id="test-model", call_type="embedding", embedding_input_tokens=10,
    )
    model.embed.return_value = ([[0.1, 0.2, 0.3]], _dummy_embed_stats)
    return model


# -- Tests --------------------------------------------------------------------


class TestSummaryLayerConstruction:
    """SummaryLayer construction and defaults."""

    def test_defaults(self, tmp_path):
        """SummaryLayer uses default config when none provided."""
        layer = SummaryLayer(storage_dir=tmp_path)
        assert isinstance(layer._config, HierarchicalIndexConfig)
        assert layer._cache is not None

    def test_custom_config(self, tmp_path):
        """SummaryLayer accepts a custom config."""
        config = HierarchicalIndexConfig(summary_model="custom-model")
        layer = SummaryLayer(config=config, storage_dir=tmp_path)
        assert layer._config.summary_model == "custom-model"

    def test_accepts_model_instance(self, tmp_path):
        """SummaryLayer stores a provided LiteLLMModel instance."""
        mock_model = _make_mock_model()
        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        assert layer._model is mock_model


class TestSummaryLayerGetOrBuild:
    """SummaryLayer.get_or_build_summaries with mocked LiteLLMModel."""

    def test_generates_summaries_for_items(self, tmp_path):
        """Items without cached summaries are summarized via LLM + embedding."""
        mock_model = _make_mock_model()

        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        items = [_make_dict_item("/data/file1.txt", "Hello world")]
        result = layer.get_or_build_summaries(items)

        assert len(result) == 1
        assert result[0].path == "/data/file1.txt"
        assert result[0].summary == "A summary of the file."
        assert result[0].embedding == [0.1, 0.2, 0.3]
        mock_model.generate.assert_called_once()
        mock_model.embed.assert_called_once()

    def test_caches_summaries(self, tmp_path):
        """Generated summaries are cached; second call does not invoke LLM."""
        mock_model = _make_mock_model()
        generate_response = MagicMock(spec=ChatMessage)
        generate_response.content = "Cached summary."
        mock_model.generate.return_value = generate_response
        mock_model.embed.return_value = ([[1.0, 2.0]], LLMCallStats(
            model_id="test-model", call_type="embedding", embedding_input_tokens=10,
        ))

        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        items = [_make_dict_item("/data/a.txt", "Content A")]

        # First call -- generates and caches
        result1 = layer.get_or_build_summaries(items)
        assert len(result1) == 1
        call_count_after_first = mock_model.generate.call_count

        # Second call -- should hit cache
        result2 = layer.get_or_build_summaries(items)
        assert len(result2) == 1
        assert result2[0].summary == "Cached summary."
        # No additional LLM calls
        assert mock_model.generate.call_count == call_count_after_first

    def test_mixed_cached_and_new(self, tmp_path):
        """Already-cached items skip LLM; only new items are summarized."""
        mock_model = _make_mock_model()
        generate_response = MagicMock(spec=ChatMessage)
        generate_response.content = "New summary."
        mock_model.generate.return_value = generate_response
        mock_model.embed.return_value = ([[0.5]], LLMCallStats(
            model_id="test-model", call_type="embedding", embedding_input_tokens=10,
        ))

        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)

        # Pre-populate cache for /a.txt
        cached_entry = FileSummaryEntry(
            path="/data/a.txt", summary="Pre-cached", embedding=[1.0]
        )
        layer._cache.save(cached_entry)

        items = [
            _make_dict_item("/data/a.txt", "Content A"),
            _make_dict_item("/data/b.txt", "Content B"),
        ]
        result = layer.get_or_build_summaries(items)

        assert len(result) == 2
        summaries_by_path = {r.path: r.summary for r in result}
        assert summaries_by_path["/data/a.txt"] == "Pre-cached"
        assert summaries_by_path["/data/b.txt"] == "New summary."

    def test_skips_binary_files(self, tmp_path):
        """Items with binary file extensions are silently skipped."""
        mock_model = _make_mock_model()
        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        items = [
            _make_dict_item("/data/image.jpg", "binary data"),
            _make_dict_item("/data/archive.zip", "zip data"),
        ]
        result = layer.get_or_build_summaries(items)
        assert result == []

    def test_skips_items_without_path(self, tmp_path):
        """Items with no path key are skipped."""
        mock_model = _make_mock_model()
        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        item = {"contents": "no path key"}
        result = layer.get_or_build_summaries([item])
        assert result == []

    def test_skips_items_with_empty_text(self, tmp_path):
        """Items whose serialized text content is whitespace-only are skipped.

        Note: ``_get_file_text`` serialises the entire dict to JSON, so
        a dict with ``contents=""`` still produces non-empty text.  To
        trigger the skip, mock ``_get_file_text`` to return an empty
        string.
        """
        mock_model = _make_mock_model()
        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        items = [_make_dict_item("/data/empty.txt", "")]
        with patch.object(layer, "_get_file_text", return_value=""):
            result = layer.get_or_build_summaries(items)
        assert result == []
        mock_model.generate.assert_not_called()

    def test_embedding_failure_skips_item(self, tmp_path):
        """When embedding generation fails, the item is skipped."""
        mock_model = _make_mock_model()

        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        with patch.object(layer, "_generate_embedding", return_value=None):
            items = [_make_dict_item("/data/fail.txt", "Some content")]
            result = layer.get_or_build_summaries(items)

        assert result == []

    def test_llm_failure_uses_fallback_summary(self, tmp_path):
        """When LLM completion fails, a fallback preview summary is used."""
        mock_model = _make_mock_model()
        mock_model.generate.side_effect = Exception("API error")
        mock_model.embed.return_value = ([[0.1]], LLMCallStats(
            model_id="test-model", call_type="embedding", embedding_input_tokens=10,
        ))

        layer = SummaryLayer(model=mock_model, storage_dir=tmp_path)
        items = [_make_dict_item("/data/fallback.txt", "Some content here")]
        result = layer.get_or_build_summaries(items)

        assert len(result) == 1
        assert result[0].summary.startswith("File containing:")
