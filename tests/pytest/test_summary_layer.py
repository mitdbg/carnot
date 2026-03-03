"""Unit tests for :class:`SummaryLayer`.

Tests cover:
1. Cache hit path — summaries already cached are returned without LLM calls.
2. Cache miss path — new summaries are generated, cached, and returned.
3. Mixed path — some cached, some new.
4. Skipping binary files and items without paths.
5. Graceful handling of LLM/embedding failures.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from carnot.data.item import DataItem
from carnot.index.models import FileSummaryEntry, HierarchicalIndexConfig
from carnot.index.summary_layer import SummaryLayer

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_data_item(path: str, contents: str = "some text") -> DataItem:
    """Create a DataItem with a path and text content."""
    item = DataItem(path=path)
    item._dict = {"contents": contents}
    return item


# ── Tests ────────────────────────────────────────────────────────────────────


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


class TestSummaryLayerGetOrBuild:
    """SummaryLayer.get_or_build_summaries with mocked LLM."""

    @patch("carnot.index.summary_layer.litellm")
    def test_generates_summaries_for_items(self, mock_litellm, tmp_path):
        """Items without cached summaries are summarized via LLM + embedding."""
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="A summary of the file."))]
        )
        mock_litellm.embedding.return_value = MagicMock(
            data=[{"embedding": [0.1, 0.2, 0.3]}]
        )

        layer = SummaryLayer(storage_dir=tmp_path)
        items = [_make_data_item("/data/file1.txt", "Hello world")]
        result = layer.get_or_build_summaries(items)

        assert len(result) == 1
        assert result[0].path == "/data/file1.txt"
        assert result[0].summary == "A summary of the file."
        assert result[0].embedding == [0.1, 0.2, 0.3]

    @patch("carnot.index.summary_layer.litellm")
    def test_caches_summaries(self, mock_litellm, tmp_path):
        """Generated summaries are cached; second call does not invoke LLM."""
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Cached summary."))]
        )
        mock_litellm.embedding.return_value = MagicMock(
            data=[{"embedding": [1.0, 2.0]}]
        )

        layer = SummaryLayer(storage_dir=tmp_path)
        items = [_make_data_item("/data/a.txt", "Content A")]

        # First call — generates and caches
        result1 = layer.get_or_build_summaries(items)
        assert len(result1) == 1
        call_count_after_first = mock_litellm.completion.call_count

        # Second call — should hit cache
        result2 = layer.get_or_build_summaries(items)
        assert len(result2) == 1
        assert result2[0].summary == "Cached summary."
        # No additional LLM calls
        assert mock_litellm.completion.call_count == call_count_after_first

    @patch("carnot.index.summary_layer.litellm")
    def test_mixed_cached_and_new(self, mock_litellm, tmp_path):
        """Already-cached items skip LLM; only new items are summarized."""
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="New summary."))]
        )
        mock_litellm.embedding.return_value = MagicMock(
            data=[{"embedding": [0.5]}]
        )

        layer = SummaryLayer(storage_dir=tmp_path)

        # Pre-populate cache for /a.txt
        cached_entry = FileSummaryEntry(
            path="/data/a.txt", summary="Pre-cached", embedding=[1.0]
        )
        layer._cache.save(cached_entry)

        items = [
            _make_data_item("/data/a.txt", "Content A"),
            _make_data_item("/data/b.txt", "Content B"),
        ]
        result = layer.get_or_build_summaries(items)

        assert len(result) == 2
        summaries_by_path = {r.path: r.summary for r in result}
        assert summaries_by_path["/data/a.txt"] == "Pre-cached"
        assert summaries_by_path["/data/b.txt"] == "New summary."

    def test_skips_binary_files(self, tmp_path):
        """Items with binary file extensions are silently skipped."""
        layer = SummaryLayer(storage_dir=tmp_path)
        items = [
            _make_data_item("/data/image.jpg", "binary data"),
            _make_data_item("/data/archive.zip", "zip data"),
        ]
        result = layer.get_or_build_summaries(items)
        assert result == []

    def test_skips_items_without_path(self, tmp_path):
        """Items with no path attribute are skipped."""
        layer = SummaryLayer(storage_dir=tmp_path)
        item = DataItem()  # No path
        result = layer.get_or_build_summaries([item])
        assert result == []

    @patch("carnot.index.summary_layer.litellm")
    def test_skips_items_with_empty_text(self, mock_litellm, tmp_path):
        """Items whose text content is empty are skipped."""
        layer = SummaryLayer(storage_dir=tmp_path)
        items = [_make_data_item("/data/empty.txt", "")]
        result = layer.get_or_build_summaries(items)
        assert result == []
        mock_litellm.completion.assert_not_called()

    @patch("carnot.index.summary_layer.litellm")
    def test_embedding_failure_skips_item(self, mock_litellm, tmp_path):
        """When embedding generation fails, the item is skipped."""
        mock_litellm.completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Summary text."))]
        )
        mock_litellm.embedding.return_value = MagicMock(data=[])

        # Patch _generate_embedding to return None (simulating failure)
        layer = SummaryLayer(storage_dir=tmp_path)
        with patch.object(layer, "_generate_embedding", return_value=None):
            items = [_make_data_item("/data/fail.txt", "Some content")]
            result = layer.get_or_build_summaries(items)

        assert result == []

    @patch("carnot.index.summary_layer.litellm")
    def test_llm_failure_uses_fallback_summary(self, mock_litellm, tmp_path):
        """When LLM completion fails, a fallback preview summary is used."""
        mock_litellm.completion.side_effect = Exception("API error")
        mock_litellm.embedding.return_value = MagicMock(
            data=[{"embedding": [0.1]}]
        )

        layer = SummaryLayer(storage_dir=tmp_path)
        items = [_make_data_item("/data/fallback.txt", "Some content here")]
        result = layer.get_or_build_summaries(items)

        assert len(result) == 1
        assert result[0].summary.startswith("File containing:")
