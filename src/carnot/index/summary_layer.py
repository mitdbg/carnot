"""Shared summary-generation layer for semantic indices.

Consolidates file summarization (LLM + embedding) and per-file caching
into a single component that both :class:`FlatFileIndex` and
:class:`HierarchicalFileIndex` can use.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from carnot.index.models import FileSummaryEntry, HierarchicalIndexConfig
from carnot.index.sem_indices_cache import FileSummaryCache
from carnot.storage.config import StorageConfig

if TYPE_CHECKING:
    from carnot.agents.models import LiteLLMModel

logger = logging.getLogger(__name__)

# File types to skip when summarizing
_SKIP_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".zip", ".exe", ".bin"}

# maximum number of characters to summarize (to avoid LLM context limits)
_MAX_SUMMARY_CHARS = 50_000

# maximum number of characters to include in LLM fallback preview
_MAX_PREVIEW_CHARS = 500

# default temperature for LLM summarization (can be tuned for more creative vs. focused summaries)
_SUMMARY_TEMPERATURE = 0.3


class SummaryLayer:
    """Builds and caches :class:`FileSummaryEntry` objects from dict instances.

    Combines LLM-based summarization with embedding generation and
    transparent per-file caching via :class:`FileSummaryCache`.

    Construction parameters:

    - *model* (``LiteLLMModel | None``): the model instance used for
      LLM completion and embedding calls.  When ``None``, a default
      ``LiteLLMModel`` is created using ``config.summary_model``.
    - *config* (``HierarchicalIndexConfig | None``): controls which
      embedding and summary models to use.  Defaults to
      ``HierarchicalIndexConfig()`` when ``None``.
    - *api_key* (``str | None``): OpenAI API key.  Falls back to the
      ``OPENAI_API_KEY`` environment variable.
    - *storage_dir* (``Path | None``): explicit directory for the
      summary cache.  When ``None``, uses ``StorageConfig().summaries_dir``.

    Representation invariant:
        - ``_cache`` is a :class:`FileSummaryCache` instance.
        - ``_config`` is a non-``None`` :class:`HierarchicalIndexConfig`.
        - ``_model`` is a :class:`LiteLLMModel` instance.

    Abstraction function:
        Represents a service that, given a list of dictionaries, produces corresponding
        ``FileSummaryEntry`` objects — fetching from cache when available, and generating
        via LLM + embedding otherwise.
    """

    def __init__(
        self,
        model: LiteLLMModel | None = None,
        config: HierarchicalIndexConfig | None = None,
        api_key: str | None = None,
        storage_dir: Path | None = None,
    ):
        self._config = config or HierarchicalIndexConfig()
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        cache_dir = storage_dir or StorageConfig().summaries_dir
        self._cache = FileSummaryCache(storage_dir=cache_dir)

        from carnot.agents.models import LiteLLMModel as _LiteLLMModel
        if model is not None:
            self._model = model
        else:
            self._model = _LiteLLMModel(
                model_id=self._config.summary_model,
                api_key=self._api_key,
            )

        self._embedding_model = _LiteLLMModel(
            model_id=self._config.embedding_model,
            api_key=self._api_key,
        )

    # ── Public API ──────────────────────────────────────────────────────

    def get_or_build_summaries(
        self, items: list[dict]
    ) -> list[FileSummaryEntry]:
        """Return :class:`FileSummaryEntry` objects for each item.

        Checks the cache first; any items whose summaries are missing
        are summarized via LLM and then cached.

        Requires:
            - *items* is a list of dict instances.

        Returns:
            A list of ``FileSummaryEntry`` objects for all items that
            could be summarized.  Items without a valid path, with a
            binary file extension, or that fail summarization are
            silently skipped.

        Raises:
            None.  Errors for individual items are logged and skipped.
        """
        paths = [
            i.get("path")
            for i in items
            if i.get("path") and Path(i["path"]).suffix.lower() not in _SKIP_SUFFIXES
        ]

        loaded, missing_paths = self._cache.load_many(paths)
        items_to_compute = [
            i for i in items if i.get("path") in missing_paths
        ]

        if items_to_compute:
            new_entries = self._build_file_summaries(items_to_compute)
            for entry in new_entries:
                loaded[entry.path] = entry

        return list(loaded.values())

    # ── Private helpers ─────────────────────────────────────────────────

    def _build_file_summaries(
        self, items: list[dict]
    ) -> list[FileSummaryEntry]:
        """Generate summaries for items that are not in cache.

        Requires:
            - *items* is a list of dictionaries with valid ``"path"`` keys.

        Returns:
            A list of newly-generated ``FileSummaryEntry`` objects.

        Raises:
            None.  Errors for individual items are logged and skipped.
        """
        entries: list[FileSummaryEntry] = []

        for item in items:
            if not item['path']:
                continue
            if Path(item['path']).suffix.lower() in _SKIP_SUFFIXES:
                continue
            try:
                text = self._get_file_text(item)
                if not text.strip():
                    continue

                summary_text = self._generate_summary(item['path'], text)
                embedding = self._generate_embedding(summary_text)
                if embedding is None:
                    continue

                entry = FileSummaryEntry(
                    path=item['path'],
                    summary=summary_text,
                    embedding=embedding,
                )
                entries.append(entry)

                try:
                    self._cache.save(entry)
                except Exception as e:
                    logger.warning(
                        f"Failed to persist summary for {item['path']}: {e}"
                    )
            except Exception as e:
                logger.warning(f"Failed to summarize {item['path']}: {e}")

        return entries

    @staticmethod
    def _get_file_text(item: dict) -> str:
        """Extract text content from a dict.

        Requires:
            - *item* is a dict instance.

        Returns:
            The text content, or an empty string if extraction fails.

        Raises:
            None.
        """
        try:
            return json.dumps(item, indent=2)
        except Exception:
            return ""

    def _generate_summary(self, file_path: str, text_content: str) -> str:
        """Generate a summary of *text_content* via LLM.

        Requires:
            - *text_content* is a non-empty string.

        Returns:
            A summary string.

        Raises:
            None.  Falls back to a truncated preview on LLM failure.
        """
        if len(text_content) > _MAX_SUMMARY_CHARS:
            text_content = (
                text_content[:_MAX_SUMMARY_CHARS] + "\n\n[... content truncated ...]"
            )

        prompt = f"""Analyze the following file and create a comprehensive summary for semantic routing and retrieval.

File: {file_path}

Content:
{text_content}

Create a detailed summary (1-2 paragraphs) that:
1. Identifies the main topics, themes, and purpose of the document
2. Highlights key entities, names, dates, events, and numerical data mentioned
3. Captures important relationships, decisions, or conclusions
4. Uses specific terms and phrases that would match relevant search queries
5. Notes the document type and any structural elements (e.g., contract clauses, email thread)

The summary should be rich enough to enable accurate routing when users search for specific content. Be thorough but focused—include concrete details rather than generic descriptions.

Summary:"""

        try:
            from carnot.agents.models import ChatMessage

            message = ChatMessage(role="user", content=prompt)
            response = self._model.generate(
                messages=[message],
                temperature=_SUMMARY_TEMPERATURE,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(
                f"LLM summarization failed for {file_path}: {e}, using preview fallback",
            )
            preview = text_content[:_MAX_PREVIEW_CHARS].replace("\n", " ")
            return f"File containing: {preview}..."

    def _generate_embedding(self, text: str) -> list[float] | None:
        """Generate an embedding vector for *text*.

        Requires:
            - *text* is a non-empty string.

        Returns:
            A list of floats, or ``None`` if embedding generation
            fails.

        Raises:
            None.  Errors are logged.
        """
        try:
            embeddings, _embed_stats = self._embedding_model.embed(texts=[text])
            return embeddings[0]
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None
