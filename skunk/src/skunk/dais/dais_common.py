"""Shared helpers for the DAIS slim pipeline.

Two things every DAIS script needs:

* **Generic identifiers.** Unlike OfficeQA (whose filenames encode ``year``/``month``),
  the DAIS corpus filenames are treated opaquely: ``file_id`` is the JSON/PDF stem and the
  ``doc_id`` / ``chunk_id`` are built purely from ``file_id`` + ``page_id`` + element id.
  These mirror the OfficeQA ``page_key`` / ``unique_element_id`` schemes structurally so the
  downstream SearchAgent (``doc_id`` + ``chunk_id``) and ChromaDB metadata work unchanged.

* **``DaisLLM``.** A thin wrapper over two ``LLMClient``s — Gemini (genai) and OpenRouter —
  that runs every call on Gemini first and, on a *persistent* rate-limit / transient error
  (after the genai client's own backoff), automatically retries the same call on OpenRouter
  with the ``google/``-prefixed model id. This is the automatic fallback the user asked for
  on the standalone (non-SearchAgent) LLM calls.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from pathlib import Path

from skunk.config import SkunkConfig
from skunk.common import B64Image
from skunk.llm_client import LLMClient, LLMResponse, _is_retryable

log = logging.getLogger(__name__)

# Element types that carry no useful retrievable text — skipped both when embedding
# (matches ``compute_officeqa_element_embeddings.py``) and when building cleaned pages.
SKIP_TYPES = {"page_number", "page_footer", "page_header", "figure"}

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

# A standalone 4-digit year token (1700–2099), not glued to other digits — distinguishes the
# publication year from incidental digit runs in the file id (e.g. govinfo serialset numbers).
_YEAR4_RE = re.compile(r"(?<!\d)(1[789]\d\d|20\d\d)(?!\d)")
# Two-digit transition years (e.g. `annrpt95`, `appendix98`); no 4-digit token present.
_YEAR2_RE = re.compile(r"(?:appendix|annrpt)(\d{2})")


# --------------------------------------------------------------------------- ids
def file_id_of(path: str | Path) -> str:
    """The corpus file identifier: the bare stem of a parsed-JSON or PDF file."""
    return Path(path).stem


def doc_id_of(file_id: str, page_id: int | str) -> str:
    """Document (== page) id, the SearchAgent ``doc_id`` / OfficeQA ``page_key`` analogue.

    ``page_id`` is **1-indexed** (matches the PDF viewer's page number)."""
    return f"{file_id}_{page_id}"


def chunk_id_of(file_id: str, page_id: int | str, element_id: int | str) -> str:
    """Per-element id, the SearchAgent ``chunk_id`` / OfficeQA ``unique_element_id`` analogue.

    ``page_id`` is **1-indexed** (matches the PDF viewer's page number)."""
    return f"{file_id}_{page_id}_{element_id}"


def pdf_path_for(file_id: str, pdfs_dir: str | Path) -> Path:
    """``<pdfs_dir>/<file_id>.pdf`` — the generic inverse of :func:`file_id_of`."""
    return Path(pdfs_dir) / f"{file_id}.pdf"


def year_of(file_id: str) -> int | None:
    """Publication year parsed from the file id, or None if not determinable.

    The DAIS corpus is annual; filenames embed the year but inconsistently across families:
      * ``combined_statement__historical__cs-1872``  -> 1872  (4-digit)
      * ``combined_statement__modern__2001__c01``    -> 2001  (4-digit)
      * ``govinfo_receipts__1893__SERIALSET-…``      -> 1893  (4-digit, before the serial id)
      * ``combined_statement__transition__annrpt95`` -> 1995  (2-digit transition years)
      * ``combined_statement__transition__appendix00`` -> 2000

    Strategy: take the first standalone 4-digit token in 1700–2099; otherwise a 2-digit
    transition year, mapped ``NN >= 30 -> 19NN`` else ``20NN`` (the transition era is the
    late-1990s rollover, so this band is unambiguous for this corpus).
    """
    m4 = _YEAR4_RE.search(file_id)
    if m4:
        return int(m4.group(1))
    m2 = _YEAR2_RE.search(file_id)
    if m2:
        nn = int(m2.group(1))
        return 1900 + nn if nn >= 30 else 2000 + nn
    return None


def source_of(file_id: str) -> str:
    """Coarse source/era family for filtering.

    Returns one of ``historical`` / ``modern`` / ``transition`` (the era sub-field of the
    ``combined_statement__<era>__…`` names) or ``govinfo_receipts`` (its own top-level
    source). The specific file/section within an era is preserved by :func:`file_id_of`
    (and hence ``doc_id`` / ``chunk_id``), so this column stays low-cardinality."""
    parts = file_id.split("__")
    if parts[0] == "combined_statement" and len(parts) > 1:
        return parts[1]
    return parts[0]


# --------------------------------------------------------------------------- LLM
class DaisLLM:
    """Two-provider LLM client (genai + OpenRouter) with automatic fallback.

    Provider *order* depends on the model id: a bare id (e.g. ``gemini-3.5-flash``) runs
    **genai first** with OpenRouter fallback; a provider-qualified id (e.g.
    ``google/gemini-2.5-flash``) runs **OpenRouter first** with genai fallback. The latter
    is the escape hatch when genai is unhealthy (e.g. 500 storms or hung streams that the
    per-call retry can't recover from).

    ``call(...)`` is drop-in compatible with ``LLMClient.call`` (returns an
    ``LLMResponse`` with a ``.text`` field), so the table-correction logic can use it
    exactly like the stock ``LLMClient``.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        *,
        base_config: SkunkConfig | None = None,
        enable_fallback: bool = True,
    ) -> None:
        base = base_config or SkunkConfig.from_env()
        # genai uses the bare model id; OpenRouter wants the `google/<model>` id.
        self.model = model.removeprefix("google/")
        self.or_model = model if "/" in model else f"google/{model}"
        # LLMClient construction is cheap (the underlying SDK client is built lazily on
        # first use), so an OpenRouter key is only required if/when we actually fall back.
        self._genai = LLMClient(
            dataclasses.replace(base, llm_provider="genai", llm_model=self.model)
        )
        self._openrouter = LLMClient(
            dataclasses.replace(base, llm_provider="openrouter", llm_model=self.or_model)
        )
        self._enable_fallback = enable_fallback
        # A provider-qualified id (has a "/") means "use OpenRouter as the primary".
        self._prefer_openrouter = "/" in model

    def call(
        self,
        system: str,
        user: str,
        images: list[B64Image] | None = None,
        **kwargs,
    ) -> LLMResponse:
        primary, secondary, p_name, s_name = (
            (self._openrouter, self._genai, "openrouter", "genai")
            if self._prefer_openrouter
            else (self._genai, self._openrouter, "genai", "openrouter")
        )
        try:
            return primary.call(system=system, user=user, images=images, **kwargs)
        except Exception as e:  # noqa: BLE001 - decide fallback vs re-raise below
            # Only fall back for transient/throttling failures; a non-retryable error
            # (bad request, context overflow, auth) would fail on the other provider too.
            if not (self._enable_fallback and _is_retryable(e)):
                raise
            log.warning(
                "%s call failed (%s: %s); falling back to %s",
                p_name, type(e).__name__, e, s_name,
            )
            return secondary.call(system=system, user=user, images=images, **kwargs)
