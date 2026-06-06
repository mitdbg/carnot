"""Per-page summarize stage: one LLM pass per page -> clean retrieval summary.

The deterministic catalog build (`catalog.py`) gives a green scaffold per
page: verbatim table titles, column/row labels, figure markers, and a
sanitized page text. This stage runs ONE LLM judgment per page over that
scaffold to:

  - decide whether the page carries retrievable content (`is_content`);
  - typo-fix the table `title` / `column_headers` / `row_headers`
    IN PLACE, same shape/order (this also gives the placer cleaner input);
  - produce the retrieval-facing fields the semantic / year filters read
    (`summary`, `date_interval`, `keywords`).

It produces `keywords` from scratch (the deterministic catalog build no longer
harvests them). It never touches `banner_self`, `printed_page`, or `section` —
those are placement/merge inputs.

Label parity is the top correctness risk: a typo-fixed list MUST keep its
length/order. The batch is re-issued on a page-count mismatch (positional
misalignment would silently corrupt verdicts); within a page, a label list
whose length doesn't match the verbatim input is left verbatim.
"""

from __future__ import annotations

import json
import logging

from skunk.common import LLMClient, chunk, parse_json_response
from skunk.corpus import page_sanitized_text

from .catalog import BLANK_CHAR_THRESHOLD, has_visual_elements
from .data_model import BuildPage

log = logging.getLogger(__name__)


# 1–2 line corpus hint. Hardcoded — this package is single-corpus
# (Treasury Bulletin); there is no profile/registration layer to inject it.
_TREASURY_HINT = (
    "Corpus: the U.S. Treasury Bulletin, a monthly statistical publication on "
    "federal finance (public debt, receipts and outlays, ownership of Treasury "
    "securities, exchange stabilization, foreign currency positions, etc.). "
    "Dates use U.S. conventions: fiscal year FY{N} runs Oct 1 of {N-1} through "
    "Sep 30 of {N}; a calendar year runs Jan 1 through Dec 31. Resolve partial "
    "or relative dates against the bulletin's own publication month."
)

# Page-text budget per page in the prompt (topic, not the numeric grid).
_PAGE_TEXT_CHARS = 2000

# Pages summarized per LLM call. One judgment per page; batched for throughput.
_BATCH_SIZE = 12

# Re-issue a batch whose response length doesn't match its input this many
# times before giving up (and leaving those pages verbatim).
_PARITY_MAX_RETRIES = 3


_SYSTEM = """\
You clean the retrieval metadata for one page of a statistical document at a time.
You get a compact machine-extracted scaffold per page (table titles, column/row labels,
figure markers, short sanitized text) plus a one-line corpus hint — not the numeric values.

Return one JSON object per input page with these fields:
  - "is_content": true if the page reports retrievable data/figures/prose; false for
    blank, toc, index, front-matter, or masthead pages.
  - "page_role": one of "content", "toc", "index", "front_matter", "masthead", "blank".
  - "blocks": EXACTLY one entry per input block, SAME order. Each:
      - "title": block title, typos fixed (stay faithful; don't invent).
      - "column_headers": input column_headers, typos fixed — SAME count and ORDER
        (never add, drop, reorder, or merge).
      - "row_labels": input row_labels, typos fixed — SAME count and ORDER.
      - "summary": short noun phrase for what the block is about, or null.
  - "date_interval": data span as ["YYYY-MM", "YYYY-MM"] months using the corpus
    hint's fiscal/calendar conventions; null if the page has no datable span.
  - "keywords": salient terms, topics, and entities a reader would search for
    to find this page (may be empty).

Output a SINGLE JSON array of these objects, one per input page, in order. No prose, no fences.
"""


def _norm_interval(raw) -> tuple[str, str] | None:
    """Coerce an LLM `date_interval` to a clean `YYYY-MM` (low, high) tuple, or
    None. Any day component is dropped, so the stored span is month-granular."""
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = raw
        if lo and hi:
            return (str(lo).strip()[:7], str(hi).strip()[:7])
    return None


class PageSummarizer:
    """Drives the per-page LLM pass for one bulletin at a time."""

    def summarize_bulletin(
        self,
        *,
        bulletin: str,
        rows: list[BuildPage],
        pages: dict[int, list[dict]],
        llm: LLMClient,
        batch_size: int = _BATCH_SIZE,
    ) -> int:
        """Enrich `rows` in place. Returns the number of pages updated by the
        LLM (truly-blank pages are handled deterministically and not counted)."""
        rows_sorted = sorted(rows, key=lambda r: r.page)

        # Deterministic green skip: a near-empty page with no visual element
        # carries nothing to summarize. Mark it non-content without an LLM call.
        targets: list[tuple[BuildPage, list[dict]]] = []
        prev_title = ""
        prev_titles: list[str] = []
        for r in rows_sorted:
            els = pages.get(r.page, [])
            if r.char_count < BLANK_CHAR_THRESHOLD and not has_visual_elements(els):
                r.is_content = False
                continue
            targets.append((r, els))
            prev_titles.append(prev_title)
            # Verbatim primary title threads continuation context forward,
            # independent of batching order.
            if r.primary_title:
                prev_title = r.primary_title

        n_updated = 0
        offset = 0
        for batch in chunk(targets, batch_size):
            inputs = [
                self._page_input(i, r, els, prev_titles[offset + i])
                for i, (r, els) in enumerate(batch)
            ]
            results = self._call_batch(bulletin, inputs, llm)
            for (r, _els), out in zip(batch, results):
                if self._apply(r, out):
                    n_updated += 1
            offset += len(batch)
        return n_updated

    # -- per-page prompt input -------------------------------------------------

    def _page_input(
        self, idx: int, row: BuildPage, elements: list[dict], prev_title: str,
    ) -> dict:
        blocks = [
            {
                "kind": b.kind,
                "title": b.title or "",
                "column_headers": list(b.column_headers),
                "row_labels": list(b.row_headers),
            }
            for b in row.content_blocks
        ]
        text = page_sanitized_text(elements)[:_PAGE_TEXT_CHARS]
        return {
            "id": idx,
            "n_figures": sum(1 for b in row.content_blocks if b.kind == "chart"),
            "prev_page_title": prev_title,
            "blocks": blocks,
            "text": text,
        }

    def _build_user(self, bulletin: str, inputs: list[dict]) -> str:
        return (
            f"bulletin (publication month, YYYY-MM): {bulletin}\n"
            f"corpus_hint: {_TREASURY_HINT}\n\n"
            f"pages (JSON, {len(inputs)} entries):\n"
            f"{json.dumps(inputs, ensure_ascii=False, indent=1)}\n\n"
            f"Return a JSON array of EXACTLY {len(inputs)} objects, one per input "
            f"page, in the order given."
        )

    def _call_batch(
        self, bulletin: str, inputs: list[dict], llm: LLMClient,
    ) -> list[dict | None]:
        """One batch → a list aligned to `inputs` (None where the model didn't
        return a usable object). Re-issues on page-count mismatch."""
        n = len(inputs)
        base_user = self._build_user(bulletin, inputs)
        for attempt in range(1, _PARITY_MAX_RETRIES + 1):
            user = base_user if attempt == 1 else (
                base_user
                + f"\n\nYour previous reply had the wrong number of objects. Return "
                  f"EXACTLY {n} objects, one per page, in order — nothing else."
            )
            resp = llm.call(
                system=_SYSTEM, user=user,
                temperature=0.0 if attempt == 1 else 0.4,
                call_site="summarize",
            )
            arr = self._parse_array(resp.text)
            if len(arr) == n:
                return arr
            log.warning(
                "summarize length mismatch (%s attempt %d/%d): got %d, expected %d",
                bulletin, attempt, _PARITY_MAX_RETRIES, len(arr), n,
            )
        # Exhausted retries — leave these pages verbatim (recall-safe).
        log.warning("summarize parity failed for %s — keeping %d pages verbatim",
                    bulletin, n)
        return [None] * n

    @staticmethod
    def _parse_array(text: str) -> list:
        obj = parse_json_response(text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            inner = next((v for v in obj.values() if isinstance(v, list)), None)
            if inner is not None:
                return inner
        return []

    # -- apply one page's output back onto the row -----------------------------

    def _apply(self, row: BuildPage, out: dict | None) -> bool:
        if not isinstance(out, dict):
            return False

        row.is_content = bool(out.get("is_content", True))
        row.date_interval = _norm_interval(out.get("date_interval"))
        kw = out.get("keywords")
        if isinstance(kw, list):
            row.keywords = [str(x).strip() for x in kw if str(x).strip()][:20]

        out_blocks = out.get("blocks")
        if isinstance(out_blocks, list) and len(out_blocks) == len(row.content_blocks):
            for b, ob in zip(row.content_blocks, out_blocks):
                if not isinstance(ob, dict):
                    continue
                t = ob.get("title")
                if isinstance(t, str) and t.strip():
                    b.title = t.strip()
                # Labels are typo-fixed IN PLACE: overwrite ONLY when the
                # length matches, else keep the verbatim list (parity guard).
                ch = ob.get("column_headers")
                if isinstance(ch, list) and len(ch) == len(b.column_headers):
                    b.column_headers = [str(x) for x in ch]
                rl = ob.get("row_labels")
                if isinstance(rl, list) and len(rl) == len(b.row_headers):
                    b.row_headers = [str(x) for x in rl]
                summ = ob.get("summary")
                if isinstance(summ, str) and summ.strip():
                    b.summary = summ.strip()
        return True
