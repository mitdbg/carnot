"""Runtime helpers for the BM25 retrieve experiment.

Builds per-chapter `Bm25Index` instances from the catalog and reranks a
year-filtered candidate list. Shared by `RetrieveExecutor` (production
operator path) and `eval/eval_retrieve.py` (`--retriever
chapter-year-bm25`), so both stay in lockstep.

Scaffold — see the BM25 integration plan. Delete this file alongside
`bm25.py` to remove the experiment.
"""

from __future__ import annotations

from statistics import median
from typing import Any

from .bm25 import Bm25Index, tokenize
from .schema import PageCatalogRow


# Title repetition factor — titles are the highest-signal field on the
# page, and the bag-of-words doc would otherwise be dominated by long
# column-header lists. df is unaffected (set-cardinality, see Bm25Index).
_TITLE_REPEAT = 3


def page_doc_tokens(row: PageCatalogRow) -> list[str]:
    """Bag-of-words for one page. Titles 3x; headers + keywords 1x."""
    out: list[str] = []
    for block in row.content_blocks:
        if block.title:
            title_toks = tokenize(block.title)
            for _ in range(_TITLE_REPEAT):
                out.extend(title_toks)
        for header in block.column_headers:
            out.extend(tokenize(header))
        for header in block.row_headers_sample:
            out.extend(tokenize(header))
    for kw in row.keywords:
        out.extend(tokenize(kw))
    return out


def build_chapter_index(
    pages: list[dict[str, Any]],
    catalog_index: dict[tuple[str, int], PageCatalogRow],
) -> Bm25Index:
    """Build a BM25 index over the pages in one chapter. `pages` is the
    raw chapter page list (each entry has `bulletin` + `page` keys);
    `catalog_index` resolves to the actual `PageCatalogRow` for token
    extraction. Pages absent from the catalog (shouldn't happen) become
    empty docs and just score 0."""
    docs: list[tuple[tuple[str, int], list[str]]] = []
    for p in pages:
        key = (p["bulletin"], int(p["page"]))
        row = catalog_index.get(key)
        tokens = page_doc_tokens(row) if row is not None else []
        docs.append((key, tokens))
    return Bm25Index.build(docs)


def _bypass_decision(
    scores_desc: list[float], threshold: float, *, window: int = 20,
) -> tuple[bool, float, float]:
    """Confidence rule: bypass iff top1/median(top-N) >= threshold.

    Returns `(bypass, top1, median_window)`. If fewer than 2 scoring docs
    we return `bypass=False` to avoid acting on noise."""
    if len(scores_desc) < 2:
        return False, (scores_desc[0] if scores_desc else 0.0), 0.0
    top1 = scores_desc[0]
    window_scores = scores_desc[:max(2, min(window, len(scores_desc)))]
    med = median(window_scores)
    if med <= 0:
        return False, top1, med
    return (top1 / med) >= threshold, top1, med


def bm25_rerank(
    pages: list[dict[str, Any]],
    index: Bm25Index,
    *,
    question: str,
    key: str,
    top_k: int,
    threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rerank `pages` (the year-filtered survivors) by BM25 against
    `question + 2*key`. If the top score clearly dominates (`top1 /
    median20 >= threshold`), truncate to `top_k`; otherwise preserve all
    survivors in BM25 order. Pages with score 0 are appended after the
    scoring ones (preserving recall when bypass is off).

    Returns `(reranked_pages, meta)`. `meta` includes the decision and
    diagnostic numbers used by the trace.
    """
    if not pages:
        return [], {"enabled": True, "bypass": False, "n_before": 0,
                    "n_after": 0, "top1": 0.0, "median20": 0.0, "ratio": 0.0}

    # Build the query: question 1x, concept key 2x (planner's compressed
    # intent — short, so it'd wash out otherwise).
    query_tokens = tokenize(question) + tokenize(key) + tokenize(key)
    scores = index.score(query_tokens)

    # Preserve input order for zero-score pages; sort scoring pages desc.
    scored: list[tuple[float, dict[str, Any]]] = []
    unscored: list[dict[str, Any]] = []
    for p in pages:
        k = (p["bulletin"], int(p["page"]))
        s = scores.get(k, 0.0)
        if s > 0:
            scored.append((s, p))
        else:
            unscored.append(p)
    scored.sort(key=lambda sp: -sp[0])

    scores_desc = [s for s, _ in scored]
    bypass, top1, med = _bypass_decision(scores_desc, threshold)
    ratio = (top1 / med) if med > 0 else 0.0

    if bypass:
        reranked = [p for _, p in scored[:top_k]]
    else:
        reranked = [p for _, p in scored] + unscored

    meta = {
        "enabled": True,
        "bypass": bypass,
        "n_before": len(pages),
        "n_after": len(reranked),
        "n_scored": len(scored),
        "top1": round(top1, 4),
        "median20": round(med, 4),
        "ratio": round(ratio, 4),
        "top_k": top_k,
        "threshold": threshold,
    }
    return reranked, meta
