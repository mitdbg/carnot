"""Stdlib BM25 index for in-memory page reranking.

Scaffold for the BM25 retrieve experiment (see plan: integrate BM25
into page_index retrieval). Default OFF in `SkunkConfig.bm25_enabled`;
the operator path only touches this module when the flag is set.

Self-contained — no third-party deps, matches the dependency story of
the shipped `data/page_index/retrieve.py`. Safe to delete this whole
file along with `bm25_runtime.py` if the experiment fails its eval
gates.

Key implementation notes:
  - `df` is counted per UNIQUE token per doc (set-cardinality), so
    title repetition for in-doc weighting does not inflate IDF.
  - Standard Okapi BM25 with k1=1.5, b=0.75.
  - Postings are built eagerly at build time; scoring walks only docs
    that contain at least one query token.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


# Small frozen English stopword set. Intentionally minimal — overzealous
# stopword removal kills discrimination on table titles (e.g. "Statement
# of the Public Debt").
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "will", "with",
})


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"^\d{4}$")


def tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric split, drop stopwords / 4-digit years /
    length-1 tokens. No stemming."""
    if not text:
        return []
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 2:
            continue
        if tok in _STOPWORDS:
            continue
        if _YEAR_RE.match(tok):
            continue
        out.append(tok)
    return out


@dataclass
class Bm25Index:
    """In-memory BM25 over a fixed doc set. doc_id is opaque (caller picks)."""
    doc_ids: list[Any]
    tf: list[Counter]                       # per-doc term frequencies
    dl: list[int]                           # per-doc length (sum of tf)
    df: dict[str, int]                      # global doc frequency
    postings: dict[str, list[int]]          # token -> doc-index list
    N: int
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(
        cls, docs: Iterable[tuple[Any, list[str]]],
        *, k1: float = 1.5, b: float = 0.75,
    ) -> "Bm25Index":
        """Build from `(doc_id, tokens)` pairs.

        `tokens` may contain repetitions (e.g. title tokens repeated 3x
        for in-doc weighting). `df` only counts each token once per doc.
        """
        doc_ids: list[Any] = []
        tf_list: list[Counter] = []
        dl: list[int] = []
        df: dict[str, int] = {}
        postings: dict[str, list[int]] = {}
        for doc_id, tokens in docs:
            idx = len(doc_ids)
            doc_ids.append(doc_id)
            tf = Counter(tokens)
            tf_list.append(tf)
            dl.append(sum(tf.values()))
            for tok in tf:                  # unique tokens only — df guard
                df[tok] = df.get(tok, 0) + 1
                postings.setdefault(tok, []).append(idx)
        N = len(doc_ids)
        avgdl = (sum(dl) / N) if N else 0.0
        return cls(doc_ids=doc_ids, tf=tf_list, dl=dl, df=df,
                   postings=postings, N=N, avgdl=avgdl, k1=k1, b=b)

    def _idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        # Robertson/Sparck-Jones IDF with the +1 smoothing variant —
        # always non-negative, even for tokens in >half of docs.
        return math.log(((self.N - df + 0.5) / (df + 0.5)) + 1.0)

    def score(self, query_tokens: list[str]) -> dict[Any, float]:
        """BM25 score per doc that contains ≥1 query token. Docs absent
        from the result dict implicitly score 0.0."""
        if not query_tokens or self.N == 0 or self.avgdl == 0:
            return {}
        # Dedup query tokens — repeated query terms shouldn't double-count
        # IDF; in-doc frequency saturation already lives in the tf term.
        q_unique = set(query_tokens)
        scores: dict[int, float] = {}
        for tok in q_unique:
            postings = self.postings.get(tok)
            if not postings:
                continue
            idf = self._idf(tok)
            for doc_idx in postings:
                f = self.tf[doc_idx].get(tok, 0)
                if f == 0:
                    continue
                norm = 1.0 - self.b + self.b * (self.dl[doc_idx] / self.avgdl)
                denom = f + self.k1 * norm
                contrib = idf * (f * (self.k1 + 1.0)) / denom
                scores[doc_idx] = scores.get(doc_idx, 0.0) + contrib
        return {self.doc_ids[i]: s for i, s in scores.items()}

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[Any, float]]:
        """Convenience: return at most k highest-scoring (doc_id, score)
        pairs, sorted descending."""
        scores = self.score(query_tokens)
        items = sorted(scores.items(), key=lambda kv: -kv[1])
        return items[:k]
