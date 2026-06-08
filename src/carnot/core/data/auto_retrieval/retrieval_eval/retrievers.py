"""Retriever API: search(query, documents, k) -> list of (doc_id, score).

Caching: Dense/SPLADE/ColBERTv2 cache encodings by doc_id. Call
encode_corpus(full_corpus) once, then search() with any subset reuses
the cache — no re-encoding.  BM25 auto-rebuilds when the doc set changes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)
Result = Tuple[str, float]


class DenseRetriever:
    """OpenAI text-embedding-3-large. Caches embeddings by doc_id."""

    _RETRY_DELAY = 5
    _MAX_RETRIES = 10

    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        max_tokens: int = 8191,
        batch_size: int = 512,
    ):
        import openai
        import tiktoken

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            organization=os.environ.get("OPENAI_ORG_KEY", ""),
        )
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self._cache: Dict[str, np.ndarray] = {}

    def _truncate(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        return self.tokenizer.decode(tokens)

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        import openai as _openai

        truncated = [self._truncate(t) for t in texts]
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                resp = self.client.embeddings.create(input=truncated, model=self.model_name)
                vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs /= np.maximum(norms, 1e-9)
                return vecs
            except _openai.RateLimitError:
                logger.warning("Rate limit, retry %d/%d", attempt, self._MAX_RETRIES)
                time.sleep(self._RETRY_DELAY)
        raise RuntimeError("Exceeded OpenAI retry limit")

    def encode_corpus(self, documents: Dict[str, str]) -> None:
        """Cache embeddings for all doc_ids not yet cached. Token-aware batching."""
        uncached = [(did, documents[did]) for did in documents if did not in self._cache]
        if not uncached:
            return
        ids, texts = zip(*uncached)
        # Build token-aware batches (OpenAI max 300k tokens/request, stay under 250k)
        max_tokens_per_batch = 250_000
        batch_ids: list[str] = []
        batch_texts: list[str] = []
        batch_tokens = 0
        total_done = 0
        for did, text in zip(ids, texts):
            truncated = self._truncate(text)
            tok_count = len(self.tokenizer.encode(truncated))
            if batch_texts and batch_tokens + tok_count > max_tokens_per_batch:
                logger.info("DenseRetriever: embedding batch %d-%d / %d (%d tokens)", total_done, total_done + len(batch_texts), len(texts), batch_tokens)
                vecs = self._embed_batch(batch_texts)
                for bid, vec in zip(batch_ids, vecs):
                    self._cache[bid] = vec
                total_done += len(batch_texts)
                batch_ids, batch_texts, batch_tokens = [], [], 0
            batch_ids.append(did)
            batch_texts.append(text)
            batch_tokens += tok_count
        if batch_texts:
            logger.info("DenseRetriever: embedding batch %d-%d / %d (%d tokens)", total_done, total_done + len(batch_texts), len(texts), batch_tokens)
            vecs = self._embed_batch(batch_texts)
            for bid, vec in zip(batch_ids, vecs):
                self._cache[bid] = vec

    def search(self, query: str, documents: Dict[str, str], k: int = 20) -> List[Result]:
        if not documents:
            return []
        self.encode_corpus(documents)
        doc_ids = list(documents.keys())
        doc_matrix = np.stack([self._cache[did] for did in doc_ids])
        q_vec = self._embed_batch([query])[0]
        scores = doc_matrix @ q_vec
        k = min(k, len(doc_ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(doc_ids[i], float(scores[i])) for i in top_idx]


class SpladeRetriever:
    """SPLADE v2 (naver/splade-cocondenser-ensembledistil). Caches sparse vectors by doc_id.

    Encoding: log(1 + ReLU(logits)) -> max-pool -> quantize (matches Pyserini).
    """

    def __init__(
        self,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        device: str = "cpu",
        doc_max_length: int = 512,
        query_max_length: int = 256,
        batch_size: int = 32,
    ):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = device
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.doc_max_length = doc_max_length
        self.query_max_length = query_max_length
        self.batch_size = batch_size
        self._cache: Dict[str, Dict[int, float]] = {}

    def _encode_texts(self, texts: List[str], max_length: int) -> List[Dict[int, float]]:
        import torch

        inputs = self.tokenizer(
            texts, max_length=max_length, truncation=True, padding="longest",
            return_attention_mask=True, return_token_type_ids=False,
            return_tensors="pt", add_special_tokens=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).logits
            activated = torch.log(1 + torch.relu(logits)) * inputs["attention_mask"].unsqueeze(-1)
            agg, _ = torch.max(activated, dim=1)
            agg = agg.cpu().numpy()

        results: List[Dict[int, float]] = []
        for row in agg:
            nonzero = np.nonzero(row)[0]
            d: Dict[int, float] = {}
            for tok_id in nonzero:
                qw = round(float(row[tok_id]) / 5 * 256)
                if qw > 0:
                    d[int(tok_id)] = qw
            results.append(d)
        return results

    def encode_corpus(self, documents: Dict[str, str]) -> None:
        """Cache sparse representations for all doc_ids not yet cached."""
        uncached = [(did, documents[did]) for did in documents if did not in self._cache]
        if not uncached:
            return
        ids, texts = zip(*uncached)
        for start in range(0, len(texts), self.batch_size):
            batch_ids = ids[start : start + self.batch_size]
            batch_texts = texts[start : start + self.batch_size]
            logger.info("SpladeRetriever: encoding %d-%d / %d", start, start + len(batch_texts), len(texts))
            weights = self._encode_texts(list(batch_texts), self.doc_max_length)
            for did, w in zip(batch_ids, weights):
                self._cache[did] = w

    def search(self, query: str, documents: Dict[str, str], k: int = 20) -> List[Result]:
        if not documents:
            return []
        self.encode_corpus(documents)
        q_weights = self._encode_texts([query], self.query_max_length)[0]
        doc_ids = list(documents.keys())
        scores = np.zeros(len(doc_ids), dtype=np.float64)
        for i, did in enumerate(doc_ids):
            doc_w = self._cache[did]
            for tok_id, qw in q_weights.items():
                if tok_id in doc_w:
                    scores[i] += qw * doc_w[tok_id]
        k = min(k, len(doc_ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(doc_ids[i], float(scores[i])) for i in top_idx]


class BM25Retriever:
    """Okapi BM25 via rank_bm25. Auto-rebuilds index when doc set changes."""

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1 = k1
        self.b = b
        self._cached_doc_set: frozenset | None = None
        self._cached_doc_ids: List[str] = []
        self._bm25 = None

    def search(self, query: str, documents: Dict[str, str], k: int = 20) -> List[Result]:
        if not documents:
            return []
        from rank_bm25 import BM25Okapi

        doc_set = frozenset(documents.keys())
        if doc_set != self._cached_doc_set:
            self._cached_doc_ids = list(documents.keys())
            tokenized = [text.lower().split() for text in documents.values()]
            self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
            self._cached_doc_set = doc_set

        scores = self._bm25.get_scores(query.lower().split())
        k = min(k, len(self._cached_doc_ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._cached_doc_ids[i], float(scores[i])) for i in top_idx]


class HybridRetriever:
    """BM25 + Dense with score interpolation: alpha * sparse + dense.

    Normalization: mean-centered range normalization (matches Pyserini).
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever | None = None,
        alpha: float = 0.1,
        normalization: bool = False,
    ):
        self.dense = dense
        self.bm25 = bm25 or BM25Retriever()
        self.alpha = alpha
        self.normalization = normalization

    def encode_corpus(self, documents: Dict[str, str]) -> None:
        self.dense.encode_corpus(documents)

    def search(self, query: str, documents: Dict[str, str], k: int = 20) -> List[Result]:
        if not documents:
            return []
        k0 = min(max(k * 5, 100), len(documents))
        dense_hits = self.dense.search(query, documents, k0)
        sparse_hits = self.bm25.search(query, documents, k0)

        dense_scores = dict(dense_hits)
        sparse_scores = dict(sparse_hits)
        min_d = min(dense_scores.values()) if dense_scores else 0.0
        max_d = max(dense_scores.values()) if dense_scores else 1.0
        min_s = min(sparse_scores.values()) if sparse_scores else 0.0
        max_s = max(sparse_scores.values()) if sparse_scores else 1.0

        fused: List[Result] = []
        for doc in set(dense_scores) | set(sparse_scores):
            d = dense_scores.get(doc, min_d)
            s = sparse_scores.get(doc, min_s)
            if self.normalization:
                d_range = max_d - min_d
                s_range = max_s - min_s
                d = (d - (min_d + max_d) / 2) / d_range if d_range > 0 else 0.0
                s = (s - (min_s + max_s) / 2) / s_range if s_range > 0 else 0.0
            fused.append((doc, self.alpha * s + d))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:k]


class ColBERTZeroRetriever:
    """ColBERT-Zero (lightonai/ColBERT-Zero) via PyLate. Builds PLAID index, post-filters for subsets.

    Critical: uses prompt_name="query" / "document" for asymmetric encoding.
    pip install pylate faiss-cpu
    """

    def __init__(
        self,
        model_name: str = "lightonai/ColBERT-Zero",
        index_folder: str = "/tmp/colbert_zero_index",
        index_name: str = "eval_index",
        batch_size: int = 32,
    ):
        from pylate import models

        self.model = models.ColBERT(model_name_or_path=model_name)
        self.index_folder = index_folder
        self.index_name = index_name
        self.batch_size = batch_size
        self._indexed_ids: list[str] = []
        self._indexed_set: frozenset[str] | None = None
        self._retriever = None

    def encode_corpus(self, documents: Dict[str, str]) -> None:
        """Build PLAID index. Skips if already indexed with same doc set."""
        doc_set = frozenset(documents.keys())
        if doc_set == self._indexed_set:
            return

        from pylate import indexes, retrieve

        self._indexed_ids = list(documents.keys())
        texts = list(documents.values())

        logger.info("ColBERTZeroRetriever: encoding %d documents ...", len(texts))
        doc_embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            is_query=False,
            prompt_name="document",
            show_progress_bar=True,
        )

        os.makedirs(self.index_folder, exist_ok=True)
        index = indexes.PLAID(
            index_folder=self.index_folder,
            index_name=self.index_name,
            override=True,
        )
        index.add_documents(
            documents_ids=self._indexed_ids,
            documents_embeddings=doc_embeddings,
        )

        self._retriever = retrieve.ColBERT(index=index)
        self._indexed_set = doc_set
        logger.info("ColBERTZeroRetriever: indexed %d documents", len(self._indexed_ids))

    def search(self, query: str, documents: Dict[str, str], k: int = 20) -> List[Result]:
        if not documents:
            return []
        doc_set = set(documents.keys())
        if self._indexed_set is None or not doc_set.issubset(self._indexed_set):
            self.encode_corpus(documents)

        is_subset = len(doc_set) < len(self._indexed_set)
        k_search = min(len(self._indexed_ids), k * 3) if is_subset else k

        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            is_query=True,
            prompt_name="query",
        )

        results = self._retriever.retrieve(
            queries_embeddings=query_embedding,
            k=k_search,
        )

        # results is list-of-lists (one per query); each item is a dict with id + score
        hits = results[0] if results else []
        filtered = [
            (hit["id"], float(hit["score"]))
            for hit in hits
            if hit["id"] in doc_set
        ]
        return filtered[:k]
