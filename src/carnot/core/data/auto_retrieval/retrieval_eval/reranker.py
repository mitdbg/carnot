"""Cross-encoder reranker: rerank(query, results, documents, k) -> list of (doc_id, score)."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)
Result = Tuple[str, float]


class CrossEncoderReranker:
    """BAAI/bge-reranker-base cross-encoder. Scores (query, passage) pairs via logits."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        batch_size: int = 64,
        max_length: int = 512,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _score_pairs(self, pairs: List[List[str]]) -> np.ndarray:
        all_scores: List[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                return_tensors="pt", max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            all_scores.extend(logits.cpu().tolist())
        return np.array(all_scores, dtype=np.float32)

    def rerank(
        self,
        query: str,
        results: List[Result],
        documents: Dict[str, str],
        k: Optional[int] = None,
    ) -> List[Result]:
        """Re-score and re-order retrieval results using cross-encoder."""
        if not results:
            return []
        pairs = [[query, documents.get(did, "")] for did, _ in results]
        scores = self._score_pairs(pairs)
        reranked = [(did, float(s)) for (did, _), s in zip(results, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:k] if k is not None else reranked
