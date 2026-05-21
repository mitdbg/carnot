from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from .config import PipelineConfig
from .corpus import SearchResult
from .planner import SubQuery, decompose_question, is_hard_question, retry_hints


class StrongRetriever:
    def __init__(self, base_retriever, config: PipelineConfig):
        self.base = base_retriever
        self.config = config
        self.rrf_k = 60

    def plan(self, question: str):
        return self.base.plan(question)

    def search(self, question: str, k: int = 100) -> List[SearchResult]:
        plan = self.base.plan(question)
        hard = is_hard_question(question, plan)

        if hard and self.config.decompose_hard:
            subs = decompose_question(question, self.config)
            merged = self._search_subqueries(subs, k=max(k, self.config.lateon_candidate_k))
            if hard and self.config.retry_hard and merged:
                retry_query = retry_hints(question, subs[0].query if subs else question, self.config)
                if retry_query:
                    retry_results = self.base.search(retry_query, k=max(k, self.config.lateon_candidate_k))
                    merged = self._rrf_merge([merged, retry_results])
            return self._finalize(merged, k)

        results = self.base.search(question, k=k)
        if hard and self.config.retry_hard:
            retry_query = retry_hints(question, question, self.config)
            if retry_query:
                retry_results = self.base.search(retry_query, k=max(k, self.config.lateon_candidate_k))
                results = self._rrf_merge([results, retry_results])
        return self._finalize(results, k)

    def _search_subqueries(self, subs: List[SubQuery], k: int) -> List[SearchResult]:
        if len(subs) <= 1:
            query = subs[0].query if subs else ""
            return self.base.search(query, k=k) if query else []

        workers = min(self.config.parallel_llm, len(subs))

        def _run(sub: SubQuery) -> List[SearchResult]:
            return self.base.search(sub.query, k=k)

        batches: List[List[SearchResult]] = []
        if workers <= 1:
            for sub in subs:
                batches.append(_run(sub))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run, sub): sub for sub in subs}
                for future in as_completed(futures):
                    batches.append(future.result())
        return self._rrf_merge(batches)

    def _rrf_merge(self, batches: List[List[SearchResult]]) -> List[SearchResult]:
        scores: Dict[Tuple[str, int], float] = defaultdict(float)
        best: Dict[Tuple[str, int], SearchResult] = {}
        for batch in batches:
            for rank, result in enumerate(batch, start=1):
                page_id = result.record.page_id
                if page_id is None:
                    continue
                key = (result.record.source_file, page_id)
                scores[key] += 1.0 / (self.rrf_k + rank)
                if key not in best or result.score > best[key].score:
                    best[key] = result
        ranked = sorted(scores, key=lambda key: scores[key], reverse=True)
        out = []
        for key in ranked:
            result = best[key]
            result.score = scores[key]
            result.channel = result.channel + "+strong" if result.channel else "strong"
            out.append(result)
        return out

    def _finalize(self, results: List[SearchResult], k: int) -> List[SearchResult]:
        for index, result in enumerate(results[:k], start=1):
            result.rank = index
        return results[:k]

    def close(self) -> None:
        closer = getattr(self.base, "close", None)
        if callable(closer):
            closer()
