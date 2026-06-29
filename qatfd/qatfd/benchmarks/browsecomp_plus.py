"""BrowseComp-Plus benchmark.

Questions: skunk/browsecomp-plus/browsecomp_plus_decrypted.jsonl
  (query_id / query / answer / gold_docs[...]).
Corpus + index: the prebuilt ChromaDB collection `qwen-browsecomp-plus` under
  skunk/.chromadb (element-level, Qwen3-Embedding-8B). The doc_id->text map is
  reconstructed from the element-embedding metadata (cleaned text per element),
  avoiding a re-download of the HF corpus.
Scoring: LLM-as-judge, single-nugget correctness (see judge.py).
Held-out test set: KARL's 230 calibrated-subset query_ids (data/karl_bcp_test_ids.json).
"""

from __future__ import annotations

import glob
import json
import os

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_single_nugget
from qatfd.config import BrowseCompPlusConfig
from qatfd.constants import BROWSECOMP_PLUS
from qatfd.paths import resolve_under_skunk
from qatfd.types import Question


class BrowseCompPlusBenchmark(Benchmark):
    name = BROWSECOMP_PLUS
    config: BrowseCompPlusConfig

    def __init__(self, config: BrowseCompPlusConfig) -> None:
        # resolve paths and store config
        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.bcp_questions = str(resolve_under_skunk(config.bcp_questions))
        config.bcp_metadata_glob = str(resolve_under_skunk(config.bcp_metadata_glob))
        config.bcp_test_ids_path = str(resolve_under_skunk(config.bcp_test_ids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
        super().__init__(config)

    def load_questions(self) -> list[Question]:
        questions: list[Question] = []
        with open(self.config.bcp_questions) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                gold_docs = [str(d.get("docid")) for d in rec.get("gold_docs", []) if d.get("docid") is not None]
                # Evidence docs (superset of gold; the docs needed to answer, whether
                # or not they state the final answer) — kept for evidence-level recall.
                evidence_docs = [str(d.get("docid")) for d in rec.get("evidence_docs", []) if d.get("docid") is not None]
                questions.append(
                    Question(
                        qid=str(rec["query_id"]),
                        text=str(rec["query"]),
                        gold=str(rec.get("answer", "")),
                        gold_docs=gold_docs,
                        meta={"n_gold_docs": len(gold_docs), "evidence_docs": evidence_docs},
                    )
                )
        return questions

    def _build_document_map(self) -> dict[str, str]:
        """doc_id -> full document text, reconstructed by concatenating each document's
        element `cleaned` text (ordered by element_id) from the embedding metadata. This
        is the same text the vectors were built from, so it matches what search returns."""
        paths = sorted(glob.glob(self.config.bcp_metadata_glob))
        if not paths:
            raise FileNotFoundError(
                f"no BrowseComp-Plus element metadata found at {self.config.bcp_metadata_glob}; "
                f"expected metadata_rank*.json alongside the element embeddings."
            )
        # docid -> {element_id: cleaned}
        by_doc: dict[str, dict[int, str]] = {}
        for p in paths:
            with open(p) as f:
                meta = json.load(f)
            for entry in meta.values():
                docid = str(entry["docid"])
                eid = int(entry.get("element_id", 0))
                by_doc.setdefault(docid, {})[eid] = entry.get("cleaned", "")
        return {
            docid: "\n\n".join(elements[k] for k in sorted(elements))
            for docid, elements in by_doc.items()
        }

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_chroma_collection()
        document_map = self._build_document_map()
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=document_map,
            config=self.config,
        )

    async def score(self, question: Question, predicted: str, ctx) -> dict:
        return await judge_single_nugget(
            ctx, question=question.text, gold=question.gold, predicted=predicted, model=self.config.judge_model
        )

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        # BrowseComp-Plus relevance is labeled at the DOCUMENT level (gold ⊆ evidence);
        # there are no chunk-level relevance labels, so document-level recall — over
        # gold docs and over the broader evidence set — is the meaningful metric, and
        # mirrors what upstream reports. Retrieved chunks are collapsed to doc_ids by
        # the systems, so `retrieved` here is already a doc_id list.
        return {
            "gold_doc_recall": doc_recall(retrieved, question.gold_docs),
            "evidence_doc_recall": doc_recall(retrieved, question.meta.get("evidence_docs", [])),
        }

    def test_qids(self) -> set[str]:
        assert os.path.exists(self.config.bcp_test_ids_path), f"test_qids_path {self.config.bcp_test_ids_path} does not exist; expected a JSON file containing the test query_ids (e.g. KARL's calibrated subset) for BrowseComp-Plus."
        with open(self.config.bcp_test_ids_path) as f:
            data = json.load(f)
        
        assert "query_ids" in data, f"test_qids_path {self.config.bcp_test_ids_path} should contain a JSON object with a 'query_ids' field listing the qids in the test set."
        return {str(q) for q in data["query_ids"]}
