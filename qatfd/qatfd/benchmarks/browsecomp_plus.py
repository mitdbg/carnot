"""BrowseComp-Plus benchmark.

Questions: benchmarks/browsecomp-plus/browsecomp_plus_decrypted.jsonl
  (query_id / query / answer / gold_docs[...]).
Corpus + index: the prebuilt ChromaDB collection `browsecomp-plus-qwen-8b` under
  benchmarks/browsecomp-plus/chromadb (element-level, Qwen3-Embedding-8B). The doc_id->text
  map is reconstructed from the element-embedding metadata (cleaned text per element),
  avoiding a re-download of the HF corpus.
Scoring: LLM-as-judge, single-nugget correctness (see judge.py).
Dev/test split: benchmarks/browsecomp-plus/browsecomp_plus_splits.json (test = KARL's 230
  calibrated-subset query_ids; dev = 50 sampled seed 0 from the rest); see scripts/make_splits.py.
"""

from __future__ import annotations

import glob
import json

from jinja2 import Environment, StrictUndefined

from skunk.common import ExecutionContext

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_single_nugget
from qatfd.config import BrowseCompPlusConfig
from qatfd.constants import BROWSECOMP_PLUS
from qatfd.paths import resolve_under_benchmarks
from qatfd.prompts import load_qatfd_prompts
from qatfd.types import Question

_ENV = Environment(
    autoescape=False, keep_trailing_newline=True, undefined=StrictUndefined
)
_PROMPTS = load_qatfd_prompts("browsecomp_plus")


class BrowseCompPlusBenchmark(Benchmark):
    name = BROWSECOMP_PLUS
    config: BrowseCompPlusConfig
    corpus_details = _ENV.from_string(_PROMPTS["corpus_details"]).render()

    def __init__(self, config: BrowseCompPlusConfig) -> None:
        # all benchmark data (questions, metadata, prompts) resolves under qatfd/benchmarks/.
        config.bcp_questions = str(resolve_under_benchmarks(config.bcp_questions))
        config.bcp_metadata_glob = str(resolve_under_benchmarks(config.bcp_metadata_glob))
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
            answer_format_hint=self.answer_format_hint,
            compute_objective=self.compute_objective,
            corpus_details=self.corpus_details,
        )

    async def score(self, question: Question, predicted: str, ctx: ExecutionContext) -> dict:
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
