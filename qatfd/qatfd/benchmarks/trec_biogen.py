"""TREC-BioGen benchmark (2025 edition, Task A).

Questions: the 2025 BioGen Task A JSON (`2025_task_a.json`) — 40 biomedical questions, each
  with an expert reference answer split into sentences, each sentence carrying its cited PMIDs
  (`existing_supported_citations`). Task A's expert answers are the BioGen-2024 answers
  re-released against the 2025 corpus.
Gold nuggets: the organizer/BioACE `baseline_labels.json` (decompositional facts per answer
  sentence), flattened per question (~24.6 nuggets/q).
Corpus + index: the 2025 BioGen document collection — 26,805,982 PubMed abstracts, embedded
  with Qwen3-Embedding-0.6B into a Chroma collection; the doc_id (PMID) -> abstract map is
  rebuilt from the embedding metadata.
Scoring: nugget-completion recall via KARL's D.1 completeness judge (see judge.py). NOTE the
  official nuggets are finer-grained than KARL's consolidated set, so the absolute number is
  not directly comparable to KARL's reported 85.0 (see TrecBiogenConfig.nuggets_path).
Held-out test set: `test_ids_path` if given, else ALL 40 questions (run with --split test).
"""

from __future__ import annotations

import chromadb
import json
import os
from collections import OrderedDict

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.config import TrecBiogenConfig
from qatfd.constants import TREC_BIOGEN
from qatfd.paths import resolve_under_skunk
from qatfd.types import Question


def _qa_id(rec: dict) -> str:
    """qa_id from either the labels (`meta_data`) or submission (`metadata`) record shape."""
    meta = rec.get("meta_data") or rec.get("metadata") or {}
    return str(meta["qa_id"])


class _ChromaDocMap:
    """Lazy `doc_id -> abstract text`, served from the chroma `documents` column so eval never
    materializes all 26.8M abstracts in RAM (a plain dict would be ~60 GB). BioGen stores one
    chunk per doc with row id ``f"{pmid}_0"`` and metadata ``doc_id = pmid``; the systems only do
    keyed lookups (``.get`` / ``[]`` / ``in``) and never iterate, so this stands in for the dict.
    A small LRU keeps recently-read docs hot within a question."""

    def __init__(self, collection, cache_size: int = 4096) -> None:
        self._collection = collection
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cap = cache_size

    def _fetch(self, doc_id: str) -> str | None:
        # one chunk per doc: row id is f"{doc_id}_0"; fall back to a doc_id filter if absent.
        got = self._collection.get(ids=[f"{doc_id}_0"], include=["documents"])
        docs = got.get("documents") or []
        if docs and docs[0]:
            return docs[0]
        got = self._collection.get(where={"doc_id": doc_id}, include=["documents"])
        docs = [d for d in (got.get("documents") or []) if d]
        return "\n\n".join(docs) if docs else None

    def get(self, doc_id, default=None):
        key = str(doc_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        text = self._fetch(key)
        if text is None:
            return default
        self._cache[key] = text
        if len(self._cache) > self._cap:
            self._cache.popitem(last=False)
        return text

    def __getitem__(self, doc_id):
        text = self.get(doc_id)
        if text is None:
            raise KeyError(doc_id)
        return text

    def __contains__(self, doc_id) -> bool:
        return self.get(doc_id) is not None


class TrecBiogenBenchmark(Benchmark):
    name = TREC_BIOGEN
    config: TrecBiogenConfig

    # BioGen answers are multi-sentence biomedical reports; nugget recall rewards covering as
    # many distinct factual assertions as the evidence supports, so ask for breadth (not the
    # terse single-value form OfficeQA wants). Citations are not needed for the recall score.
    answer_format_hint = (
        "Write a thorough, well-organized answer that covers every distinct, well-supported "
        "factual assertion relevant to the question — group related points and state each claim "
        "explicitly. Prefer completeness over brevity; do not omit a supported fact for concision."
    )

    def __init__(self, config: TrecBiogenConfig) -> None:
        config.chromadb_dir = str(resolve_under_skunk(config.chromadb_dir))
        config.task_a_path = str(resolve_under_skunk(config.task_a_path))
        config.biogen_metadata_glob = str(resolve_under_skunk(config.biogen_metadata_glob))
        config.nuggets_path = str(resolve_under_skunk(config.nuggets_path))
        if config.test_ids_path:
            config.test_ids_path = str(resolve_under_skunk(config.test_ids_path))
        if config.prompts_path:
            config.prompts_path = str(resolve_under_skunk(config.prompts_path))
        super().__init__(config)

    # ---- questions + nuggets --------------------------------------------------

    def _load_nuggets(self) -> dict[str, list[str]]:
        """qid -> gold nuggets. Accepts the `baseline_labels.json` shape (a list of records,
        each `answer[].nuggets` flattened per question) or a plain {qid: [nugget, ...]} dict."""
        path = self.config.nuggets_path
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): [str(n) for n in v] for k, v in data.items()}
        nuggets: dict[str, list[str]] = {}
        for rec in data:
            flat = [str(n) for s in rec.get("answer", []) for n in (s.get("nuggets") or [])]
            nuggets[_qa_id(rec)] = flat
        return nuggets

    def load_questions(self) -> list[Question]:
        with open(self.config.task_a_path) as f:
            records = json.load(f)
        nuggets_by_qid = self._load_nuggets()

        questions: list[Question] = []
        for rec in records:
            qid = _qa_id(rec)
            sentences = rec.get("answer", [])
            ref_answer = " ".join(str(s.get("text", "")).strip() for s in sentences).strip()
            gold_docs = sorted(
                {str(c) for s in sentences for c in (s.get("existing_supported_citations") or [])}
            )
            # nuggets: overlay file if present for this qid, else one nugget per sentence.
            nuggets = nuggets_by_qid.get(qid) or [
                str(s.get("text", "")).strip() for s in sentences if str(s.get("text", "")).strip()
            ]
            questions.append(
                Question(
                    qid=qid,
                    text=str((rec.get("meta_data") or rec.get("metadata"))["question"]),
                    gold=ref_answer,
                    gold_docs=gold_docs,
                    meta={"nuggets": nuggets, "n_nuggets": len(nuggets)},
                )
            )
        return questions

    # ---- retrieval substrate --------------------------------------------------

    def _build_resources(self) -> BenchmarkResources:
        if not os.path.exists(self.config.chromadb_dir):
            raise FileNotFoundError(f"chromadb_dir {self.config.chromadb_dir} does not exist")

        client = chromadb.PersistentClient(path=self.config.chromadb_dir)
        try:
            collection = client.get_collection(name=self.config.chromadb_collection)
        except Exception as e:
            raise RuntimeError(
                f"chroma collection {self.config.chromadb_collection!r} not found under {self.config.chromadb_dir}."
            ) from e

        # 26.8M abstracts is far too much to hold in a {pmid: text} dict (~60 GB RAM). The text is
        # already in the chroma `documents` column, so serve it lazily + cached — only the docs a
        # question actually reads get materialized. (The systems only do keyed lookups on the doc
        # map; BioGen never iterates it, so the lazy mapping is a drop-in for the dict.)
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=_ChromaDocMap(collection),
            config=self.config,
        )

    # ---- scoring + metrics ----------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx) -> dict:
        return await judge_nugget_recall(
            ctx,
            question=question.text,
            nuggets=question.meta.get("nuggets", []),
            predicted=predicted,
            model=self.config.judge_model,
            judge_system="You are a careful biomedical answer-evaluation judge.",
            partial_credit=self.config.partial_credit,
        )

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        # Relevance is labeled at the PMID (document) level — the expert-cited supporting
        # PMIDs — and abstracts are one chunk per doc, so document-level recall is the metric.
        return {"doc_recall": doc_recall(retrieved, question.gold_docs)}

    def test_qids(self) -> set[str]:
        if self.config.test_ids_path:
            assert os.path.exists(self.config.test_ids_path), (
                f"test_ids_path {self.config.test_ids_path} does not exist; expected a JSON object "
                f"with a 'query_ids' field listing the held-out TREC-BioGen qids."
            )
            with open(self.config.test_ids_path) as f:
                data = json.load(f)
            assert "query_ids" in data, f"test_ids_path {self.config.test_ids_path} must have a 'query_ids' field."
            return {str(q) for q in data["query_ids"]}

        # no explicit split: the whole 40-question set is the held-out comparison set.
        with open(self.config.task_a_path) as f:
            return {_qa_id(rec) for rec in json.load(f)}
