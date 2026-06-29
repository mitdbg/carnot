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

import json
import os
from collections import OrderedDict

import chromadb

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


class MergedCollection:
    """Duck-typed stand-in for a chromadb ``Collection`` that fans a query/get out across N shard
    collections and merges the results, so the SearchAgent and the doc map can treat a sharded
    corpus as one collection. BioGen's 26.8M abstracts are split into one collection per embedding
    rank (``f"{base}_r{i}"``) because chromadb 1.5.x's metadata-segment compaction fails on a single
    collection that large; each ~6.7M-row shard compacts fine.

    Only the methods skunk's retrieval actually calls are implemented — ``query`` (search_corpus),
    ``get`` (grep_corpus + the doc map), and ``count`` (verification). A PMID lives in exactly one
    shard (a doc's chunks all come from one source file -> one rank), so ``get`` by id/where simply
    concatenates and the lone owning shard supplies the hit."""

    def __init__(self, collections: list) -> None:
        assert collections, "MergedCollection needs at least one shard collection"
        self._collections = collections

    @property
    def name(self) -> str:
        return self._collections[0].name

    def count(self) -> int:
        return sum(c.count() for c in self._collections)

    def query(self, **kwargs) -> dict:
        """Run the same query on every shard, then keep the globally closest ``n_results`` by
        distance (chroma's default L2 space: smaller = closer)."""
        n_results = kwargs.get("n_results", 10)
        include = kwargs.get("include") or ["metadatas", "documents", "distances"]
        rows: list[tuple] = []
        for c in self._collections:
            r = c.query(**kwargs)
            ids = (r.get("ids") or [[]])[0]
            n = len(ids)
            dists = (r.get("distances") or [[None] * n])[0]
            docs = (r.get("documents") or [[None] * n])[0]
            metas = (r.get("metadatas") or [[None] * n])[0]
            for i in range(n):
                rows.append((dists[i], ids[i], docs[i], metas[i]))
        # None distances (shouldn't happen when "distances" is included) sort last, deterministically.
        rows.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0.0))
        rows = rows[:n_results]
        out: dict = {"ids": [[t[1] for t in rows]]}
        if "distances" in include:
            out["distances"] = [[t[0] for t in rows]]
        if "documents" in include:
            out["documents"] = [[t[2] for t in rows]]
        if "metadatas" in include:
            out["metadatas"] = [[t[3] for t in rows]]
        return out

    def get(self, **kwargs) -> dict:
        """Concatenate ``get`` across shards (each PMID is in one shard). Honors ``limit`` as a
        global cap, short-circuiting once enough rows are collected."""
        limit = kwargs.get("limit")
        include = kwargs.get("include") or []
        ids_acc: list = []
        docs_acc: list = []
        metas_acc: list = []
        want_docs = want_metas = False
        for c in self._collections:
            r = c.get(**kwargs)
            ids_acc.extend(r.get("ids") or [])
            if r.get("documents") is not None:
                want_docs = True
                docs_acc.extend(r["documents"])
            if r.get("metadatas") is not None:
                want_metas = True
                metas_acc.extend(r["metadatas"])
            if limit is not None and len(ids_acc) >= limit:
                break
        if limit is not None:
            ids_acc, docs_acc, metas_acc = ids_acc[:limit], docs_acc[:limit], metas_acc[:limit]
        out: dict = {"ids": ids_acc}
        if want_docs or "documents" in include:
            out["documents"] = docs_acc
        if want_metas or "metadatas" in include:
            out["metadatas"] = metas_acc
        return out


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
        if getattr(config, "chromadb_shard_dirs", None):
            config.chromadb_shard_dirs = [
                d if os.path.isabs(d) else str(resolve_under_skunk(d)) for d in config.chromadb_shard_dirs
            ]
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

    def _open_collection(self):
        """Single collection, or a MergedCollection over `chromadb_num_shards` per-rank shards
        (`f"{chromadb_collection}_r{i}"`). Each shard lives in its OWN chroma dir — its own
        chroma.sqlite3 — so no single metadata segment reaches the scale where chromadb 1.5.x
        compaction fails. Embedded mode: shard i is at `chromadb_shard_dirs[i]` if given, else
        `{chromadb_dir}/r{i}`. Server mode: all shards are opened by name from the one server."""
        n = getattr(self.config, "chromadb_num_shards", 1) or 1
        if n <= 1:
            return self._open_chroma_collection()
        base = self.config.chromadb_collection
        if self.config.chromadb_host:
            client = self._chroma_client()
            return MergedCollection([self._get_shard(client, f"{base}_r{i}", "server") for i in range(n)])
        shard_dirs = getattr(self.config, "chromadb_shard_dirs", None)
        if shard_dirs and len(shard_dirs) != n:
            raise ValueError(f"chromadb_shard_dirs has {len(shard_dirs)} entries but chromadb_num_shards={n}.")
        clients: dict[str, object] = {}
        shards = []
        for i in range(n):
            d = shard_dirs[i] if shard_dirs else os.path.join(self.config.chromadb_dir, f"r{i}")
            if d not in clients:
                if not os.path.exists(d):
                    raise FileNotFoundError(
                        f"biogen shard dir {d!r} does not exist; expected {n} per-shard chroma dirs "
                        f"(default {self.config.chromadb_dir}/r0..r{n - 1}, or set benchmarks.chromadb_shard_dirs)."
                    )
                clients[d] = chromadb.PersistentClient(path=d)
            shards.append(self._get_shard(clients[d], f"{base}_r{i}", d))
        return MergedCollection(shards)

    def _get_shard(self, client, name: str, where: str):
        try:
            return client.get_collection(name=name)
        except Exception as e:
            raise RuntimeError(f"biogen shard collection {name!r} not found ({where}).") from e

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_collection()
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
