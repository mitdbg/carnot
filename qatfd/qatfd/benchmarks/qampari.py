"""QAMPARI benchmark (Tel Aviv University; the KARLBench sub-benchmark).

Questions: the QAMPARI test JSONL (`test_data.jsonl`) — 1000 open-domain questions over Wikipedia,
  each with MANY answers (>=5, ~14.7 on average). Per record: `qid`, `question_text`, and
  `answer_list[]`, where every answer carries `answer_text`, `aliases`, and `proof[]` (each proof
  names the supporting Wikipedia article via `found_in_url`). QAMPARI tests EXHAUSTIVE entity search
  — success depends on comprehensive recall, not on finding a single supporting mention.
Gold nuggets: the gold answer entities themselves — one nugget per `answer_text`. The score is
  nugget-completion recall (KARL "treats each entity as a separate nugget"), graded by the LLM judge
  in judge.py so an alias / paraphrase of the entity still counts.
Corpus + index: the FULL QAMPARI chunked Wikipedia (~25.9M ~100-word passages), embedded with
  Qwen3-Embedding-0.6B into a Chroma collection (see compute_qampari_embeddings.py). We index the
  whole corpus — KARL's "chunks containing a gold answer entity" subset can't be reliably
  reproduced (the answer entities are common strings that match most of Wikipedia, and the
  entity-LINK annotations that would scope it aren't in the chunk metadata), so — like TREC-BioGen
  — we embed everything and let retrieval do the work. The same index serves both the dev and test
  question splits. The retrieval unit (`doc_id`) is the Wikipedia ARTICLE (its numeric `page_id`;
  see CORPUS_MODEL.md — before 2026-07-09 it was the ~100-word chunk itself; existing collections
  are migrated in place by scripts/migrate_doc_ids.py); the Chroma row id stays the released chunk
  id ("{page_id}__{n}"). The article-text doc map is assembled lazily from the Chroma `documents`
  column, all of an article's chunks joined in `element_id` order (a materialized map of 25.9M
  chunks would be far too large for RAM), keyed by page_id.
Gold docs / recall: a proof identifies the supporting Wikipedia ARTICLE, and the released chunk
  `url` is a curid URL while proofs use title-slug URLs — so the gold join key is the normalized
  Wikipedia TITLE. gold_docs are normalized article titles; recall = fraction of gold articles
  retrieved (`doc_recall`), mapping each retrieved page_id -> title via the title in any of its
  chunks' Chroma metadata (small per-question lookups, cached).
Dev/test splits: TEST is all 1000 `test_data.jsonl` questions — KARL's exact eval set, confirmed by
  its appendix query "What did James B. Longacre design?" appearing only there. DEV is 50 questions
  sampled (seed 0) from `train_data.jsonl` (the `dev_questions_path` extract), disjoint from test so
  it never leaks. Both are loaded together (load_questions) and served by the one shared Wikipedia
  index; the runner selects a split via benchmarks/qampari/qampari_splits.json (scripts/make_splits.py) — dev = the 50
  train_data samples, test = all 1000 test_data questions.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from urllib.parse import unquote, urlparse

from skunk.common import ExecutionContext

from qatfd.benchmarks.base import Benchmark, BenchmarkResources, doc_recall
from qatfd.benchmarks.judge import judge_nugget_recall
from qatfd.config import QampariConfig
from qatfd.constants import QAMPARI
from qatfd.paths import resolve_under_benchmarks
from qatfd.types import Question

# Grader persona for the nugget judge: QAMPARI answers are encyclopedic entities (people, films,
# places), so a neutral evaluator — not TREC-BioGen's biomedical one — judges entity presence.
_JUDGE_SYSTEM = (
    "You are a careful answer-evaluation judge for open-domain, multi-answer questions over "
    "encyclopedic (Wikipedia) text. Each decompositional fact is one gold answer entity; judge "
    "whether the predicted answer names that entity (an alias or alternate surface form counts)."
)


def _norm_title(title: object) -> str:
    """Canonical form of a Wikipedia article title: underscores -> spaces, whitespace collapsed,
    lowercased. The single join key between a proof's article and a corpus chunk's article. Accepts
    `object` because chroma types metadata values as a broad union; the `str(title)` below coerces."""
    return " ".join(str(title).replace("_", " ").split()).lower()


def _article_from_url(url: str) -> str:
    """Normalized article identity for a proof's `found_in_url`. Proofs use title-slug URLs
    (".../wiki/The_Helen_Morgan_Story"), so take the slug after `/wiki/`, url-decode it, and
    normalize. Falls back to the raw (normalized) URL for the rare non-`/wiki/` form."""
    path = urlparse(str(url)).path
    marker = "/wiki/"
    if marker in path:
        return _norm_title(unquote(path.split(marker, 1)[1]))
    return _norm_title(unquote(url))


class _ChromaDocMap:
    """Lazy `page_id -> article text`, assembled on demand from the Chroma collection: every chunk
    whose `doc_id` metadata is the article's page_id, ordered by `element_id` (the chunk's `__n`
    ordinal within the article) and joined. Serving articles lazily keeps eval RAM flat (25.9M
    chunks; a materialized article map would be tens of GB). The systems only do keyed lookups
    (`.get` / `[]` / `in`), never iterate — so this stands in for the dict. A small LRU keeps
    recently-read articles hot. Mirrors TREC-BioGen's lazy doc map."""

    def __init__(self, collection, cache_size: int = 1024) -> None:
        self._collection = collection
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cap = cache_size

    def get(self, doc_id, default=None):
        key = str(doc_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        got = self._collection.get(where={"doc_id": key}, include=["documents", "metadatas"])
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        if not docs:
            return default
        ordered = sorted(zip((int((m or {}).get("element_id", 0)) for m in metas), docs))
        text = "\n\n".join(d for _, d in ordered if d)
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


class QampariBenchmark(Benchmark):
    name = QAMPARI
    config: QampariConfig

    # QAMPARI questions have many correct answers; the metric is entity recall, so the answer must
    # be an exhaustive list. Push for breadth/recall over a single best answer.
    answer_format_hint = (
        "This question has MANY correct answers. List every entity that answers it that the "
        "documents support — as many as you can find — as a comma-separated list of entity names. "
        "Do not stop at one answer; prefer completeness (recall) over brevity, and include an "
        "entity whenever the documents support it."
    )
    compute_objective = (
        "the final answer to this question will be graded on nugget recall — the fraction of the "
        "reference answer's key facts that the final answer covers."
    )

    def __init__(self, config: QampariConfig) -> None:
        # all benchmark data (questions, dev extract, prompts) resolves under qatfd/benchmarks/.
        config.questions_path = str(resolve_under_benchmarks(config.questions_path))
        if config.dev_questions_path:
            config.dev_questions_path = str(resolve_under_benchmarks(config.dev_questions_path))
        # the chroma collection, set in _build_resources; recall_metrics reads the `title` metadata
        # of the retrieved articles' chunks from it to map page_ids to gold article titles.
        self._collection = None
        self._title_cache: dict[str, str] = {}
        super().__init__(config)

    # ---- questions ------------------------------------------------------------

    @staticmethod
    def _gold_articles(rec: dict) -> list[str]:
        """Normalized titles of every Wikipedia article cited by any answer's proof, de-duplicated
        in first-seen order. These are the gold supporting documents for retrieval recall."""
        articles: list[str] = []
        seen: set[str] = set()
        for ans in rec.get("answer_list", []):
            for pr in ans.get("proof", []):
                url = pr.get("found_in_url")
                if not url:
                    continue
                art = _article_from_url(url)
                if art and art not in seen:
                    seen.add(art)
                    articles.append(art)
        return articles

    def _questions_from_file(self, path: str) -> list[Question]:
        questions: list[Question] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                answers = rec.get("answer_list", [])
                # one nugget per gold answer entity (KARL: "each entity is a separate nugget").
                nuggets = [str(a["answer_text"]).strip() for a in answers if str(a.get("answer_text", "")).strip()]
                # aliases per answer, for reference / debugging (the judge handles aliases semantically).
                aliases = {str(a.get("answer_text", "")): list(a.get("aliases", [])) for a in answers}
                questions.append(
                    Question(
                        qid=str(rec["qid"]),
                        text=str(rec["question_text"]),
                        gold=", ".join(nuggets),
                        gold_docs=self._gold_articles(rec),
                        meta={
                            "nuggets": nuggets,
                            "n_nuggets": len(nuggets),
                            "aliases": aliases,
                        },
                    )
                )
        return questions

    def load_questions(self) -> list[Question]:
        # test_data.jsonl (the 1000 KARL qids) PLUS the small train-sampled dev extract, so both the
        # test and dev splits resolve from one load (the runner filters by split qid). test/dev qids
        # are disjoint by construction (different source files -> distinct `__test`/`__train` suffixes).
        questions = self._questions_from_file(self.config.questions_path)
        dev_path = self.config.dev_questions_path
        if dev_path and os.path.exists(dev_path):
            questions.extend(self._questions_from_file(dev_path))
        return questions

    # ---- retrieval substrate --------------------------------------------------

    def _build_resources(self) -> BenchmarkResources:
        collection = self._open_chroma_collection()
        self._collection = collection
        # ~25.9M chunks is far too much for an in-RAM {chunk_id: text} dict, so serve the passage text
        # lazily + cached from the chroma `documents` column (only the chunks a question actually reads
        # get materialized). The systems only do keyed lookups, so the lazy mapping is a drop-in.
        return BenchmarkResources(
            chroma_collection=collection,
            document_map=_ChromaDocMap(collection),
            answer_format_hint=self.answer_format_hint,
            compute_objective=self.compute_objective,
        )

    # ---- scoring + metrics ----------------------------------------------------

    async def score(self, question: Question, predicted: str, ctx: ExecutionContext) -> dict:
        return await judge_nugget_recall(
            ctx,
            question=question.text,
            nuggets=question.meta.get("nuggets", []),
            predicted=predicted,
            model=self.config.judge_model,
            judge_system=_JUDGE_SYSTEM,
            partial_credit=self.config.partial_credit,
        )

    def _titles_for(self, retrieved: list[str]) -> list[str]:
        """Normalized article title for each retrieved doc_id (an article page_id). Titles live in
        every chunk's chroma metadata, so resolve each page_id from any one of its chunks (a
        `where={"doc_id": ...}, limit=1` get), cached across questions. Pre-migration report rows
        carry chunk_ids instead of page_ids, so unresolved ids fall back to a row-id lookup — the
        same metric then recomputes identically on old rows."""
        assert self._collection is not None  # resources (and _collection) are built before scoring
        todo = [str(r) for r in retrieved if str(r) not in self._title_cache]
        for pid in dict.fromkeys(todo):
            got = self._collection.get(where={"doc_id": pid}, limit=1, include=["metadatas"])
            metas = got.get("metadatas") or []
            if metas:
                self._title_cache[pid] = _norm_title((metas[0] or {}).get("title", ""))
        missing = [pid for pid in dict.fromkeys(todo) if pid not in self._title_cache]
        if missing:
            got = self._collection.get(ids=missing, include=["metadatas"])
            for cid, meta in zip(got.get("ids") or [], got.get("metadatas") or []):
                self._title_cache[str(cid)] = _norm_title((meta or {}).get("title", ""))
        return [self._title_cache.get(str(r), "") for r in retrieved]

    def recall_metrics(self, retrieved: list[str] | None, question: Question) -> dict[str, float]:
        # Relevance is labeled at the Wikipedia-article level (a proof cites a supporting article),
        # and the retrieval unit is the article too (doc_id = page_id; see CORPUS_MODEL.md), but the
        # gold join key is the normalized TITLE (proofs cite title-slug URLs), so map each retrieved
        # page_id to its title via the chunk metadata and report article-level doc_recall.
        if retrieved is None:
            ret_articles: list[str] | None = None
        else:
            ret_articles = self._titles_for(retrieved)
        return {"doc_recall": doc_recall(ret_articles, question.gold_docs)}
