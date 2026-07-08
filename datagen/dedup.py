"""Question-answer pair de-duplication.

Given a freshly-generated batch of synthetic QA pairs and a ChromaDB
collection holding pre-computed embeddings for the benchmark's question /
answer corpus (plus any previously-accepted synthetic pairs), filter the
batch through six funnel stages:

1. Exact-string match against existing questions.
2. Exact-string match against existing answers.
3. LLM judge across the remaining synthetic pairs in this batch.
4. Embed surviving Q + A; look up top-k most similar existing Q and A.
5. LLM judge against each top-k neighbour Q's full QA pair.
6. LLM judge against each top-k neighbour A's full QA pair.

For every input pair we emit a JSON funnel record so the full attrition
funnel can be replayed off-line. Kept pairs have their Q + A embeddings
upserted into the collection so subsequent batches can deduplicate against
them.

The judge prompt and the metadata schema for the collection are described
in ``prompts.yaml`` and in ``METADATA_*`` constants below, respectively.
"""

from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

# The dedup judge is a best-effort filter: a transient OpenRouter failure (e.g.
# a 504 envelope the SDK can't unmarshal) must NOT take down a whole seed's
# dedup. Retry a few times, then fall back to "not a duplicate" (keep the pair),
# consistent with the module's bias toward keeping possible duplicates over
# silently dropping good pairs.
_JUDGE_MAX_RETRIES = 3
_JUDGE_BACKOFF_BASE_SEC = 1.0
_JUDGE_BACKOFF_MAX_SEC = 10.0

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from openrouter import OpenRouter

    from datagen.harness import QAPair


# metadata schema used on the qa_collection. Every embedding row carries:
#   kind ∈ {"question", "answer"}
#   qa_id          a unique id linking a Q and its A together
#   is_synthetic   True for synthetic pairs accepted by this pipeline,
#                  False for pre-loaded benchmark validation pairs.
METADATA_KIND = "kind"
METADATA_QA_ID = "qa_id"
METADATA_IS_SYNTHETIC = "is_synthetic"
KIND_QUESTION = "question"
KIND_ANSWER = "answer"

# page size when walking the entire qa_collection for the exact-match step
_PAGE_SIZE = 5000

# match a JSON object containing a "duplicate" key anywhere in the model output
_DUPLICATE_RE = re.compile(r'"duplicate"\s*:\s*"(yes|no)"', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Config + record types
# ---------------------------------------------------------------------------

@dataclass
class DedupConfig:
    qa_collection: Collection
    emb_model_id: str
    or_client: OpenRouter
    judge_model_id: str
    judge_prompt_template: str
    top_k: int = 20
    judge_parallelism: int = 8
    service_tier: str | None = "flex"


@dataclass
class _ExistingPair:
    """In-memory view of one existing QA pair, indexed by qa_id."""
    qa_id: str
    question: str | None
    answer: str | None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def dedup_batch(batch: list[QAPair], cfg: DedupConfig) -> tuple[list[QAPair], list[dict]]:
    """De-duplicate ``batch`` against ``cfg.qa_collection`` (and within itself).

    Returns ``(kept, funnel_data)``. ``kept`` is the surviving subset of ``batch``
    in original order; ``funnel_data`` is a list of one record per input pair
    capturing every funnel decision (one element per ``batch`` entry, also
    in input order).

    Side effect: the Q and A embeddings of every kept pair are upserted into
    ``cfg.qa_collection`` so subsequent ``dedup_batch`` calls within the same
    process see them.
    """
    funnel_data: list[dict] = [_make_funnel_record(p) for p in batch]
    active: list[int] = list(range(len(batch)))

    # ---- step 1+2: exact-string match against existing Q and A. ----------
    existing_questions: dict[str, str] = _scan_existing(cfg.qa_collection, KIND_QUESTION)
    existing_answers: dict[str, str] = _scan_existing(cfg.qa_collection, KIND_ANSWER)

    # invert to {text -> qa_id} for O(1) lookup; later-inserted duplicates of the
    # same text would silently collide here -- accept first.
    question_to_qid = {question: qid for qid, question in existing_questions.items()}
    answer_to_qid = {answer: qid for qid, answer in existing_answers.items()}

    surviving: list[int] = []
    for idx in active:
        record = funnel_data[idx]
        synthetic_question = batch[idx].question
        synthetic_answer = _canonical_answer_string(batch[idx].answer)

        match_qid = question_to_qid.get(synthetic_question)
        if match_qid is not None:
            record["exact_question_match_id"] = match_qid
            record["filtered"] = True
            continue

        match_qid = answer_to_qid.get(synthetic_answer)
        if match_qid is not None:
            record["exact_answer_match_id"] = match_qid
            record["filtered"] = True
            continue

        surviving.append(idx)

    active = surviving

    # ---- step 3: intra-batch judge on remaining synthetic pairs. ---------
    # compare each surviving pair against every other surviving pair earlier
    # in the batch; if duplicate, drop the later one.
    surviving = []
    for idx in active:
        is_dup_of: str | None = None
        if surviving:
            decisions = _judge_many(
                cfg,
                [(batch[idx], batch[j]) for j in surviving],
            )
            for j, dup in zip(surviving, decisions, strict=True):
                if dup:
                    is_dup_of = batch[j].qa_id
                    break
        if is_dup_of is not None:
            funnel_data[idx]["intra_batch_duplicate_of"] = is_dup_of
            funnel_data[idx]["filtered"] = True
        else:
            surviving.append(idx)
    active = surviving

    if not active:
        return [], funnel_data

    # ---- step 4: embed surviving synthetic Q + A; look up neighbours. ----
    question_texts = [batch[i].question for i in active]
    answer_texts = [_canonical_answer_string(batch[i].answer) for i in active]
    question_embs = _embed(cfg.or_client, cfg.emb_model_id, question_texts)
    answer_embs = _embed(cfg.or_client, cfg.emb_model_id, answer_texts)

    question_neighbors_per_pair = _query_neighbors(cfg.qa_collection, question_embs, cfg.top_k, KIND_QUESTION)
    answer_neighbors_per_pair = _query_neighbors(cfg.qa_collection, answer_embs, cfg.top_k, KIND_ANSWER)

    # `active` is the slot ordering for question_embs / answer_embs / *_neighbors_per_pair.
    # Capture the mapping now since `active` will be filtered again in step 5/6.
    idx_to_slot = {idx: s for s, idx in enumerate(active)}

    for slot, idx in enumerate(active):
        funnel_data[idx]["topk_similar_questions"] = question_neighbors_per_pair[slot]
        funnel_data[idx]["topk_similar_answers"] = answer_neighbors_per_pair[slot]

    # map qa_id -> (question, answer) using cached existing maps.
    pair_view: dict[str, _ExistingPair] = {}
    for qid, qtext in existing_questions.items():
        pair_view.setdefault(qid, _ExistingPair(qid, None, None)).question = qtext
    for qid, atext in existing_answers.items():
        pair_view.setdefault(qid, _ExistingPair(qid, None, None)).answer = atext

    # ---- step 5 + 6: judge each surviving pair against neighbours. -------
    surviving = []
    for slot, idx in enumerate(active):
        synth = batch[idx]

        q_dup_id = _judge_against_neighbors(cfg, synth, question_neighbors_per_pair[slot], pair_view)
        if q_dup_id is not None:
            funnel_data[idx]["question_neighbor_duplicate_of"] = q_dup_id
            funnel_data[idx]["filtered"] = True
            continue

        a_dup_id = _judge_against_neighbors(cfg, synth, answer_neighbors_per_pair[slot], pair_view)
        if a_dup_id is not None:
            funnel_data[idx]["answer_neighbor_duplicate_of"] = a_dup_id
            funnel_data[idx]["filtered"] = True
            continue

        surviving.append(idx)
    active = surviving

    # ---- step 7: insert surviving pairs' embeddings into the collection. -
    kept_pairs: list[QAPair] = []
    upsert_ids: list[str] = []
    upsert_embs: list[list[float]] = []
    upsert_docs: list[str] = []
    upsert_meta: list[dict] = []

    for idx in active:
        slot = idx_to_slot[idx]
        synth = batch[idx]
        a_text = _canonical_answer_string(synth.answer)
        upsert_ids.append(f"{synth.qa_id}:q")
        upsert_embs.append(list(question_embs[slot]))
        upsert_docs.append(synth.question)
        upsert_meta.append({
            METADATA_KIND: KIND_QUESTION,
            METADATA_QA_ID: synth.qa_id,
            METADATA_IS_SYNTHETIC: True,
        })
        upsert_ids.append(f"{synth.qa_id}:a")
        upsert_embs.append(list(answer_embs[slot]))
        upsert_docs.append(a_text)
        upsert_meta.append({
            METADATA_KIND: KIND_ANSWER,
            METADATA_QA_ID: synth.qa_id,
            METADATA_IS_SYNTHETIC: True,
        })
        kept_pairs.append(synth)

    if upsert_ids:
        cfg.qa_collection.upsert(
            ids=upsert_ids,
            embeddings=upsert_embs,  # type: ignore
            documents=upsert_docs,
            metadatas=upsert_meta,  # type: ignore
        )

    return kept_pairs, funnel_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_funnel_record(pair: QAPair) -> dict:
    return {
        "qa_id": pair.qa_id,
        "question": pair.question,
        "answer": pair.answer,
        "filtered": False,
        "exact_question_match_id": None,
        "exact_answer_match_id": None,
        "intra_batch_duplicate_of": None,
        "topk_similar_questions": None,
        "topk_similar_answers": None,
        "question_neighbor_duplicate_of": None,
        "answer_neighbor_duplicate_of": None,
    }


def _canonical_answer_string(answer: list[str]) -> str:
    """Sort nuggets and join into a stable canonical string for matching.

    Used for both the exact-match step and as the input to the answer
    embedding. Sorting makes the representation invariant to nugget order so
    two pairs that list the same nuggets in different orders are treated as
    duplicates by step 1/2.
    """
    return "\n".join(sorted(n.strip() for n in answer))


def _bullet_list_answer(answer: str | list[str] | None) -> str:
    """Format an answer (synthetic list[str] or existing str) as bullets."""
    if answer is None:
        return "  - (no answer recorded)"
    items = [n.strip() for n in answer if n.strip()] if isinstance(answer, list) else [answer.strip()]
    if not items:
        return "  - (empty)"
    return "\n".join(f"  - {n}" for n in items)


def _scan_existing(collection: Collection, kind: str) -> dict[str, str]:
    """Return ``{qa_id: text}`` for every existing embedding of ``kind``."""
    out: dict[str, str] = {}
    offset = 0
    where = {METADATA_KIND: kind}
    while True:
        page = collection.get(
            where=where,  # type: ignore
            include=["documents", "metadatas"],
            limit=_PAGE_SIZE,
            offset=offset,
        )
        page_ids = page["ids"]
        if not page_ids:
            break
        docs = page.get("documents") or [None] * len(page_ids)
        metas = page.get("metadatas") or [{}] * len(page_ids)
        for doc, meta in zip(docs, metas, strict=True):
            qid = str((meta or {}).get(METADATA_QA_ID, ""))
            if not qid or doc is None:
                continue

            out.setdefault(qid, str(doc))
        offset += len(page_ids)
        if len(page_ids) < _PAGE_SIZE:
            break
    return out


def _embed(or_client: OpenRouter, model_id: str, texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings using OpenRouter's embeddings API."""
    if not texts:
        return []
    safe = [t if t else " " for t in texts]
    resp = or_client.embeddings.generate(input=safe, model=model_id)  # type: ignore
    return [list(d.embedding) for d in resp.data]  # type: ignore


def _query_neighbors(
    collection: Collection,
    embeddings: list[list[float]],
    top_k: int,
    kind: str,
) -> list[list[dict]]:
    """Return top-k neighbours per input embedding, restricted to ``kind``.

    Each neighbour is ``{"qa_id": str, "distance": float}``.
    """
    if not embeddings:
        return []
    where = {METADATA_KIND: kind}
    results = collection.query(
        query_embeddings=embeddings,  # type: ignore
        n_results=top_k,
        where=where,  # type: ignore
        include=["metadatas", "distances"],
    )
    out: list[list[dict]] = []
    metas_per = results.get("metadatas") or [[]] * len(embeddings)
    dists_per = results.get("distances") or [[]] * len(embeddings)
    for metas, dists in zip(metas_per, dists_per, strict=True):
        neighbours = []
        for meta, dist in zip(metas or [], dists or [], strict=True):
            qid = str((meta or {}).get(METADATA_QA_ID, ""))
            if not qid:
                continue
            neighbours.append({"qa_id": qid, "distance": float(dist)})
        out.append(neighbours)
    return out


def _judge_against_neighbors(
    cfg: DedupConfig,
    synth: QAPair,
    neighbours: list[dict],
    pair_view: dict[str, _ExistingPair],
) -> str | None:
    """Return the first neighbour qa_id judged a duplicate, or None."""
    if not neighbours:
        return None

    # materialise the validation pair for each neighbour. Skip neighbours we
    # can't fully reconstruct (e.g. Q present in collection but A missing).
    candidates: list[tuple[str, _ExistingPair]] = []
    for n in neighbours:
        ep = pair_view.get(n["qa_id"])
        if ep is None:
            continue
        candidates.append((n["qa_id"], ep))
    if not candidates:
        return None

    decisions = _judge_many(
        cfg,
        [(synth, ep) for _, ep in candidates],
    )
    for (qid, _), dup in zip(candidates, decisions, strict=True):
        if dup:
            return qid
    return None


def _judge_many(
    cfg: DedupConfig,
    comparisons: list[tuple[QAPair, QAPair | _ExistingPair]],
) -> list[bool]:
    """Run the dedup judge over a list of (generated, validation) pairs in parallel."""
    if not comparisons:
        return []

    def _one(args):
        gen, val = args
        prompt = _render_judge_prompt(cfg.judge_prompt_template, gen, val)
        return _judge_call(cfg.or_client, cfg.judge_model_id, prompt, cfg.service_tier)

    with ThreadPoolExecutor(max_workers=max(1, cfg.judge_parallelism)) as pool:
        return list(pool.map(_one, comparisons))


def _render_judge_prompt(
    template: str,
    generated: QAPair,
    validation: QAPair | _ExistingPair,
) -> str:
    """Render the Jinja judge template for one (generated, validation) pair."""
    from jinja2 import Template  # local import to avoid loading at module import

    if hasattr(validation, "answer") and isinstance(validation.answer, list):
        val_answer = _bullet_list_answer(validation.answer)  # type: ignore[union-attr]
    else:
        val_answer = _bullet_list_answer(getattr(validation, "answer", None))

    return Template(template).render(
        generated_question=generated.question,
        generated_answer=_bullet_list_answer(generated.answer),
        validation_question=getattr(validation, "question", "") or "",
        validation_answer=val_answer,
    )


def _judge_call(
    or_client: OpenRouter,
    model_id: str,
    prompt: str,
    service_tier: str | None = None,
) -> bool:
    """Issue a single LLM judge call and parse the JSON ``duplicate`` field.

    The call is retried with exponential backoff on transient API failures
    (network errors, 5xx, or an error envelope the SDK can't unmarshal into a
    ChatCompletion). On persistent failure we default to ``False`` (not a
    duplicate -> keep the pair) rather than letting one flaky judge call abort
    the whole seed's dedup -- keeping a possible duplicate is the lesser evil.
    """
    kwargs: dict = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    if service_tier is not None:
        kwargs["service_tier"] = service_tier

    text: str | None = None
    for attempt in range(_JUDGE_MAX_RETRIES + 1):
        try:
            resp = or_client.chat.send(**kwargs)  # type: ignore
            text = resp.choices[0].message.content or ""  # type: ignore
            break
        except Exception as e:
            if attempt >= _JUDGE_MAX_RETRIES:
                print(f"  WARN: dedup judge call failed after retries ({e}); keeping pair")
                return False
            delay = min(_JUDGE_BACKOFF_BASE_SEC * (2 ** attempt), _JUDGE_BACKOFF_MAX_SEC)
            time.sleep(delay + random.uniform(0.0, 1.0))

    m = _DUPLICATE_RE.search(text or "")  # type: ignore
    if m is None:
        # if the judge fails to follow format, do not filter the pair
        # we'd rather keep a possible duplicate than drop a good pair silently.
        return False
    return m.group(1).lower() == "yes"
