"""Tool implementations for the Bootstrap, Enrich, and Search agents.
"""

from __future__ import annotations

import gc
import inspect
import json
import re

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from chromadb import Collection
from chromadb.api import ClientAPI
from chromadb.errors import NotFoundError

from skunk.common import PageLocator, estimate_tokens, render_page_b64
from skunk.config import SearchAgentConfig
from skunk.agents.multi_turn_agent import Tool
from skunk.storage.document_map import DocumentMap
from skunk.usage import match_model_entry

from qatfd.config import BootstrapConfig, EnrichConfig
from qatfd.constants import METADATA_LIST_DELIMITER
from qatfd.prompts import load_qatfd_prompts

if TYPE_CHECKING:
    from skunk.llm_client import LLMClient

# chroma fetches the rows of a regex (`where_document`) get by binding every matched id into one SQL
# `IN (...)`, so a grep whose match set exceeds SQLite's bound-parameter cap fails with "too many SQL
# variables". Greps that may exceed it are run as an id scan (no row fetch, no cap) plus batched row fetches.
_GREP_MAX_LIMIT = 32_766
# rows per batched grep fetch: small when embeddings ride along (a 4096-dim embedding is ~40 KB as JSON,
# and chroma 1.x rejects responses over ~40 MB), larger for the text-only fetch the collection agents use
_GREP_ROW_BATCH_WITH_EMBEDDINGS = 256
_GREP_ROW_BATCH_TEXT_ONLY = 2048
# batch size for iterating over collection for map
_ITER_BATCH_SIZE = 256
# batch size for copying chunks between collections: chroma 1.x rejects requests over 40 MB (413), and a
# 4096-dim float embedding is ~40 KB as JSON, so keep each get/upsert to a few hundred chunks
_COPY_BATCH_SIZE = 256
# pages (get / upsert round-trips) between forced `gc.collect()` calls in the batched chroma loops (`_grep_one`,
# `_iter_get`, `_CollectionToolBase._copy`). chromadb's HttpClient is httpx, and in httpx 0.28 a finished
# `Response` and its `BoundSyncStream` point at each other, so a response (and the multi-MB JSON body cached on
# it) is only ever freed by the cyclic collector, never by reference counting. `bytes` do not count toward the
# collector's allocation threshold, so on the runner's large long-lived heap a full collection almost never
# fires on its own and the bodies pile up: ~20 MB per 256-chunk copy batch, measured as +8.8 GB of RSS on one
# 40k-chunk copy vs +0.9 GB with periodic collects. A full collection costs ~60 ms on that heap, so every 32
# pages bounds the retained bodies to well under 1 GB per tool call at negligible cost.
_GC_EVERY_PAGES = 32
# default cap on the number of chunks one create/add/copy/merge may move (guards against a loose
# `metadata_filter` over the base collection); the tools take an explicit `max_copy_chunks` to override
_DEFAULT_MAX_COPY_CHUNKS = 100_000
# chromadb collection-name rule (3-512 chars from [a-zA-Z0-9._-], starting and ending alphanumeric)
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")
# collection-metadata flag marking collections created/managed by qatfd agents (vs. the base corpus)
COLLECTION_FLAG_KEY = "is_working_set"

# regular expression for capturing json (or code) blocks
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_]*)\n(.*?)```", re.DOTALL)

# templates / strings for each tool's docstring
_PROMPTS = load_qatfd_prompts("tools")

# sanitization for map outputs
_SCALAR = (str, int, float, bool, type(None))

# metadata keys a map may never overwrite (they identify the chunk / document)
_RESERVED_FIELDS = frozenset({"chunk_id", "doc_id"})

def _sanitize_fields(obj) -> dict:
    if not isinstance(obj, dict):
        return {"map_error": f"expected a dict of fields, got {type(obj).__name__}"}
    clean = {
        k: (v if isinstance(v, _SCALAR) else json.dumps(v))
        for k, v in obj.items() if isinstance(k, str) and k not in _RESERVED_FIELDS
    }
    return clean or {"map_error": "empty output"}


def _map_fn_takes_metadata(fn: Callable) -> bool:
    """Does `fn` accept the `(text, metadata)` pair? True for a var-positional signature (every sandbox-defined
    function / lambda) and for two or more positional parameters; False for a one-parameter plain callable."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return True
    if any(p.kind == p.VAR_POSITIONAL for p in params):
        return True
    return sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)) >= 2


def _chunk_metadata(chunk_id: str, meta: dict | None) -> dict:
    """The metadata dict a map sees for one chunk: chroma's row metadata with `chunk_id` guaranteed."""
    return {"chunk_id": chunk_id, **(meta or {})}


def _doc_level_metadata(page: dict) -> dict[str, dict]:
    """doc_id -> the metadata a document-level map sees: the key/value pairs shared by every chunk of that
    document in this page (chunk-specific keys such as chunk_id / element_id drop out), `doc_id` always kept."""
    shared: dict[str, dict] = {}
    for meta in page["metadatas"]:
        meta = meta or {}
        doc_id = meta.get("doc_id", "?")
        if doc_id not in shared:
            shared[doc_id] = {k: v for k, v in meta.items() if k != "chunk_id"}
        else:
            cur = shared[doc_id]
            shared[doc_id] = {k: v for k, v in cur.items() if k in meta and meta[k] == v}
        shared[doc_id]["doc_id"] = doc_id
    return shared

# helper functions for calling query and get on collections
def _query_one(chroma_client: ClientAPI, name: str, query_kwargs: dict) -> tuple[str, dict]:
    c = chroma_client.get_collection(name)
    return name, c.query(**query_kwargs)  # type: ignore

def _get_one(chroma_client: ClientAPI, name: str, get_kwargs: dict) -> tuple[str, dict]:
    c = chroma_client.get_collection(name)
    return name, c.get(**get_kwargs)  # type: ignore

def _grep_one(chroma_client: ClientAPI, name: str, get_kwargs: dict, include: list[str], limit: int | None) -> tuple[str, dict]:
    """One collection's regex get. A `limit` at or under `_GREP_MAX_LIMIT` is a single get; an unlimited
    (or larger) grep scans the matching ids first (ids alone are not subject to the SQL-variable cap) and
    then fetches the rows in id batches, returning the same `{ids, metadatas, documents[, embeddings]}`
    dict a single get would, in match order."""
    c = chroma_client.get_collection(name)
    if limit is not None and limit <= _GREP_MAX_LIMIT:
        return name, c.get(include=include, limit=limit, **get_kwargs)  # type: ignore
    scan_kwargs = dict(get_kwargs, limit=limit) if limit is not None else get_kwargs
    ids = list(c.get(include=[], **scan_kwargs)["ids"])  # type: ignore
    out: dict = {"ids": [], **{key: [] for key in include}}
    batch = _GREP_ROW_BATCH_WITH_EMBEDDINGS if "embeddings" in include else _GREP_ROW_BATCH_TEXT_ONLY
    for page, i in enumerate(range(0, len(ids), batch), start=1):
        wanted = ids[i : i + batch]
        got = c.get(ids=wanted, include=include)  # type: ignore
        pos = {chunk_id: j for j, chunk_id in enumerate(got["ids"])}
        order = [pos[chunk_id] for chunk_id in wanted if chunk_id in pos]  # keep the scan's order
        out["ids"].extend(got["ids"][j] for j in order)
        for key in include:
            rows = got.get(key)
            out[key].extend(rows[j] for j in order) if rows is not None else out[key].extend([None] * len(order))
        if page % _GC_EVERY_PAGES == 0:
            gc.collect()
    return name, out

def _iter_get(collection: Collection, include: list[str], batch_size: int = _ITER_BATCH_SIZE) -> Iterator[dict]:
    """Yield collection.get() pages of at most batch_size rows. Pages by id, not offset."""
    ids = collection.get(include=[])["ids"]
    for page, i in enumerate(range(0, len(ids), batch_size), start=1):
        yield collection.get(ids=ids[i : i + batch_size], include=include) # type: ignore
        if page % _GC_EVERY_PAGES == 0:
            gc.collect()

def _format_search_results(collection_to_results: dict, is_query: bool, include_embeddings: bool) -> dict[str, list[SearchResult]]:
    # format return results as dict of collection-->list[SearchResult]
    full_results = {}
    for collection, results in collection_to_results.items():
        assert "documents" in results and "metadatas" in results
        ids = results["ids"][0] if is_query else results["ids"]
        texts = results["documents"][0] if is_query else results["documents"]
        metadatas = results["metadatas"][0] if is_query else results["metadatas"]
        embeddings = [None] * len(ids)
        if include_embeddings:
            embeddings = results["embeddings"][0] if is_query else results["embeddings"]

        chunks: list[SearchResult] = []
        for rank, (_id, text, meta, emb) in enumerate(zip(ids, texts, metadatas, embeddings, strict=True), 1):
            doc_id = meta.get("doc_id", "?")
            est_num_tokens = estimate_tokens(text)
            header = f"[{rank}] collection={collection} | chunk_id={_id} | doc_id={doc_id} | est_num_tokens={est_num_tokens} | metadata={json.dumps(meta)}"
            chunks.append(SearchResult(_id, doc_id, header, text, meta, emb))

        full_results[collection] = chunks

    return full_results


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    header: str | None
    text: str
    metadata: dict
    embedding: list[float] | None

@dataclass
class MapSample:
    chunk_id: str
    doc_id: str
    text: str
    fields: dict
    error: str | None

@dataclass
class MapResult:
    samples: list[MapSample]
    error: str | None


# ---- collection metadata: schema (`fields`) persistence -------------------------------------------
# chroma collection metadata cannot hold lists, so a collection's schema is persisted under the "fields"
# key as a text block with one "  - name (type): description" line per field. Agents are shown it verbatim.
_FIELD_LINE_RE = re.compile(r"^\s*-\s*([^\s(]+)")


def format_metadata_fields(fields: list[dict]) -> str:
    """Render field specs (`{"name", "type", "desc"}` dicts) as the `fields` metadata string."""
    return "".join(f"  - {f['name']} ({f['type']}): {f['desc']}\n" for f in fields)


def _type_name(t) -> str | None:
    """A field `type` rendered as text: a class by its name (`str` -> "str"), a typing / PEP 604 form by its
    repr (`float | None`, `list[str]`, `Optional[int]`), a string unchanged; None for anything unrenderable."""
    if isinstance(t, str):
        return t
    if isinstance(t, type):
        return t.__name__
    if t is None:
        return None
    return str(t).replace("typing.", "")


def normalize_metadata_fields(fields):
    """`fields` with each spec's `type` rendered as text (see `_type_name`), so agents may declare types as
    Python types (`str`, `float | None`, `list[str]`) as well as strings — the tool docs call `type` "the
    Python type of the field", and the sandbox evaluates a bare `str` to the class. Specs are copied, not
    mutated; anything that is not a list of dicts passes through for `validate_metadata_fields` to reject."""
    if not isinstance(fields, list):
        return fields
    return [{**f, "type": _type_name(f.get("type"))} if isinstance(f, dict) else f for f in fields]


def validate_metadata_fields(fields) -> str | None:
    """None if `fields` is a well-formed list of `{"name", "type", "desc"}` dicts, else an error message.
    Run `normalize_metadata_fields` first so a Python-type `type` is accepted."""
    if not isinstance(fields, list) or not fields:
        return "`fields` must be a non-empty list of {name, type, desc} dicts"
    for f in fields:
        if not isinstance(f, dict) or any(not isinstance(f.get(k), str) or not f[k].strip() for k in ("name", "type", "desc")):
            return f"each field must be a dict with non-empty string values for name, type, and desc; got {f!r}"
    return None


def append_action(collection, action: str) -> None:
    """Append one tool call to the collection's `actions` log (a METADATA_LIST_DELIMITER-joined string)."""
    metadata = dict(collection.metadata or {})
    action_items = [a for a in (metadata.get("actions") or "").split(METADATA_LIST_DELIMITER) if a]
    action_items.append(action)
    metadata["actions"] = METADATA_LIST_DELIMITER.join(action_items)
    collection.modify(metadata=metadata)


def merge_metadata_fields(*field_strs: str | None) -> str:
    """Union of `fields` strings: one line per distinct field name, first occurrence wins, order kept."""
    seen: set[str] = set()
    out: list[str] = []
    for fs in field_strs:
        for line in (fs or "").splitlines():
            if not line.strip():
                continue
            m = _FIELD_LINE_RE.match(line)
            key = m.group(1) if m else line.strip()
            if key in seen:
                continue
            seen.add(key)
            out.append(line.rstrip() + "\n")
    return "".join(out)


def _persist_fields(collection, fields: list[dict]) -> None:
    """Merge `fields` into the collection's `fields` metadata schema."""
    meta = dict(collection.metadata or {})
    meta["fields"] = merge_metadata_fields(meta.get("fields"), format_metadata_fields(fields))
    collection.modify(metadata=meta)


def managed_collection_metadata(description: str, fields: str = "", created_by: str = "", created_by_agent_type: str = "") -> dict:
    """Metadata for a brand-new agent-managed collection. Given to `create_collection` atomically so no
    sibling ever observes the collection with `metadata=None`."""
    return {COLLECTION_FLAG_KEY: True, "description": description, "fields": fields, "actions": "", "created_by": created_by, "created_by_agent_type": created_by_agent_type}


def is_managed_collection(collection: Collection) -> bool:
    return bool((collection.metadata or {}).get(COLLECTION_FLAG_KEY))


# ---- ResultSet: the search/grep payload, composable in agent code ------------------------------------
class ResultSet(dict):
    """The payload returned by `search_corpus` / `grep_corpus`: a plain dict (`tool`, `tool_kwargs`,
    `results: {collection: [SearchResult]}`, optional `error`) so tracing/rendering code that indexes it
    keeps working, plus set algebra over chunks so agents can compose results in Python before writing
    them to a collection: `a | b`, `a & b`, `a - b`, `a.where(fn)`. Membership is by `chunk_id`; ids are
    global across collections (copies keep their ids), so a chunk counts once wherever it was found."""

    @property
    def results(self) -> dict[str, list[SearchResult]]:
        """Simple accessor for the "results" field of the dictionary."""
        return self.get("results") or {}

    def _indexed(self) -> dict[str, tuple[str, SearchResult]]:
        """chunk_id -> (source collection, SearchResult); first occurrence wins."""
        out: dict[str, tuple[str, SearchResult]] = {}
        for collection, hits in self.results.items():
            for hit in hits:
                out.setdefault(hit.chunk_id, (collection, hit))
        return out

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._indexed().keys())

    @property
    def doc_ids(self) -> list[str]:
        return list(dict.fromkeys(hit.doc_id for _, hit in self._indexed().values()))

    def by_source(self) -> dict[str, list[str]]:
        """source collection -> chunk ids to copy from it (each chunk attributed to one source)."""
        out: dict[str, list[str]] = {}
        for chunk_id, (collection, _) in self._indexed().items():
            out.setdefault(collection, []).append(chunk_id)
        return out

    def __len__(self) -> int:
        return len(self._indexed())

    def __repr__(self) -> str:
        return (
            f"ResultSet(tool={self.get('tool')!r}, chunks={len(self)}, docs={len(self.doc_ids)}, "
            f"collections={list(self.results.keys())!r})"
        )

    def provenance(self) -> dict:
        """How this result set was produced, JSON-friendly: the tool and its kwargs (a `grep_corpus`
        pattern / limit, a `search_corpus` query / top_k, ...) or, for a composed set, the operator and
        the provenance of its operands. Recorded in a collection's action ledger by create/add so the
        Enrich agent can see exactly which searches built a collection."""
        out = {"tool": self.get("tool"), "tool_kwargs": self.get("tool_kwargs"), "num_chunks": len(self)}
        if self.get("error"):
            out["error"] = self["error"]
        return out

    @staticmethod
    def _from_indexed(indexed: dict[str, tuple[str, SearchResult]], tool_kwargs: dict, errors: list[str]) -> ResultSet:
        results: dict[str, list[SearchResult]] = {}
        for collection, hit in indexed.values():
            results.setdefault(collection, []).append(hit)
        rs = ResultSet({"tool": "result_set", "tool_kwargs": tool_kwargs, "results": results})
        # an operand that failed (e.g. a grep that hit the SQL-variable cap) must not turn into a silently
        # empty operand: the composed set carries the failure so create/add refuse it, like the raw result
        if errors:
            rs["error"] = "; ".join(errors)
        return rs

    def where(self, fn: Callable[[dict], bool]) -> ResultSet:
        """Keep only chunks whose metadata dict satisfies `fn`."""
        kept = {cid: (c, h) for cid, (c, h) in self._indexed().items() if fn(h.metadata or {})}
        return self._from_indexed(kept, {"op": "where", "left": self.provenance()}, [e for e in [self.get("error")] if e])

    def _combine(self, other: dict, op: str) -> ResultSet:
        if not isinstance(other, dict) or "results" not in other:
            raise TypeError(f"unsupported operand for {op}: ResultSet and {type(other).__name__}")
        other_rs = other if isinstance(other, ResultSet) else ResultSet(other)
        a, b = self._indexed(), other_rs._indexed()
        if op == "|":
            merged = {**a, **{k: v for k, v in b.items() if k not in a}}
        elif op == "&":
            merged = {k: v for k, v in a.items() if k in b}
        elif op == "-":
            merged = {k: v for k, v in a.items() if k not in b}
        else:
            raise Exception(f"Unsupported operator: {op}")
        errors = [e for e in (self.get("error"), other_rs.get("error")) if e]
        return self._from_indexed(merged, {"op": op, "left": self.provenance(), "right": other_rs.provenance()}, errors)

    # `other` is typed as dict to stay a valid override of dict.__or__; a ResultSet is what is expected.
    def __or__(self, other: dict) -> ResultSet:  # type: ignore[override]
        return self._combine(other, "|")

    def __and__(self, other: dict) -> ResultSet:
        return self._combine(other, "&")

    def __sub__(self, other: dict) -> ResultSet:
        return self._combine(other, "-")


class SearchCorpusTool(Tool):
    name = "search_corpus"
    doc: str = _PROMPTS["search_corpus"]

    def __init__(
        self,
        chroma_client: ClientAPI,
        llm_client: LLMClient,
        usage_key: str = "default",
        timeout_s: float | None = None,
        max_parallel_chroma_queries: int = 16,
        include_embeddings: bool = False,
    ):
        self._chroma_client = chroma_client
        self._llm_client = llm_client
        self._usage_key = usage_key
        self._timeout_s = timeout_s
        self._max_parallel_chroma_queries = max_parallel_chroma_queries
        self._include_embeddings = include_embeddings

    def _embed_query(self, query: str) -> list[float]:
        """Embed `query` with the same model that produced the stored embeddings, via the
        LLMClient — which owns backend dispatch (OpenRouter / vLLM), the process-wide "embed"
        rate bucket, retry, and usage accounting."""
        return self._llm_client.embed_query(query, usage_key=self._usage_key, timeout_s=self._timeout_s)

    def __call__(
        self,
        collections: list[str],
        query: str,
        top_k: int,
        metadata_filter: dict | None = None,
    ) -> dict:
        tool_kwargs = {"collections": collections, "query": query, "top_k": top_k, "metadata_filter": metadata_filter}

        # embed the query
        query_embedding = self._embed_query(query)

        # build query kwargs
        query_kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": (
                ["metadatas", "documents", "embeddings"]
                if self._include_embeddings
                else ["metadatas", "documents"]
            ),
        }
        if metadata_filter is not None:
            query_kwargs["where"] = metadata_filter

        # query results from the appropriate collection(s)
        collection_to_results = {}
        try:
            if len(collections) == 1:
                name, results = _query_one(self._chroma_client, collections[0], query_kwargs)
                collection_to_results[name] = results
            else:
                workers = max(min(len(collections), self._max_parallel_chroma_queries), 1)
                query_fn = lambda name: _query_one(self._chroma_client, name, query_kwargs)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    collection_to_results = dict(pool.map(query_fn, collections))

        except Exception as e:
            return ResultSet({"tool": self.name, "tool_kwargs": tool_kwargs, "results": {}, "error": f"search_corpus error: {e}"})

        # format return results as dict of collection-->list[SearchResult]
        full_results = _format_search_results(collection_to_results, is_query=True, include_embeddings=self._include_embeddings)

        # return results to agent
        return ResultSet({"tool": self.name, "tool_kwargs": tool_kwargs, "results": full_results})


class GrepCorpusTool(Tool):
    name = "grep_corpus"
    doc: str = _PROMPTS["grep_corpus"]

    def __init__(self, chroma_client: ClientAPI, max_parallel_chroma_queries: int = 16, include_embeddings: bool = False):
        self._chroma_client = chroma_client
        self._max_parallel_chroma_queries = max_parallel_chroma_queries
        self._include_embeddings = include_embeddings

    def __call__(
        self,
        collections: list[str],
        pattern: str,
        metadata_filter: dict | None = None,
        limit: int | None = None,
    ) -> dict:
        tool_kwargs = {"collections": collections, "pattern": pattern, "limit": limit, "metadata_filter": metadata_filter}
        get_kwargs: dict = {"where_document": {"$regex": pattern}}
        if metadata_filter is not None:
            get_kwargs["where"] = metadata_filter
        # embeddings only when the caller keeps them (the search agent's working-set upserts); the collection
        # agents copy chunks server-side by id, so their greps never need the (very large) vectors
        include = ["metadatas", "documents"] + (["embeddings"] if self._include_embeddings else [])

        # query results from the appropriate collection
        collection_to_results = {}
        try:
            if len(collections) == 1:
                name, results = _grep_one(self._chroma_client, collections[0], get_kwargs, include, limit)
                collection_to_results[name] = results
            else:
                workers = max(min(len(collections), self._max_parallel_chroma_queries), 1)
                get_fn = lambda name: _grep_one(self._chroma_client, name, get_kwargs, include, limit)
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    collection_to_results = dict(pool.map(get_fn, collections))

        except Exception as e:
            return ResultSet({"tool": self.name, "tool_kwargs": tool_kwargs, "results": {}, "error": f"grep_corpus error: {e}"})

        # format return results as dict of collection-->list[SearchResult]
        full_results = _format_search_results(collection_to_results, is_query=False, include_embeddings=self._include_embeddings)

        # return results to agent
        return ResultSet({"tool": self.name, "tool_kwargs": tool_kwargs, "results": full_results})


class ReadDocumentTool(Tool):
    """Fetch the full (or partial) text of one or more retrieval units by `doc_id` from `document_map`.

    A "document" is the corpus's retrieval unit (see the module docstring), so what this
    returns is a PDF page for OfficeQA/FinanceBench, an abstract for TREC-BioGen, an
    assembled Wikipedia article for QAMPARI, or a source file for FreshStack."""

    name = "read_document"
    doc = _PROMPTS["read_document"]

    def __init__(self, document_map: DocumentMap):
        self._document_map = document_map

    def __call__(self, doc_ids: list[str], start_char_indices: list[int] | None = None, end_char_indices: int | None | list[int | None] = None) -> dict:
        tool_kwargs = {"doc_ids": doc_ids, "start_char_indices": start_char_indices, "end_char_indices": end_char_indices}
        err_msg = "`start_char_indices` and `end_char_indices` must be None or also be lists of the same length as `doc_ids`."
        assert start_char_indices is None or (isinstance(start_char_indices, list) and len(doc_ids) == len(start_char_indices)), err_msg
        assert end_char_indices is None or (isinstance(end_char_indices, list) and len(doc_ids) == len(end_char_indices)), err_msg
        start_indices = list(start_char_indices) if isinstance(start_char_indices, list) else [start_char_indices] * len(doc_ids)
        end_indices = list(end_char_indices) if isinstance(end_char_indices, list) else [end_char_indices] * len(doc_ids)

        docs: list[dict] = []
        for did, start_idx, end_idx in zip(doc_ids, start_indices, end_indices, strict=True):
            text = self._document_map.get(did)
            if text is None:
                body = "[no such document (or no content in document)]"
            else:
                body = text
                if start_idx is not None or end_idx is not None:
                    start_idx = start_idx or 0
                    end_idx = end_idx if end_idx is not None else len(text)
                    body = text[start_idx:end_idx]
            docs.append({"doc_id": did, "text": body})

        return {"tool": self.name, "tool_kwargs": tool_kwargs, "docs": docs}


class ViewFigureTool(Tool):
    """Render the full page for a doc_id the agent saw while reading — typically because
    the page text contains a `<figure id=N>` placeholder — and hand it back as an image
    observation. We render the *whole* page (not a bbox crop) so the agent sees any
    figures in context, and so OCR coordinate errors can't clip a chart.
    """

    name = "view_figure"
    doc = _PROMPTS["view_figure"]

    def __init__(
        self,
        document_map: DocumentMap,
        page_locator: PageLocator,
        *,
        renders_dir: str | Path | None = None,
        dpi: int = 300,
        fmt: str = "png",
    ):
        self._document_map = document_map
        self._page_locator = page_locator
        self._renders_dir = renders_dir
        self._dpi = dpi
        self._fmt = fmt

    def __call__(self, doc_id: str) -> dict:
        # return an error if the agent caller hallucinated the doc_id
        if doc_id not in self._document_map:
            return {"tool": self.name, "tool_kwargs": {"doc_id": doc_id}, "error": f"no such document: {doc_id!r}"}

        # lookup the PageSource with the page locator
        page_source = self._page_locator.lookup(doc_id)
        if page_source is None:
            return {"tool": self.name, "tool_kwargs": {"doc_id": doc_id}, "error": f"No page image available for this doc_id: {doc_id!r}"}

        try:
            img = render_page_b64(
                page_source.pdf_path, page_source.page_num,
                renders_dir=self._renders_dir, dpi=self._dpi, fmt=self._fmt,
            )
        except Exception as e:
            return {"tool": self.name, "tool_kwargs": {"doc_id": doc_id}, "error": f"view_figure render error: {e}"}
        if img is None:
            return {
                "tool": self.name,
                "tool_kwargs": {"doc_id": doc_id},
                "error": f"could not render page for doc_id={doc_id} (PDF missing)",
            }
        return {
            "tool": self.name,
            "tool_kwargs": {"doc_id": doc_id},
            "doc_id": doc_id,
            "mime": img.mime,
            "data": img.data,
        }


class PruneTool(Tool):
    name = "prune"
    doc = _PROMPTS["prune"]
    reclaimer = True

    def __call__(
        self,
        chunk_ids: list[str] | None = None,
        doc_ids: list[str] | None = None,
    ) -> dict:
        return {
            "tool": self.name,
            "tool_kwargs": {"chunk_ids": chunk_ids, "doc_ids": doc_ids},
            "chunk_ids": chunk_ids,
            "doc_ids": doc_ids,
        }


class SemanticFilterTool(Tool):
    name = "semantic_filter"
    doc: str = _PROMPTS["sem_filter_tool"]

    # NOTE: _JUDGE_USER_WRAPPER captures user prompt from `_judge_one()`
    #       _JUDGE_TRUNC_WRAPPER used in `_truncate_doc_for_judge()`
    _SEM_FILTER_JUDGE_PROMPT = _PROMPTS["sem_filter_judge"]
    _JUDGE_USER_WRAPPER = "Document:\n\n\nFilter Condition: \n"
    _JUDGE_TRUNC_MARKER = "\n…(truncated to fit judge context)"

    def __init__(
        self,
        chroma_client: ClientAPI,
        llm_client: LLMClient,
        document_map: DocumentMap,
        config: SearchAgentConfig,
        model: str,
        *,
        usage_key: str = "default",
        max_parallel_chroma_queries: int = 16,
        include_embeddings: bool = False,
    ) -> None:
        self._chroma_client = chroma_client
        self._llm_client = llm_client
        self._document_map = document_map
        self._config = config
        self._model = model
        self._usage_key = usage_key
        self._max_parallel_chroma_queries = max_parallel_chroma_queries
        self._include_embeddings = include_embeddings

        # resolve the model's context limit
        self._context_limit = match_model_entry(self._model, llm_client.config.llm_context_limits)

    @staticmethod
    def _parse_bool(text: str) -> bool:
        """True/False from the judge reply. Recall-safe: default TRUE (keep) if unclear."""
        t = (text or "").strip().lower()
        if t.startswith("false"):
            return False
        if t.startswith("true"):
            return True
        if "false" in t and "true" not in t:
            return False
        return True

    def _error(self, msg: str, tool_kwargs: dict) -> dict:
        return ResultSet({"tool": self.name, "tool_kwargs": tool_kwargs, "results": {}, "kept_doc_ids": set(), "rejected_doc_ids": set(), "error": msg})

    def _metadata_only_selection(self, collections: list[str], where: dict) -> dict[str, list[SearchResult]]:
        """Use metadata filtering to return a set of candidates and candidate ids."""
        get_kwargs: dict = {
            "where": where,
            "include": ["metadatas", "documents", "embeddings"]
        }

        collection_to_results = {}
        if len(collections) == 1:
            name, results = _get_one(self._chroma_client, collections[0], get_kwargs)
            collection_to_results[name] = results
        else:
            workers = max(min(len(collections), self._max_parallel_chroma_queries), 1)
            get_fn = lambda name: _get_one(self._chroma_client, name, get_kwargs)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                collection_to_results = dict(pool.map(get_fn, collections))

        # format return results as dict of collection-->list[SearchResult]
        full_results = _format_search_results(collection_to_results, is_query=False, include_embeddings=self._include_embeddings)

        return full_results

    def _vector_search_selection(self, collections: list[str], search_str: str, top_k: int, where: dict | None) -> dict[str, list[SearchResult]]:
        """Use vector search to return a set of candidates and candidate ids."""
        emb = self._llm_client.embed_query(search_str, usage_key=self._usage_key, timeout_s=self._config.request_timeout_s)
        query_kwargs: dict = {
            "query_embeddings": [emb],
            "n_results": top_k,
            "include": ["metadatas", "documents", "embeddings"],
        }
        if where is not None:
            query_kwargs["where"] = where

        collection_to_results = {}
        if len(collections) == 1:
            name, results = _query_one(self._chroma_client, collections[0], query_kwargs)
            collection_to_results[name] = results
        else:
            workers = max(min(len(collections), self._max_parallel_chroma_queries), 1)
            query_fn = lambda name: _query_one(self._chroma_client, name, query_kwargs)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                collection_to_results = dict(pool.map(query_fn, collections))

        # format return results as dict of collection-->list[SearchResult]
        full_results = _format_search_results(collection_to_results, is_query=True, include_embeddings=self._include_embeddings)

        return full_results

    def _grep_selection(self, collections: list[str], pattern: str, limit: int | None, where: dict | None) -> dict[str, list[SearchResult]]:
        """Use grep to return a set of candidates and candidate ids."""
        get_kwargs: dict = {
            "where_document": {"$regex": pattern},
            "include": ["metadatas", "documents", "embeddings"],
        }
        if where is not None:
            get_kwargs["where"] = where
        if limit is not None:
            get_kwargs["limit"] = limit

        collection_to_results = {}
        if len(collections) == 1:
            name, results = _get_one(self._chroma_client, collections[0], get_kwargs)
            collection_to_results[name] = results
        else:
            workers = max(min(len(collections), self._max_parallel_chroma_queries), 1)
            get_fn = lambda name: _get_one(self._chroma_client, name, get_kwargs)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                collection_to_results = dict(pool.map(get_fn, collections))

        # format return results as dict of collection-->list[SearchResult]
        full_results = _format_search_results(collection_to_results, is_query=False, include_embeddings=self._include_embeddings)

        return full_results

    def _truncate_doc_for_judge(self, text: str, budget_tokens: int) -> str:
        """Head-truncate `text` to `budget_tokens` (marker included), returning (text, was_truncated)."""
        token_est = estimate_tokens(text)
        if token_est > budget_tokens:
            ratio = budget_tokens / token_est
            return text[:int(ratio * len(text))] + self._JUDGE_TRUNC_MARKER
        return text

    def _judge_doc_token_budget(self, predicate: str) -> int:
        """Max tokens of document text that fit in one judge request under `context_limit` (tokens):
        the fixed overhead (prompt scaffolding, counted via `estimate_tokens`) and the judge's output
        headroom (`max_output_tokens`) are subtracted, then the remaining token budget is returned
        with a safety factor. Returns 0 when the fixed overhead alone already exceeds the limit."""
        assert isinstance(self._context_limit, int)
        overhead_tokens = self._config.semantic_filter_max_output_tokens + estimate_tokens(
            self._SEM_FILTER_JUDGE_PROMPT + self._JUDGE_USER_WRAPPER + predicate
        )
        budget_tokens = self._context_limit * self._config.semantic_filter_context_frac - overhead_tokens
        return max(0, int(budget_tokens))

    def _judge_one(self, predicate: str, doc_id: str, item_text: str) -> tuple[str, bool]:
        user = f"Document:\n{item_text}\n\nFilter Condition: {predicate}\n"
        try:
            messages = [
                {"role": "system", "content": self._SEM_FILTER_JUDGE_PROMPT},
                {"role": "user", "content": user},
            ]
            resp = self._llm_client.call(
                messages=messages,
                temperature=0.0,
                model=self._model,
                call_site="semfilter",
                provider_order=self._config.semantic_filter_provider_order,
                max_output_tokens=self._config.semantic_filter_max_output_tokens,
                disable_reasoning=self._config.semantic_filter_disable_reasoning,
                usage_key=self._usage_key,
                timeout_s=self._config.request_timeout_s,
            )
        except Exception:
            return doc_id, True  # recall-safe: keep on error
        return doc_id, self._parse_bool(resp.text)

    def _filter_docs(self, predicate: str, collection_to_results: dict[str, list[SearchResult]]) -> tuple[set[str], set[str], dict[str, list[SearchResult]]]:
        """Returns the subset of `doc_ids` which are kept and rejected based on whether their document text satisfies `predicate`.
    
        When `self._context_limit` (the judge model's context window, in tokens) is set, each
        document's text is head-truncated so the judge request fits — otherwise a document
        larger than the judge's window would 400 and, via `_judge_one`'s recall-safe fallback,
        be kept unjudged. `context_limit=None` sends the full text.
        """
        all_doc_ids = set([res.doc_id for _, results in collection_to_results.items() for res in results])
        if len(all_doc_ids) > self._config.semantic_filter_max_candidate_docs:
            num_docs = len(all_doc_ids)
            limit = self._config.semantic_filter_max_candidate_docs
            raise Exception(f"Semantic filter candidate selection returned too many unique documents: {num_docs} > {limit}")

        texts = [(doc_id, self._document_map.get(doc_id, "") or "") for doc_id in all_doc_ids]
        if self._context_limit:
            budget = self._judge_doc_token_budget(predicate)
            texts = [(doc_id, self._truncate_doc_for_judge(t, budget)) for doc_id, t in texts]

        workers = max(min(self._config.semantic_filter_max_workers, len(all_doc_ids)), 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(
                lambda tup: self._judge_one(predicate, *tup),
                texts,
            ))

        kept_doc_ids, rejected_doc_ids = set(), set()
        for doc_id, verdict in verdicts:
            if verdict:
                kept_doc_ids.add(doc_id)
            else:
                rejected_doc_ids.add(doc_id)

        final_collection_to_results: dict[str, list[SearchResult]] = {}
        for collection, results in collection_to_results.items():
            final_collection_to_results[collection] = []
            for result in results:
                if result.doc_id in kept_doc_ids:
                    final_collection_to_results[collection].append(result)

        return kept_doc_ids, rejected_doc_ids, final_collection_to_results

    def __call__(
        self,
        collections: list[str],
        predicate: str,
        search_str: str | None = None,
        top_k: int | None = None,
        pattern: str | None = None,
        limit: int | None = None,
        metadata_filter: dict | None = None,
    ) -> dict:
        tool_kwargs = {
            "collections": collections,
            "predicate": predicate,
            "search_str": search_str,
            "top_k": top_k,
            "pattern": pattern,
            "limit": limit,
            "metadata_filter": metadata_filter,
        }

        if not predicate or not str(predicate).strip():
            return self._error("semantic_filter error: `predicate` is required.", tool_kwargs)

        if search_str is not None and top_k is None:
            return self._error(
                "semantic_filter error: search_str requires top_k (it is the query for the top_k vector search).",
                tool_kwargs,
            )
        if limit is not None and pattern is None:
            return self._error(
                "semantic_filter error: limit requires pattern (it is the limit for the grep expression).",
                tool_kwargs,
            )

        # get mapping from collection -> candidate chunk results
        if top_k is not None:
            try:
                collection_to_results = self._vector_search_selection(collections, search_str or predicate, top_k, metadata_filter)
            except Exception as e:
                return self._error(f"semantic_filter error while executing vector search: {e}", tool_kwargs)

        elif pattern is not None:
            try:
                collection_to_results = self._grep_selection(collections, pattern, limit, metadata_filter)
            except Exception as e:
                return self._error(f"semantic_filter error while executing grep: {e}", tool_kwargs)

        else:
            try:
                assert isinstance(metadata_filter, dict), "metadata filter needs to be a dict"
                collection_to_results = self._metadata_only_selection(collections, metadata_filter)
            except Exception as e:
                return self._error(f"semantic_filter error while executing metadata: {e}", tool_kwargs)

        # filter documents using judge model
        try:
            kept_doc_ids, rejected_doc_ids, collection_to_results = self._filter_docs(predicate, collection_to_results)
        except Exception as e:
            return self._error(f"semantic_filter error while filtering documents: {e}", tool_kwargs)

        # return results to the agent
        result = {
            "tool": self.name,
            "tool_kwargs": tool_kwargs,
            "results": collection_to_results,
            "kept_doc_ids": kept_doc_ids,
            "rejected_doc_ids": rejected_doc_ids,
        }

        return ResultSet(result)


# TODO: stream progress back to the agent and enable it to interrupt / change model if cost is too high?
# NOTE: would need to have async tool calls
class SemanticMapTool(Tool):
    name = "semantic_map"
    doc: str = _PROMPTS["sem_map_tool"]

    # NOTE: _JUDGE_USER_WRAPPER captures user prompt from `_judge_one()`
    #       _JUDGE_TRUNC_WRAPPER used in `_truncate_doc_for_judge()`
    _SEM_MAP_JUDGE_PROMPT = _PROMPTS["sem_map_judge"]
    _JUDGE_USER_WRAPPER = "Document:\n\n\nMetadata:\n\n\nFields:\n\n"
    _JUDGE_TRUNC_MARKER = "\n…(truncated to fit judge context)"

    def __init__(
        self,
        chroma_client: ClientAPI,
        llm_client: LLMClient,
        document_map: DocumentMap,
        config: BootstrapConfig | EnrichConfig,
        model: str,
        *,
        usage_key: str = "default",
    ) -> None:
        self._chroma_client = chroma_client
        self._llm_client = llm_client
        self._document_map = document_map
        self._config = config
        self._model = model
        self._usage_key = usage_key

        # resolve the model's context limit
        self._context_limit = match_model_entry(self._model, llm_client.config.llm_context_limits)

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse the JSON dictionary from the judge reply. The prompt asks for a bare JSON object and that
        is what the judge models return, so that is tried first; a fenced ```json block and a dictionary
        embedded in surrounding prose (first `{` to the matching last `}`) are accepted as fallbacks."""
        text = (text or "").strip()
        if not text:
            return {"map_error": "empty output"}
        candidates = [text]
        candidates += [body.strip() for lang, body in _FENCE_RE.findall(text) if lang.lower() in ("json", "")]
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            candidates.append(text[start : end + 1])
        last_error = "no JSON object in output"
        for body in candidates:
            try:
                obj = json.loads(body)
            except json.JSONDecodeError as e:
                last_error = str(e)
                continue
            if isinstance(obj, dict):
                return _sanitize_fields(obj)
            last_error = f"expected a JSON object, got {type(obj).__name__}"
        return {"map_error": last_error}

    def _error(self, msg: str, tool_kwargs: dict) -> dict:
        return {"tool": self.name, "tool_kwargs": tool_kwargs, "error": msg}

    def _truncate_doc_for_judge(self, text: str, budget_tokens: int) -> str:
        """Head-truncate `text` to `budget_tokens` (marker included), returning (text, was_truncated)."""
        token_est = estimate_tokens(text)
        if token_est > budget_tokens:
            ratio = budget_tokens / token_est
            return text[:int(ratio * len(text))] + self._JUDGE_TRUNC_MARKER
        return text

    def _judge_doc_token_budget(self, fields: list[dict]) -> int:
        """Max tokens of document text that fit in one judge request under `context_limit` (tokens):
        the fixed overhead (prompt scaffolding, counted via `estimate_tokens`) and the judge's output
        headroom (`max_output_tokens`) are subtracted, then the remaining token budget is returned
        with a safety factor. Returns 0 when the fixed overhead alone already exceeds the limit."""
        assert isinstance(self._context_limit, int)
        fields_str = ""
        for field in fields:
            fields_str += f"  - {field['name']} ({field['type']}): {field['desc']}\n"
        overhead_tokens = self._config.semantic_map_max_output_tokens + estimate_tokens(
            self._SEM_MAP_JUDGE_PROMPT + self._JUDGE_USER_WRAPPER + fields_str
        )
        budget_tokens = self._context_limit * self._config.semantic_map_context_frac - overhead_tokens
        return max(0, int(budget_tokens))

    def _judge_one(self, fields: list[dict], chunk_id: str, doc_id: str, metadata: dict, item_text: str, token_budget: int | None) -> tuple[str, str, str, dict, str | None]:
        """Judge one item (a chunk, or a document for doc_level). `metadata` is shown to the judge next to the
        text so fields can be derived from ids / earlier maps; its (variable) size is charged against the
        document's token budget before the text is truncated to fit the judge's context."""
        fields_str = ""
        for field in fields:
            fields_str += f"  - {field['name']} ({field['type']}): {field['desc']}\n"
        metadata_str = json.dumps(metadata, default=str)
        if token_budget is not None:
            item_text = self._truncate_doc_for_judge(item_text, max(0, token_budget - estimate_tokens(metadata_str)))
        user = f"Document:\n{item_text}\n\nMetadata:\n{metadata_str}\n\nFields:\n{fields_str}\n"
        try:
            messages = [
                {"role": "system", "content": self._SEM_MAP_JUDGE_PROMPT},
                {"role": "user", "content": user},
            ]
            resp = self._llm_client.call(
                messages=messages,
                temperature=0.0,
                model=self._model,
                call_site="semmap",
                provider_order=self._config.semantic_map_provider_order,
                max_output_tokens=self._config.semantic_map_max_output_tokens,
                disable_reasoning=self._config.semantic_map_disable_reasoning,
                usage_key=self._usage_key,
                timeout_s=self._config.request_timeout_s,
            )
        except Exception:
            return chunk_id, doc_id, item_text, {}, "LLM call failed."
        return chunk_id, doc_id, item_text, self._parse_json(resp.text), None

    def __call__(self, collections: list[str], metadata_fields: list[dict], doc_level: bool = False) -> dict:
        metadata_fields = normalize_metadata_fields(metadata_fields)
        tool_kwargs = {
            "collections": collections,
            "metadata_fields": metadata_fields,
            "doc_level": doc_level,
        }

        if (err := validate_metadata_fields(metadata_fields)) is not None:
            return self._error(f"semantic_map error: {err}", tool_kwargs)

        outputs: dict[str, list[MapResult]] = {}
        doc_cache: dict[str, tuple[dict, str | None]] = {}  # doc_level only: doc_id -> (judged fields, error)
        token_budget = self._judge_doc_token_budget(metadata_fields) if self._context_limit else None
        workers = max(1, self._config.semantic_map_max_workers)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name in collections:
                # check if collection is small enough for map
                c = self._chroma_client.get_collection(name)
                num_chunks = c.count()
                if num_chunks > self._config.semantic_map_max_candidate_chunks:
                    outputs[name] = [MapResult(
                        samples=[],
                        error=f"Collection {name} with {num_chunks} chunks is too large for semantic map (max allowed chunks: {self._config.semantic_map_max_candidate_chunks})",
                    )]
                    continue

                # TODO: this current paging approach is going to have tail latency within every batch from `_iter_get`;
                # a streaming approach where workers constantly pull items from a queue would be faster
                outputs[name] = []
                include = ["metadatas"] if doc_level else ["metadatas", "documents"]
                for page in _iter_get(c, include=include):
                    if doc_level:
                        # one judgement per document, on the full document text plus the metadata its
                        # chunks share; every chunk of the document then receives the same fields
                        doc_meta = _doc_level_metadata(page)
                        todo = [(doc_id, doc_id, meta, self._document_map.get(doc_id, "") or "", token_budget) for doc_id, meta in doc_meta.items() if doc_id not in doc_cache]
                        judged = list(pool.map(lambda e: self._judge_one(metadata_fields, *e), todo, buffersize=workers))
                        for _, doc_id, _, fields, error in judged:
                            doc_cache[doc_id] = (fields, error)
                        page_out = []
                        for chunk_id, meta in zip(page["ids"], page["metadatas"], strict=True):
                            doc_id = meta["doc_id"]
                            text = self._document_map.get(doc_id)
                            fields, error = doc_cache[doc_id]
                            page_out.append((chunk_id, doc_id, text, fields, error))

                    else:
                        elts = [
                            (chunk_id, (meta or {}).get("doc_id", "?"), _chunk_metadata(chunk_id, meta), text, token_budget)
                            for chunk_id, meta, text in zip(page["ids"], page["metadatas"], page["documents"], strict=True)
                        ]
                        page_out = list(pool.map(lambda e: self._judge_one(metadata_fields, *e), elts, buffersize=workers))

                    # NOTE: .update() merges new metadata fields with existing ones
                    # write mapped results back to collection
                    map_samples, chunk_ids, new_metadatas = [], [], []
                    for chunk_id, doc_id, text, fields, error in page_out:
                        chunk_ids.append(chunk_id)
                        new_metadatas.append(fields)
                        map_samples.append(MapSample(chunk_id, doc_id, text, fields, error))
                    c.update(ids=chunk_ids, metadatas=new_metadatas) # type: ignore
                    outputs[name].append(MapResult(samples=map_samples, error=None))

                # persist the new fields in the collection's schema ("  - name (type): desc" lines)
                _persist_fields(c, metadata_fields)

        # return results to the agent
        return {"tool": self.name, "tool_kwargs": tool_kwargs, "results": outputs}


class MapTool(Tool):
    name = "map_collections"
    doc: str = _PROMPTS["map_tool"]

    def __init__(
        self,
        chroma_client: ClientAPI,
        document_map: DocumentMap,
        config: BootstrapConfig | EnrichConfig,
    ) -> None:
        self._chroma_client = chroma_client
        self._document_map = document_map
        self._config = config

    def _error(self, msg: str, tool_kwargs: dict) -> dict:
        return {"tool": self.name, "tool_kwargs": tool_kwargs, "error": msg}

    def __call__(
        self,
        collections: list[str],
        map_fn: Callable,
        fields: list[dict],
        doc_level: bool = False,
    ) -> dict:
        fields = normalize_metadata_fields(fields)
        tool_kwargs = {
            "collections": collections,
            "map_fn": str(map_fn),
            "fields": fields,
            "doc_level": doc_level,
        }
        if (err := validate_metadata_fields(fields)) is not None:
            return self._error(f"map error: {err}", tool_kwargs)
        declared = {f["name"] for f in fields}

        outputs: dict[str, list[MapResult]] = {}
        doc_cache: dict[str, tuple[dict, str | None]] = {}  # doc_level only: doc_id -> (judged fields, error)
        undeclared: set[str] = set()  # keys map_fn produced that were not declared in `fields`
        workers = max(1, self._config.map_max_workers)

        # sandbox-defined functions take *args (an extra positional is ignored), but a plain Python callable
        # written for the old one-argument contract would raise, so pass `metadata` only to a fn that takes it
        two_args = _map_fn_takes_metadata(map_fn)

        def safe_map_fn(text, metadata) -> tuple[dict, str | None]:
            try:
                return _sanitize_fields(map_fn(text, metadata) if two_args else map_fn(text)), None
            except Exception as e:
                return {}, str(e)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name in collections:
                outputs[name] = []
                c = self._chroma_client.get_collection(name)
                include = ["metadatas"] if doc_level else ["metadatas", "documents"]
                for page in _iter_get(c, include=include):
                    if doc_level:
                        # one map_fn call per document (full document text + the metadata its chunks share);
                        # every chunk of the document then receives the same fields
                        doc_meta = _doc_level_metadata(page)
                        todo = [(doc_id, meta, self._document_map.get(doc_id, "") or "") for doc_id, meta in doc_meta.items() if doc_id not in doc_cache]
                        for doc_id, (produced, error) in pool.map(lambda e: (e[0], safe_map_fn(e[2], e[1])), todo, buffersize=workers): # type: ignore
                            doc_cache[doc_id] = (produced, error)

                        page_out = []
                        for chunk_id, meta in zip(page["ids"], page["metadatas"], strict=True):
                            doc_id = meta["doc_id"]
                            text = self._document_map.get(doc_id)
                            output_fields, error = doc_cache[doc_id]
                            page_out.append((chunk_id, doc_id, text, output_fields, error))

                    else:
                        elts = [(chunk_id, (meta or {}).get("doc_id", "?"), _chunk_metadata(chunk_id, meta), text) for chunk_id, meta, text in zip(page["ids"], page["metadatas"], page["documents"], strict=True)]
                        page_out = [(chunk_id, doc_id, text, out[0], out[1]) for chunk_id, doc_id, text, out in pool.map(lambda e: (e[0], e[1], e[3], safe_map_fn(e[3], e[2])), elts, buffersize=workers)]

                    # NOTE: .update() merges new metadata fields with existing ones
                    # write mapped results back to collection
                    map_samples, chunk_ids, new_metadatas = [], [], []
                    for chunk_id, doc_id, text, output_fields, error in page_out:
                        chunk_ids.append(chunk_id)
                        new_metadatas.append(output_fields)
                        map_samples.append(MapSample(chunk_id, doc_id, text, output_fields, error))
                    c.update(ids=chunk_ids, metadatas=new_metadatas) # type: ignore
                    outputs[name].append(MapResult(samples=map_samples, error=None))

                    # update the set of fields which were undeclared but generated by the map
                    for _, _, _, produced, _ in page_out:
                        undeclared.update(k for k in produced if k not in declared and k != "map_error")

                # persist the declared fields in the collection's schema ("  - name (type): desc" lines)
                _persist_fields(c, fields)

        # return results to the agent
        out: dict = {"tool": self.name, "tool_kwargs": tool_kwargs, "results": outputs}
        if undeclared:
            out["warning"] = f"map_fn produced fields not declared in `fields` (they were written but not added to the schema): {sorted(undeclared)}"

        return out


# ---- collection management tools (Bootstrap / Enrich agents) ----------------------------------------
class CollectionToolError(Exception):
    """A user-facing failure inside a collection tool (rendered as the tool's `error`)."""


@dataclass
class CopyStats:
    n_requested: int = 0  # chunk ids asked for (after de-duplication)
    n_new: int = 0  # chunks that were not already in the destination
    n_existing: int = 0  # chunks that were already in the destination (re-upserted, no duplicates)
    n_missing: int = 0  # requested ids not found in their source collection
    n_docs: int = 0  # distinct doc_ids among the copied chunks
    sources: dict[str, int] | None = None  # source collection -> chunks copied from it


def _describe_results(results) -> dict | None:
    """JSON-friendly summary of a `results` argument for tool_kwargs / traces."""
    if results is None:
        return None
    if not isinstance(results, dict) or "results" not in results:
        return {"invalid": repr(results)[:200]}
    rs = results if isinstance(results, ResultSet) else ResultSet(results)
    return {**rs.provenance(), "num_docs": len(rs.doc_ids), "collections": list(rs.results.keys())}


class _CollectionToolBase(Tool):
    """Shared machinery for the collection tools: name validation, protection of the base corpus, and
    the batched server-side chunk copy (`_copy`) that create/add/copy/merge are built on."""

    def __init__(
        self,
        chroma_client: ClientAPI,
        base_collection_name: str,
        *,
        agent_id: str = "",
        agent_type: str = "",
        max_copy_chunks: int = _DEFAULT_MAX_COPY_CHUNKS,
        max_workers: int = 16,
        batch_size: int = _COPY_BATCH_SIZE,
    ) -> None:
        self._chroma_client = chroma_client
        self._base_collection_name = base_collection_name
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._max_copy_chunks = max_copy_chunks
        self._max_workers = max(1, max_workers)
        self._batch_size = max(1, batch_size)

    def _error(self, msg: str, tool_kwargs: dict) -> dict:
        return {"tool": self.name, "tool_kwargs": tool_kwargs, "error": msg}

    def _validate_name(self, name: str) -> None:
        if not isinstance(name, str) or not _COLLECTION_NAME_RE.match(name):
            raise CollectionToolError(
                f"invalid collection name {name!r}: use 3-512 characters from [a-zA-Z0-9._-], starting and ending with a letter or digit"
            )
        if name == self._base_collection_name:
            raise CollectionToolError(f"{name!r} is the base collection")

    def _get(self, name: str):
        try:
            return self._chroma_client.get_collection(name)
        except NotFoundError as e:
            raise CollectionToolError(f"collection {name!r} does not exist") from e

    def _get_managed(self, name: str):
        """A collection this agent may modify: exists, is not the base corpus, and carries the managed flag."""
        if name == self._base_collection_name:
            raise CollectionToolError(f"{name!r} is the base collection and cannot be modified")
        c = self._get(name)
        if not is_managed_collection(c):
            raise CollectionToolError(f"collection {name!r} is not an agent-managed collection")
        return c

    def _create(self, name: str, description: str, fields: str = ""):
        self._validate_name(name)
        if not isinstance(description, str) or not description.strip():
            raise CollectionToolError("`description` must be a non-empty string")
        try:
            return self._chroma_client.create_collection(
                name=name, metadata=managed_collection_metadata(description.strip(), fields=fields, created_by=self._agent_id, created_by_agent_type=self._agent_type)
            )
        except Exception as e:  # chroma's "already exists" error class varies by client/server version
            if "exist" in str(e).lower():
                raise CollectionToolError(f"collection {name!r} already exists") from e
            raise

    @staticmethod
    def _sources(results: ResultSet | list[ResultSet] | None, from_collections: list[str] | None, metadata_filter: dict | None) -> list[tuple[str, list[str] | None]]:
        """Normalize the two ways of naming chunks to copy into [(source collection, ids | None)];
        `None` ids means "everything in the source matching `metadata_filter`".
        
        Returns a list of tuples where each tuple is:
        - the name of the collection to copy from
        - the list of chunk ids to copy from that collection OR None (means that the entire collection + metadata_filter should be copied by _create_and_copy)
        """
        sources: list[tuple[str, list[str] | None]] = []
        if results is not None:
            results = results if isinstance(results, (list, tuple)) else [results]
            combined: ResultSet | None = None
            for result in results:
                if not isinstance(result, dict) or "results" not in result:
                    raise CollectionToolError("`results` must be the value returned by search_corpus / grep_corpus (or a combination of them)")
                rs = result if isinstance(result, ResultSet) else ResultSet(result)
                if rs.get("error"):
                    raise CollectionToolError(f"`results` carries an error from {rs.get('tool')}: {rs['error']}")
                combined = rs if combined is None else (combined | rs)
            # .by_source() returns mapping {collection --> chunk_ids to copy}
            sources.extend((combined or ResultSet()).by_source().items())
        if from_collections:
            if isinstance(from_collections, str):
                from_collections = [from_collections]
            sources.extend((name, None) for name in from_collections)
        elif metadata_filter is not None:
            raise CollectionToolError("`metadata_filter` only applies together with `from_collections`")
        if not sources:
            raise CollectionToolError("nothing to add: pass `results` and/or `from_collections`")
        return sources

    def _copy(self, dst_name: str, sources: list[tuple[str, list[str] | None]], metadata_filter: dict | None, action: str) -> CopyStats:
        """Copy chunks (id, embedding, document, metadata) from each source into `dst_name`, batched and
        in parallel. Upserts are keyed by chunk_id so re-adding is idempotent. The destination inherits the
        sources' schema (`fields`) and the call is appended to its `actions` log."""
        dst = self._get_managed(dst_name)
        stats = CopyStats(sources={})
        docs: set[str] = set()
        fields = [(dst.metadata or {}).get("fields")]
        for src_name, ids in sources:
            if src_name == dst_name:
                if ids is None:
                    raise CollectionToolError(f"cannot add {dst_name!r} to itself")
                # results that came from the destination are, by definition, already in it
                ids = list(dict.fromkeys(ids))
                stats.n_requested += len(ids)
                stats.n_existing += len(ids)
                stats.sources[src_name] = stats.sources.get(src_name, 0) + len(ids)  # type: ignore[union-attr]
                continue
            src = self._get(src_name)
            if ids is None:
                get_kwargs: dict = {"include": [], "limit": self._max_copy_chunks + 1}
                if metadata_filter is not None:
                    get_kwargs["where"] = metadata_filter
                ids = list(src.get(**get_kwargs)["ids"])
            ids = list(dict.fromkeys(ids))
            if len(ids) > self._max_copy_chunks:
                raise CollectionToolError(
                    f"{src_name!r} selects more than {self._max_copy_chunks} chunks; narrow the `metadata_filter` / results (or raise `max_copy_chunks`)"
                )
            stats.n_requested += len(ids)
            batches = [ids[i : i + self._batch_size] for i in range(0, len(ids), self._batch_size)]

            def copy_batch(batch: list[str]) -> tuple[int, int, int, set[str]]:
                got = src.get(ids=batch, include=["embeddings", "documents", "metadatas"])
                found = list(got["ids"])
                if not found:
                    return 0, 0, len(batch), set()
                existing = set(dst.get(ids=found, include=[])["ids"])
                dst.upsert(ids=found, embeddings=got["embeddings"], documents=got["documents"], metadatas=got["metadatas"])  # type: ignore
                batch_docs = {str(m.get("doc_id")) for m in (got.get("metadatas") or []) if m}
                return len(found) - len(existing), len(existing), len(batch) - len(found), batch_docs

            n_copied = 0
            # `buffersize` keeps only max_workers batches in flight (a plain map submits every batch up front)
            with ThreadPoolExecutor(max_workers=min(self._max_workers, max(1, len(batches)))) as pool:
                results = pool.map(copy_batch, batches, buffersize=self._max_workers)
                for page, (n_new, n_existing, n_missing, batch_docs) in enumerate(results, start=1):
                    stats.n_new += n_new
                    stats.n_existing += n_existing
                    stats.n_missing += n_missing
                    n_copied += n_new + n_existing
                    docs |= batch_docs
                    if page % _GC_EVERY_PAGES == 0:
                        gc.collect()
            assert stats.sources is not None
            stats.sources[src_name] = stats.sources.get(src_name, 0) + n_copied
            fields.append((src.metadata or {}).get("fields"))
        stats.n_docs = len(docs)

        # inherit the sources' schema + record the action (re-read metadata: concurrent tools may have changed it)
        dst = self._get(dst_name)
        dst.modify(metadata={**(dst.metadata or {}), "fields": merge_metadata_fields(*fields)})
        append_action(self._get(dst_name), action)
        return stats

    def _create_and_copy(self, name: str, description: str, sources: list, metadata_filter: dict | None, action: str) -> CopyStats | None:
        """Create `name` and populate it from `sources`. If populating fails (e.g. the copy cap), the
        just-created collection is deleted again so no empty collection is left behind."""
        self._create(name, description)
        if not sources:
            return None
        try:
            return self._copy(name, sources, metadata_filter, action)
        except CollectionToolError:
            try:
                self._chroma_client.delete_collection(name)
            except Exception:  # noqa: BLE001 — the original error is the one worth reporting
                pass
            raise

    def _summary(self, name: str, tool_kwargs: dict, stats: CopyStats | None) -> dict:
        c = self._get(name)
        return {
            "tool": self.name,
            "tool_kwargs": tool_kwargs,
            "collection": name,
            "num_chunks": c.count(),
            "fields": (c.metadata or {}).get("fields", ""),
            "stats": asdict(stats) if stats is not None else None,
        }


class CreateCollectionTool(_CollectionToolBase):
    name = "create_collection"
    doc = _PROMPTS["create_collection"]

    def __call__(
        self,
        name: str,
        description: str,
        results: ResultSet | list[ResultSet] | None = None,
        *,
        from_collections: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> dict:
        tool_kwargs = {
            "name": name,
            "description": description,
            "results": _describe_results(results) if not isinstance(results, (list, tuple)) else [_describe_results(r) for r in results],
            "from_collections": from_collections,
            "metadata_filter": metadata_filter,
        }
        try:
            populate = results is not None or from_collections or metadata_filter is not None
            sources = self._sources(results, from_collections, metadata_filter) if populate else []
            action = f"create_collection(name={name!r}, results={tool_kwargs['results']}, from_collections={from_collections}, metadata_filter={metadata_filter})"
            stats = self._create_and_copy(name, description, sources, metadata_filter, action)
            return self._summary(name, tool_kwargs, stats)
        except CollectionToolError as e:
            return self._error(f"create_collection error: {e}", tool_kwargs)


class AddToCollectionTool(_CollectionToolBase):
    name = "add_to_collection"
    doc = _PROMPTS["add_to_collection"]

    def __call__(
        self,
        name: str,
        results: ResultSet | list[ResultSet] | None = None,
        *,
        from_collections: list[str] | None = None,
        metadata_filter: dict | None = None,
    ) -> dict:
        tool_kwargs = {
            "name": name,
            "results": _describe_results(results) if not isinstance(results, (list, tuple)) else [_describe_results(r) for r in results],
            "from_collections": from_collections,
            "metadata_filter": metadata_filter,
        }
        try:
            sources = self._sources(results, from_collections, metadata_filter)
            action = f"add_to_collection(results={tool_kwargs['results']}, from_collections={from_collections}, metadata_filter={metadata_filter})"
            stats = self._copy(name, sources, metadata_filter, action)
            return self._summary(name, tool_kwargs, stats)
        except CollectionToolError as e:
            return self._error(f"add_to_collection error: {e}", tool_kwargs)


class DeleteCollectionTool(_CollectionToolBase):
    name = "delete_collection"
    doc = _PROMPTS["delete_collection"]

    def __call__(self, name: str) -> dict:
        tool_kwargs = {"name": name}
        try:
            self._get_managed(name)
            self._chroma_client.delete_collection(name)
            return {"tool": self.name, "tool_kwargs": tool_kwargs, "collection": name, "deleted": True}
        except CollectionToolError as e:
            return self._error(f"delete_collection error: {e}", tool_kwargs)


class CopyCollectionTool(_CollectionToolBase):
    name = "copy_collection"
    doc = _PROMPTS["copy_collection"]

    def __call__(self, src: str, dst: str, description: str | None = None, *, metadata_filter: dict | None = None) -> dict:
        tool_kwargs = {"src": src, "dst": dst, "description": description, "metadata_filter": metadata_filter}
        try:
            source = self._get(src)
            if description is None:
                src_desc = (source.metadata or {}).get("description") or ""
                description = f"Copy of {src!r}" + (f" ({src_desc})" if src_desc else "") + (f" filtered by {metadata_filter}" if metadata_filter else "")
            action = f"copy_collection(src={src!r}, metadata_filter={metadata_filter})"
            stats = self._create_and_copy(dst, description, [(src, None)], metadata_filter, action)
            return self._summary(dst, tool_kwargs, stats)
        except CollectionToolError as e:
            return self._error(f"copy_collection error: {e}", tool_kwargs)


class MergeCollectionsTool(_CollectionToolBase):
    name = "merge_collections"
    doc = _PROMPTS["merge_collections"]

    def __call__(self, sources: list[str], dst: str, description: str, *, metadata_filter: dict | None = None) -> dict:
        tool_kwargs = {"sources": sources, "dst": dst, "description": description, "metadata_filter": metadata_filter}
        try:
            if isinstance(sources, str) or not sources:
                raise CollectionToolError("`sources` must be a non-empty list of collection names")
            for name in sources:
                self._get(name)
            action = f"merge_collections(sources={list(sources)!r}, metadata_filter={metadata_filter})"
            stats = self._create_and_copy(dst, description, [(name, None) for name in sources], metadata_filter, action)
            return self._summary(dst, tool_kwargs, stats)
        except CollectionToolError as e:
            return self._error(f"merge_collections error: {e}", tool_kwargs)


class ListCollectionsTool(_CollectionToolBase):
    name = "list_collections"
    doc = _PROMPTS["list_collections"]

    # chromadb imposes a limit of 100 collections per-call to list_collections()
    _LIST_LIMIT = 100

    def __call__(self) -> dict:
        summaries: list[dict] = []
        offset = 0
        while True:
            page = list(self._chroma_client.list_collections(limit=self._LIST_LIMIT, offset=offset))
            for c in page:
                is_base = c.name == self._base_collection_name
                if not is_base and not is_managed_collection(c):
                    continue
                meta = c.metadata or {}
                summaries.append({
                    "name": c.name,
                    "is_base": is_base,
                    "is_working_set": meta.get("is_working_set", ""),
                    "created_by_agent_type": meta.get("created_by_agent_type", ""),
                    "description": "the base collection." if is_base else meta.get("description", ""),
                    "num_chunks": c.count(),
                    "fields": meta.get("fields", ""),
                    "actions": [a for a in (meta.get("actions") or "").split(METADATA_LIST_DELIMITER) if a],
                })
            if len(page) < self._LIST_LIMIT:
                break
            offset += self._LIST_LIMIT
        summaries.sort(key=lambda d: (not d["is_base"], d["name"]))
        return {"tool": self.name, "tool_kwargs": {}, "collections": summaries}
