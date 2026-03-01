from __future__ import annotations

import os
import re
import json
import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_TOKENIZER_MODEL = "BAAI/bge-small-en-v1.5"


def normalize_title_slug(s: str) -> str:
    if not s:
        return "untitled"
    t = unicodedata.normalize("NFC", s).strip()
    from unidecode import unidecode
    t = unidecode(t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Za-z0-9 _\-.]", "", t).strip().replace(" ", "_")
    return t or "untitled"


def stable_entity_id(title: str, text: str) -> str:
    slug = normalize_title_slug(title)
    h = hashlib.sha1((title + "\n" + (text or "")).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}"


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON on line {idx}: {e}")


def build_tokenizer(model_name: str = DEFAULT_TOKENIZER_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=100000)


def chunk_by_tokens(text: str, tokenizer, chunk_size: int, overlap: int) -> List[str]:
    toks = tokenizer.encode(text, add_special_tokens=False)
    if not toks:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(toks), step):
        end = min(start + chunk_size, len(toks))
        sub = toks[start:end]
        if not sub:
            break
        chunk_text = tokenizer.decode(sub, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(toks):
            break

    return chunks


def prepare_quest_documents(
    jsonl_path: str,
    tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    index_first_512: bool = False,
    chunk_size: int = 512,
    overlap: int = 80,
    max_docs: int = None,
) -> Iterable[Dict[str, Any]]:
    tokenizer = build_tokenizer(tokenizer_model)
    filename = os.path.basename(jsonl_path)

    for idx, raw in enumerate(read_jsonl(jsonl_path)):
        if max_docs is not None and idx >= max_docs:
            logger.info(f"Reached limit of {max_docs} documents.")
            break
        if idx % 1000 == 0:
            logger.info(f"Processing document {idx}...")

        title = (raw.get("title") or "").strip() or "untitled"
        text = (raw.get("text") or raw.get("description") or "").strip()
        entity_id = stable_entity_id(title, text)

        if index_first_512:
            toks = tokenizer.encode(text, add_special_tokens=False)
            chunk_text_str = tokenizer.decode(toks[:chunk_size], skip_special_tokens=True).strip()
            if not chunk_text_str:
                chunk_text_str = title
            yield {
                "text": chunk_text_str,
                "metadata": {
                    "entity_id": entity_id,
                    "title": title,
                    "chunk_index": 0,
                    "n_chunks": 1,
                    "source": filename,
                },
            }
        else:
            chunks = chunk_by_tokens(text, tokenizer, chunk_size, overlap)
            if not chunks:
                chunks = [title]
            n_chunks = len(chunks)
            for ci, ch in enumerate(chunks):
                yield {
                    "text": ch,
                    "metadata": {
                        "entity_id": entity_id,
                        "chunk_id": f"{entity_id}__{ci:04d}",
                        "title": title,
                        "chunk_index": ci,
                        "n_chunks": n_chunks,
                        "source": filename,
                    },
                }


@dataclass
class QuestQuery:
    query: str
    docs: List[str]
    original_query: str
    scores: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


def _iter_jsonl_from_source(source: str):
    if source.startswith("http://") or source.startswith("https://"):
        import requests
        logger.info(f"Downloading QUEST queries from URL: {source}")
        resp = requests.get(source, stream=True)
        resp.raise_for_status()
        for line_num, line in enumerate(resp.iter_lines(decode_unicode=True), 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error on line {line_num} from URL {source}: {e}")
    else:
        if not os.path.exists(source):
            logger.warning(f"QUEST query file not found: {source}")
            return
        logger.info(f"Loading QUEST queries from file: {source}")
        with open(source, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error on line {line_num} in file {source}: {e}")


def prepare_quest_queries(source: str) -> List[QuestQuery]:
    if not source:
        raise ValueError("No source provided for QUEST queries.")
    queries: List[QuestQuery] = []
    for data in _iter_jsonl_from_source(source):
        queries.append(QuestQuery(
            query=data.get("query", ""),
            docs=data.get("docs", []) or [],
            original_query=data.get("original_query", ""),
            scores=data.get("scores"),
            metadata=data.get("metadata"),
        ))
    logger.info(f"Loaded {len(queries)} QUEST queries from {source}")
    return queries
