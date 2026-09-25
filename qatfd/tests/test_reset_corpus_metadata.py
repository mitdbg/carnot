"""engaging-scripts/reset_corpus_metadata.py: stray chunk-metadata keys are found through chroma.sqlite3 and
stripped through the client on a `-v1` corpus copy; an original base collection is only ever checked."""

from __future__ import annotations

import sys
from pathlib import Path

import chromadb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engaging-scripts"))
import reset_corpus_metadata as rcm  # noqa: E402

V1 = "officeqa-qwen-8b-v1"
BASE = "officeqa-qwen-8b"
N = 60


def _build(client, name: str, stray_schema: str = "") -> None:
    """A tiny `name` collection whose chunks carry the native keys plus stray `topic` / `is_table` keys on
    some rows, and whose `fields` schema has `stray_schema` merged in (as _persist_fields would do)."""
    c = client.create_collection(name, metadata={"fields": rcm.KNOWN[name][1] + stray_schema, "hnsw:num_threads": 2})
    ids, metas = [], []
    for i in range(N):
        m: dict = {"doc_id": f"2001_03_{i // 10}", "chunk_id": f"tb_{i}", "element_id": i % 10}
        if name == BASE:
            m.update(file_id="treasury_bulletin_2001_03", year="2001", month="03", page_id=i // 10, type="text")
        if i % 2 == 0:
            m["topic"] = "debt"
        if i % 3 == 0:
            m["is_table"] = True
        ids.append(f"tb_{i}")
        metas.append(m)
    c.add(ids=ids, embeddings=[[float(i), 1.0] for i in range(N)], documents=[f"text {i}" for i in range(N)], metadatas=metas)


STRAY_SCHEMA = "  - topic (str): agent-written\n  - is_table (bool): agent-written\n"


@pytest.fixture
def client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path))


def test_allowlist_is_exactly_the_v1_copies():
    assert rcm.STRIPPABLE and all(name.endswith("-v1") for name in rcm.STRIPPABLE)
    assert V1 in rcm.STRIPPABLE and BASE not in rcm.STRIPPABLE
    assert rcm.allowed_keys(V1) == frozenset(rcm.KEEP)
    assert {"doc_id", "chunk_id", "element_id", "year", "month", "page_id", "type", "file_id"} == rcm.allowed_keys(BASE)


def test_find_stray_keys_reads_the_store(client, tmp_path):
    _build(client, V1)
    stray = rcm.find_stray_keys(str(tmp_path), V1, rcm.allowed_keys(V1))
    assert len(stray) == len({i for i in range(N) if i % 2 == 0 or i % 3 == 0})
    assert stray["tb_0"] == sorted(["topic", "is_table"]) or set(stray["tb_0"]) == {"topic", "is_table"}
    assert stray["tb_3"] == ["is_table"]
    with pytest.raises(SystemExit):
        rcm.find_stray_keys(str(tmp_path), "no-such-collection", rcm.allowed_keys(V1))


def test_strips_v1_copy(client, tmp_path):
    _build(client, V1, STRAY_SCHEMA)
    assert rcm.reset_one(client, str(tmp_path), V1, check=False, batch=7, sample=10) is True

    assert rcm.find_stray_keys(str(tmp_path), V1, rcm.allowed_keys(V1)) == {}
    c = client.get_collection(V1)
    got = c.get(ids=["tb_0", "tb_3", "tb_5"], include=["metadatas", "documents"])
    assert all(set(m) == set(rcm.KEEP) for m in got["metadatas"])
    assert got["documents"] == ["text 0", "text 3", "text 5"]  # documents untouched
    assert c.count() == N
    assert c.metadata["fields"] == rcm.KNOWN[V1][1]  # schema back to canonical …
    assert c.metadata["hnsw:num_threads"] == 2  # … every other collection-metadata key kept
    assert c.get(where={"topic": "debt"}, include=[])["ids"] == []

    # idempotent: a clean collection is reported clean and left alone
    assert rcm.reset_one(client, str(tmp_path), V1, check=False) is True


def test_check_never_writes(client, tmp_path):
    _build(client, V1, STRAY_SCHEMA)
    assert rcm.reset_one(client, str(tmp_path), V1, check=True) is False
    assert "topic" in client.get_collection(V1).get(ids=["tb_0"], include=["metadatas"])["metadatas"][0]
    assert client.get_collection(V1).metadata["fields"].endswith(STRAY_SCHEMA)


def test_base_collection_is_only_checked(client, tmp_path):
    _build(client, BASE, STRAY_SCHEMA)
    assert rcm.reset_one(client, str(tmp_path), BASE, check=False) is False
    m = client.get_collection(BASE).get(ids=["tb_0"], include=["metadatas"])["metadatas"][0]
    assert m["topic"] == "debt" and m["year"] == "2001"  # nothing stripped, native keys intact
    assert client.get_collection(BASE).metadata["fields"].endswith(STRAY_SCHEMA)


def test_clean_base_collection_passes(client, tmp_path):
    c = client.create_collection(BASE, metadata={"fields": rcm.KNOWN[BASE][1]})
    c.add(ids=["a"], embeddings=[[0.0, 1.0]], metadatas=[{"doc_id": "d", "chunk_id": "a", "element_id": 0, "year": "2001"}])
    assert rcm.reset_one(client, str(tmp_path), BASE, check=False) is True


def test_unknown_collection_is_refused(client, tmp_path):
    with pytest.raises(SystemExit):
        rcm.reset_one(client, str(tmp_path), "ws_something", check=True)
