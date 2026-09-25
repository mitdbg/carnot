"""Unit tests for qatfd.merged_chroma — MergedClient / MergedCollection over sharded chroma.

Run from the qatfd dir: python3 -m pytest tests/test_merged_chroma.py

Uses two in-process PersistentClients on separate temp dirs as the two "shard servers" (separate
dirs => separate chroma systems, so they are as independent as two HTTP servers for routing).
"""

from __future__ import annotations

import chromadb
import pytest
from chromadb.errors import NotFoundError, UniqueConstraintError

from qatfd.merged_chroma import SHARD_KEY, MergedClient, MergedCollection, Shard

BASE = "corpus"
DIM = 2


def _fill(collection, prefix: str, n: int, x0: float) -> None:
    collection.add(
        ids=[f"{prefix}{i}" for i in range(n)],
        embeddings=[[x0 + i, 0.0] for i in range(n)],
        documents=[f"doc {prefix}{i}" for i in range(n)],
        metadatas=[{"doc_id": f"D{prefix}{i}", "type": "text"} for i in range(n)],
    )


@pytest.fixture
def shards(tmp_path):
    a = chromadb.PersistentClient(path=str(tmp_path / "a"))
    b = chromadb.PersistentClient(path=str(tmp_path / "b"))
    r0 = a.create_collection(f"{BASE}_r0", metadata={"fields": "doc_id,type"})
    r1 = b.create_collection(f"{BASE}_r1", metadata={"fields": "doc_id,type"})
    _fill(r0, "a", 3, 0.0)  # a0..a2 at x=0,1,2
    _fill(r1, "b", 3, 10.0)  # b0..b2 at x=10,11,12
    return a, b


@pytest.fixture
def client(shards):
    a, b = shards
    return MergedClient(BASE, [Shard(a, f"{BASE}_r0", "A"), Shard(b, f"{BASE}_r1", "B")])


# ---- base collection ------------------------------------------------------------------------


def test_missing_shard_fails_fast(shards):
    a, b = shards
    with pytest.raises(NotFoundError, match="_r9"):
        MergedClient(BASE, [Shard(a, f"{BASE}_r0"), Shard(b, f"{BASE}_r9")])


def test_base_collection_is_merged(client):
    base = client.get_collection(BASE)
    assert isinstance(base, MergedCollection)
    assert base.name == BASE
    assert base.count() == 6
    assert base.metadata["fields"] == "doc_id,type"


def test_query_merges_by_distance_and_keeps_embeddings(client):
    base = client.get_collection(BASE)
    r = base.query(
        query_embeddings=[[2.5, 0.0], [10.5, 0.0]],
        n_results=3,
        include=["metadatas", "documents", "distances", "embeddings"],
    )
    assert r["ids"][0] == ["a2", "a1", "a0"]
    assert r["ids"][1][:2] in (["b0", "b1"], ["b1", "b0"])
    assert r["distances"][0] == sorted(r["distances"][0])
    assert r["documents"][0] == ["doc a2", "doc a1", "doc a0"]
    assert r["metadatas"][0][0]["doc_id"] == "Da2"
    assert len(r["embeddings"][0]) == 3 and list(r["embeddings"][0][0]) == [2.0, 0.0]
    # distances only fetched for ranking are stripped when not requested
    r2 = base.query(query_embeddings=[[0.0, 0.0]], n_results=4, include=["documents"])
    assert "distances" not in r2 and r2["ids"][0] == ["a0", "a1", "a2", "b0"]


def test_get_concatenates_and_caps(client):
    base = client.get_collection(BASE)
    r = base.get(ids=["a1", "b2", "zz"], include=["documents", "embeddings"])
    assert sorted(r["ids"]) == ["a1", "b2"]
    assert len(r["embeddings"]) == 2
    assert len(base.get(limit=4, include=[])["ids"]) == 4
    assert base.get(where={"doc_id": "Db1"}, include=["documents"])["documents"] == ["doc b1"]


def test_update_and_delete_route_to_owner(client, shards):
    a, b = shards
    base = client.get_collection(BASE)
    base.update(ids=["a1", "b2", "nope"], metadatas=[{"tag": "x"}, {"tag": "y"}, {"tag": "z"}])
    assert a.get_collection(f"{BASE}_r0").get(ids=["a1"])["metadatas"][0]["tag"] == "x"
    assert b.get_collection(f"{BASE}_r1").get(ids=["b2"])["metadatas"][0]["tag"] == "y"
    base.delete(ids=["a0", "b0"])
    assert base.count() == 4
    base.modify(metadata={"fields": "doc_id,type,tag"})
    assert b.get_collection(f"{BASE}_r1").metadata["fields"] == "doc_id,type,tag"
    with pytest.raises(NotImplementedError):
        base.upsert(ids=["q"], embeddings=[[0.0, 0.0]])


# ---- routing + placement ----------------------------------------------------------------------


def test_create_places_on_least_loaded_shard(client, shards):
    a, b = shards
    c1 = client.create_collection("ws_1", metadata={"fields": "f"})
    c2 = client.create_collection("ws_2", metadata={"fields": "f"})
    c3 = client.create_collection("ws_3")  # no metadata: still valid because of the shard stamp
    assert [client.shard_of(n) for n in ("ws_1", "ws_2", "ws_3")] == [0, 1, 0]
    assert c1.metadata[SHARD_KEY] == 0 and c2.metadata[SHARD_KEY] == 1 and c3.metadata[SHARD_KEY] == 0
    assert {c.name for c in a.list_collections()} == {f"{BASE}_r0", "ws_1", "ws_3"}
    assert {c.name for c in b.list_collections()} == {f"{BASE}_r1", "ws_2"}
    with pytest.raises(UniqueConstraintError):
        client.create_collection("ws_2")
    with pytest.raises(UniqueConstraintError):
        client.create_collection(BASE)


def test_get_routes_and_discovers_external_collections(client, shards):
    a, b = shards
    # created behind the client's back on shard B: found by probing, then cached
    b.create_collection("external", metadata={"k": 1})
    c = client.get_collection("external")
    assert c.metadata == {"k": 1} and client.shard_of("external") == 1
    # raw shard collections stay reachable by name
    assert client.get_collection(f"{BASE}_r1").count() == 3
    with pytest.raises(NotFoundError, match="any shard"):
        client.get_collection("missing")
    # deleted behind our back -> stale cache entry is dropped and re-probed
    b.delete_collection("external")
    with pytest.raises(NotFoundError):
        client.get_collection("external")


def test_get_or_create_is_idempotent_and_writable(client):
    ws = client.get_or_create_collection(name="ws_agent")
    ws.upsert(ids=["a1"], embeddings=[[1.0, 0.0]], documents=["doc a1"], metadatas=[{"doc_id": "Da1"}])
    again = client.get_or_create_collection(name="ws_agent")
    assert again.count() == 1 and client.shard_of("ws_agent") == 0
    assert client.get_or_create_collection(name=BASE).name == BASE  # base passthrough
    ws.update(ids=["a1"], metadatas=[{"doc_id": "Da1", "score": 3}])
    assert client.get_collection("ws_agent").get(ids=["a1"])["metadatas"][0]["score"] == 3


def test_delete_collection(client):
    client.create_collection("tmp")
    client.delete_collection("tmp")
    with pytest.raises(NotFoundError):
        client.get_collection("tmp")
    with pytest.raises(ValueError):
        client.delete_collection(BASE)
    with pytest.raises(ValueError):
        client.delete_collection(f"{BASE}_r0")


# ---- list_collections pagination -------------------------------------------------------------


def test_list_collections_collapses_shards_and_paginates(client):
    for i in range(5):
        client.create_collection(f"ws_{i}")
    names = [c.name for c in client.list_collections()]
    assert names == sorted(names) and names == [BASE, "ws_0", "ws_1", "ws_2", "ws_3", "ws_4"]
    assert f"{BASE}_r0" not in names and isinstance(client.list_collections()[0], MergedCollection)
    assert client.count_collections() == 6
    assert [c.name for c in client.list_collections(limit=2, offset=1)] == ["ws_0", "ws_1"]
    assert client.list_collections(limit=2, offset=6) == []

    # the SearchAgent's paging loop: walk pages of `limit` until a short page.
    limit, offset, seen = 4, 0, []
    page = client.list_collections(limit=limit, offset=offset)
    while len(page) > 0:
        seen.extend(c.name for c in page)
        if len(page) < limit:
            break
        offset += limit
        page = client.list_collections(limit=limit, offset=offset)
    assert seen == names


def test_one_client_per_shard_is_a_precondition(shards):
    a, _ = shards
    a.create_collection(f"{BASE}_r1", metadata={"fields": "doc_id,type"})
    with pytest.raises(ValueError, match="own client"):
        MergedClient(BASE, [Shard(a, f"{BASE}_r0"), Shard(a, f"{BASE}_r1")])
