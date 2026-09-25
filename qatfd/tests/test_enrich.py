"""Unit tests for `SearchAgentSystem._enrich()`: the EnrichAgent must be handed the base collection
summary, the existing agent-created collections, and the most recent questions (bounded by
`enrich_config.max_previous_queries`, oldest first). The agent itself is faked; chroma is two
PersistentClient shards behind a MergedClient.

Run from the qatfd dir: python3 -m pytest tests/test_enrich.py
"""

from __future__ import annotations

import asyncio
from collections import deque
from threading import Lock
from types import SimpleNamespace

import chromadb
import pytest

from qatfd.config import EnrichConfig
from qatfd.merged_chroma import MergedClient, Shard
from qatfd.systems.search_agent import SearchAgentSystem
from qatfd.tools import CreateCollectionTool
from qatfd.types import Question

BASE = "corpus"


@pytest.fixture
def client(tmp_path):
    a = chromadb.PersistentClient(path=str(tmp_path / "a"))
    b = chromadb.PersistentClient(path=str(tmp_path / "b"))
    for cl, prefix, x0 in ((a, "a", 0.0), (b, "b", 10.0)):
        c = cl.create_collection(f"{BASE}_r{0 if prefix == 'a' else 1}", metadata={"fields": "  - doc_id (str): id\n"})
        c.add(
            ids=[f"{prefix}{i}" for i in range(4)],
            embeddings=[[x0 + i, 0.0] for i in range(4)],
            documents=[f"{prefix} text {i}" for i in range(4)],
            metadatas=[{"doc_id": f"D{prefix}{i}"} for i in range(4)],
        )
    return MergedClient(BASE, [Shard(a, f"{BASE}_r0"), Shard(b, f"{BASE}_r1")])


class _FakeAgent:
    def __init__(self):
        self.inputs: list[str] = []

    async def call(self, ctx, input, **_):
        self.inputs.append(input)
        return {"done": True}


class _StubSystem(SearchAgentSystem):
    """Real `_enrich` / history bookkeeping over a faked EnrichAgent and a minimal retrieve config."""

    def __init__(self, max_previous_queries: int):
        # skip RetrieveComputeSystem.__init__ (it wants full retrieve/compute/inference configs)
        enrich_config = EnrichConfig(name="enrich", agent_id="enrich", max_previous_queries=max_previous_queries)
        self.retrieve_config = SimpleNamespace(enrich_config=enrich_config, enrich_working_sets="after", enrich_query_batch_size=2)
        self._question_lock = Lock()
        self._question_num = None
        self._question_history = deque(maxlen=max_previous_queries)
        self.agent = _FakeAgent()

    def _build_enrich_agent(self, ctx, resources):
        return self.agent


def _ctx():
    return SimpleNamespace(config=SimpleNamespace(storage=SimpleNamespace(collection_name=BASE)), llm_client=object())


def _resources(client):
    return SimpleNamespace(chroma_client=client, document_map={}, corpus_details=None)


def test_enrich_prompt_contains_collections_and_recent_queries(client):
    system = _StubSystem(max_previous_queries=3)
    CreateCollectionTool(client, BASE, agent_id="bootstrap")("ws_fed", "Fed policy chunks", from_collections=[BASE])
    for i in range(5):
        system._question_history.append(f"question {i}")

    asyncio.run(system._enrich(_ctx(), _resources(client)))
    assert len(system.agent.inputs) == 1
    text = system.agent.inputs[0]
    # base collection summary
    assert f'===== Base Collection (name="{BASE}") =====' in text and "Description: the base collection." in text
    assert "Number of chunks: 8 (the entire corpus)" in text and "  - doc_id (str): id" in text
    # existing collections
    assert "Here are summaries of the existing (agent-created) collections:" in text
    assert '===== Collection (name="ws_fed") =====' in text and "Description: Fed policy chunks" in text
    assert "create_collection(" in text
    # only the last 3 queries, oldest first, numbered
    assert "Here are the 3 most recent queries over this corpus (oldest first):" in text
    assert "  1. question 2\n  2. question 3\n  3. question 4" in text and "question 1" not in text


def test_enrich_prompt_without_collections_or_queries(client):
    system = _StubSystem(max_previous_queries=20)
    asyncio.run(system._enrich(_ctx(), _resources(client)))
    text = system.agent.inputs[0]
    assert "There are no agent-created collections yet." in text
    assert "No queries have been asked over this corpus yet." in text


def test_history_is_bounded_and_ordered():
    system = _StubSystem(max_previous_queries=2)
    for i in range(4):
        system._question_history.append(Question(qid=str(i), text=f"q{i}", gold="").text)
    assert list(system._question_history) == ["q2", "q3"]
