"""Unit tests for the vLLM backend and the per-model backend routing in `LLMClient`.

No GPU and no network: the openai-SDK request path is exercised end-to-end against a
local in-process HTTP stub speaking the OpenAI chat/embeddings API (including SSE
streaming). Runs under pytest."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from skunk.config import SearchAgentConfig
from skunk.llm_client import (
    EmptyCompletionError,
    LLMClient,
    _is_retryable,
    _OpenRouterBackend,
    _VLLMBackend,
)


def _config(**overrides) -> SearchAgentConfig:
    base = dict(
        name="t", emb_provider="openrouter", emb_model_id="emb", llm_provider="openrouter",
        llm_model="loc/m", llm_max_retries=0, llm_retry_initial_delay_s=0.0,
        llm_model_rpm={}, llm_default_rpm=1e9, llm_model_tpm={}, llm_default_tpm=None, llm_prices={},
        llm_context_limits={},
    )
    base.update(overrides)
    return SearchAgentConfig(**base)


# ---- routing ----------------------------------------------------------------------


def test_routing_vllm_map_wins_over_default_provider():
    client = LLMClient(_config(vllm_base_urls={"loc/m": "http://127.0.0.1:1/v1"}))
    assert isinstance(client._backend_for_model("loc/m"), _VLLMBackend)
    assert isinstance(client._backend_for_model("google/gemini-3.5-flash"), _OpenRouterBackend)


def test_routing_vllm_default_requires_mapping():
    client = LLMClient(_config(llm_provider="vllm", vllm_base_urls={"loc/m": "http://127.0.0.1:1/v1"}))
    assert isinstance(client._backend_for_model("loc/m"), _VLLMBackend)
    with pytest.raises(RuntimeError, match="no vllm_base_urls entry"):
        client._backend_for_model("unmapped/m")


def test_backends_share_one_usage_tracker(monkeypatch):
    client = LLMClient(_config(vllm_base_urls={"loc/m": "http://127.0.0.1:1/v1"}))

    def fake_gen(self, spec):
        toks = {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}
        return self._finish(spec, "ok", toks, 0.0, spec.model)

    monkeypatch.setattr(_VLLMBackend, "_gen_call", fake_gen)
    monkeypatch.setattr(_OpenRouterBackend, "_gen_call", fake_gen)
    client.call("sys", "user", model="loc/m")
    client.call("sys", "user", model="or/m")
    assert client.usage.n_calls == 2
    assert set(client.usage.by_model_in) == {"loc/m", "or/m"}
    assert client._backend("vllm").usage is client.usage is client._backend("openrouter").usage


def test_vllm_models_cost_zero_even_when_priced(monkeypatch):
    """A model can be priced for OpenRouter runs AND served by vLLM in this run: the
    vllm_base_urls keys are registered as UsageTracker free models, so their calls cost
    $0 while the same table still prices OpenRouter-routed models."""
    prices = {"loc/m": {"in": 100.0, "out": 100.0}, "or/m": {"in": 10.0, "out": 10.0}}
    client = LLMClient(_config(vllm_base_urls={"loc/m": "http://127.0.0.1:1/v1"}, llm_prices=prices))

    def fake_gen(self, spec):
        toks = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "total_tokens": 2_000_000}
        return self._finish(spec, "ok", toks, 0.0, spec.model)

    monkeypatch.setattr(_VLLMBackend, "_gen_call", fake_gen)
    monkeypatch.setattr(_OpenRouterBackend, "_gen_call", fake_gen)
    client.call("s", "u", model="loc/m")
    assert client.usage.price_call("loc/m", 1_000_000, 0, 1_000_000) is None  # free, not $200
    assert client.usage.cost() == 0.0
    client.call("s", "u", model="or/m")
    assert client.usage.price_call("or/m", 1_000_000, 0, 1_000_000) == 20.0  # still priced
    assert client.usage.cost() == 20.0


# ---- stub OpenAI-compatible server ---------------------------------------------------


def _chat_payload(text: str | None) -> dict:
    return {
        "id": "c1", "object": "chat.completion", "created": 1, "model": "loc/m",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    }


_SSE_CHUNKS = [
    {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "loc/m",
        "choices": [{"index": 0, "delta": {"content": "hel"}, "finish_reason": None}],
    },
    {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "loc/m",
        "choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}],
    },
    # Final usage-only chunk, sent because the request asked include_usage.
    {
        "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "loc/m",
        "choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    },
]


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ARG002 — silence per-request stderr noise
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append((self.path, body))  # type: ignore[attr-defined]
        if self.path.endswith("/chat/completions") and body.get("stream"):
            self._respond_sse()
        elif self.path.endswith("/chat/completions"):
            queued = self.server.chat_responses  # type: ignore[attr-defined]
            self._respond_json(queued.pop(0) if queued else _chat_payload("hello"))
        elif self.path.endswith("/embeddings"):
            self._respond_json({
                "object": "list", "model": body.get("model"),
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            })
        else:
            self.send_error(404)

    def _respond_json(self, payload: dict):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _respond_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in _SSE_CHUNKS:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture()
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    server.requests = []  # type: ignore[attr-defined]
    server.chat_responses = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def _stub_client(server, **overrides) -> LLMClient:
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return LLMClient(_config(vllm_base_urls={"loc/m": base_url, "emb": base_url}, **overrides))


# ---- vLLM bodies against the stub ---------------------------------------------------


def test_vllm_call_parses_text_and_usage(stub_server):
    resp = _stub_client(stub_server).call("be brief", "hi", model="loc/m")
    assert resp.text == "hello"
    assert (resp.input_tokens, resp.output_tokens) == (7, 2)
    path, body = stub_server.requests[0]
    assert path.endswith("/chat/completions")
    assert body["model"] == "loc/m"
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert "extra_body" not in body  # not set -> nothing merged


def test_vllm_extra_body_lands_in_request(stub_server):
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    client = _stub_client(stub_server, vllm_extra_body=extra)
    client.call("s", "u", model="loc/m")
    _, body = stub_server.requests[0]
    # The openai SDK merges extra_body into the top-level request JSON.
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_vllm_call_passes_max_output_tokens(stub_server):
    client = _stub_client(stub_server)
    client.call("s", "u", model="loc/m", max_output_tokens=256)
    _, body = stub_server.requests[0]
    assert body["max_tokens"] == 256
    # Default (no cap) sends no max_tokens, preserving prior behavior.
    client.call("s", "u", model="loc/m")
    _, body2 = stub_server.requests[1]
    assert "max_tokens" not in body2


def test_vllm_empty_completion_retries_then_succeeds(stub_server):
    stub_server.chat_responses.extend([_chat_payload(""), _chat_payload("second try")])
    client = _stub_client(stub_server, llm_max_retries=1)
    resp = client.call("s", "u", model="loc/m")
    assert resp.text == "second try"
    assert len(stub_server.requests) == 2  # empty 200 -> EmptyCompletionError -> one retry


def test_vllm_empty_completion_exhausts_retries(stub_server):
    stub_server.chat_responses.extend([_chat_payload(None)])
    client = _stub_client(stub_server)  # llm_max_retries=0
    with pytest.raises(EmptyCompletionError, match="vllm empty content"):
        client.call("s", "u", model="loc/m")


def test_vllm_acall(stub_server):
    resp = asyncio.run(_stub_client(stub_server).acall("s", "u", model="loc/m", max_output_tokens=64))
    assert resp.text == "hello"
    _, body = stub_server.requests[0]
    assert body["max_tokens"] == 64


def test_vllm_astream_drains_usage(stub_server):
    client = _stub_client(stub_server)
    resp = asyncio.run(
        client.astream(system="s", messages=[{"role": "user", "content": "hi"}], model="loc/m")
    )
    assert resp.text == "hello"
    assert (resp.input_tokens, resp.output_tokens) == (5, 3)
    _, body = stub_server.requests[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_vllm_embed_query(stub_server):
    client = _stub_client(stub_server, emb_provider="vllm")
    vec = client.embed_query("apples")
    assert vec == [0.1, 0.2, 0.3]
    assert client.usage.embed_tokens == 4
    path, body = stub_server.requests[0]
    assert path.endswith("/embeddings")
    assert body["model"] == "emb"


# ---- openai error taxonomy in _is_retryable ------------------------------------------


def _status_error(status: int):
    from openai import APIStatusError

    req = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
    resp = httpx.Response(status, request=req, text="err")
    return APIStatusError("boom", response=resp, body=None)


def test_is_retryable_openai_errors():
    from openai import APIConnectionError

    assert _is_retryable(_status_error(429))
    assert _is_retryable(_status_error(503))
    assert not _is_retryable(_status_error(400))
    req = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
    assert _is_retryable(APIConnectionError(request=req))
