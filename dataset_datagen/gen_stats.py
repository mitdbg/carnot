"""Per-generation usage + latency accounting for the datagen scripts.

`skunk`'s `UsageTracker` (one per `LLMClient`) buckets every generation and embedding
call by a caller-supplied `usage_key`, and `MultiTurnAgent` passes its own `agent_id`
as that key for both its LLM steps and its tools' embeddings. So the finest granularity
skunk offers is **one bucket per agent instance** — as long as every agent is given a
distinct `agent_id`, a single shared `LLMClient` still yields exact per-agent token
counts and cost. That is what `usage_snapshot` reads out.

Latency is NOT tracked by the usage tracker. Per-call latency only shows up as
`data["latency_s"]` on the `kind="call"` events the `Tracer` streams to its per-question
`.jsonl` file, and those events carry no usage key. Since the agents within one question
run sequentially, `trace_latency` recovers per-agent LLM latency by reading the slice of
that file the agent appended (byte offset before vs. after the call); wall-clock is
measured directly by the caller.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skunk.llm_client import LLMClient


@dataclass
class GenStats:
    """Cost + latency for one agent run (one line of inquiry, or one follow-up question)."""

    usage_key: str              # == the agent's agent_id; the UsageTracker bucket
    kind: str                   # "line_of_inquiry" | "follow_up_question"
    qid: str
    idx: int | None = None      # follow-up index within the question; None for the line of inquiry
    attempt: int | None = None  # used to track retries

    # wall-clock around `agent.call()` — includes rate-limit waits, retries and tool time
    wall_latency_s: float = 0.0
    # summed `latency_s` of this agent's LLM + embedding calls (excludes queueing/tool time)
    llm_latency_s: float = 0.0

    cost_usd: float = 0.0        # generation + embedding, from the price table
    embed_cost_usd: float = 0.0  # embedding share of `cost_usd`
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thinking_tokens: int = 0
    embed_tokens: int = 0
    n_llm_calls: int = 0
    n_embed_calls: int = 0
    n_steps: int = 0

    model_to_input_tokens: dict[str, int] = field(default_factory=dict)
    model_to_output_tokens: dict[str, int] = field(default_factory=dict)
    model_to_cached_tokens: dict[str, int] = field(default_factory=dict)
    model_to_think_tokens: dict[str, int] = field(default_factory=dict)
    model_to_embed_tokens: dict[str, int] = field(default_factory=dict)

    # agent outcome: "finished" | "out_of_steps" | "over_cost_budget" | ... | "error"
    terminate_state: str = "finished"
    error: str | None = None

    # used to indicate whether a generated inquiry or qa pair was deemed solvable / unique or not
    solvable: bool | None = None
    unique: bool | None = None

    def to_json(self) -> dict:
        return asdict(self)


def usage_snapshot(
    llm_client: LLMClient,
    usage_key: str,
    *,
    kind: str,
    qid: str,
    idx: int | None = None,
    attempt: int | None = None,
    wall_latency_s: float = 0.0,
    llm_latency_s: float = 0.0,
    n_steps: int = 0,
) -> GenStats:
    """Read one agent's bucket out of the shared `UsageTracker`.

    Safe to read as a plain total (rather than a before/after delta) because every agent
    is constructed with a unique `agent_id`, so no other caller ever writes this bucket.
    """
    tracker = llm_client.usage
    usage = tracker.key_to_usage.get(usage_key)
    embed = tracker.key_to_embed_usage.get(usage_key)
    stats = GenStats(
        usage_key=usage_key,
        kind=kind,
        qid=qid,
        idx=idx,
        attempt=attempt,
        n_steps=n_steps,
        wall_latency_s=round(wall_latency_s, 3),
        llm_latency_s=round(llm_latency_s, 3),
        cost_usd=tracker.cost(key=usage_key),
        embed_cost_usd=tracker.embed_cost(key=usage_key),
    )
    if usage is not None:
        stats.input_tokens = usage.input_tokens
        stats.output_tokens = usage.output_tokens
        stats.cached_tokens = usage.cached_tokens
        stats.thinking_tokens = sum(usage.model_to_think_tokens.values())
        stats.n_llm_calls = usage.n_calls
        stats.model_to_input_tokens = dict(usage.model_to_input_tokens)
        stats.model_to_output_tokens = dict(usage.model_to_output_tokens)
        stats.model_to_cached_tokens = dict(usage.model_to_cached_tokens)
        stats.model_to_think_tokens = dict(usage.model_to_think_tokens)
    if embed is not None:
        stats.embed_tokens = embed.embed_tokens
        stats.n_embed_calls = embed.n_embed_calls
        stats.model_to_embed_tokens = dict(embed.model_to_embed_tokens)
    return stats


def trace_latency(log_path: Path | str, start_offset: int) -> float:
    """Summed `data["latency_s"]` of the `kind="call"` events written to the tracer's
    `.jsonl` file from byte `start_offset` onward.

    `start_offset` is the file size captured immediately before the agent ran (the
    `Tracer` flushes after every event, so the size is exact). Only valid when a single
    agent was running over that slice — true here because the agents within one question
    run one after another on one `ExecutionContext` / `Tracer`. The uniform `call`
    envelope is emitted by `LLMClient._build_response` (generation) and
    `LLMClient.embed_query` (embeddings); retry warnings share the `call` kind but carry
    no `latency_s`, so they are skipped.
    """
    total = 0.0
    with open(log_path, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:  # tolerate a partial trailing line
                continue
            if evt.get("kind") != "call":
                continue
            latency = (evt.get("data") or {}).get("latency_s")
            if latency is not None:
                total += float(latency)
    return total


def aggregate(all_stats: list[GenStats]) -> dict:
    """Roll per-generation stats up into a run-level summary (totals + per-kind splits)."""

    def _roll(rows: list[GenStats]) -> dict:
        return {
            "n_generations": len(rows),
            "cost_usd": sum(s.cost_usd for s in rows),
            "embed_cost_usd": sum(s.embed_cost_usd for s in rows),
            "input_tokens": sum(s.input_tokens for s in rows),
            "output_tokens": sum(s.output_tokens for s in rows),
            "cached_tokens": sum(s.cached_tokens for s in rows),
            "thinking_tokens": sum(s.thinking_tokens for s in rows),
            "embed_tokens": sum(s.embed_tokens for s in rows),
            "n_llm_calls": sum(s.n_llm_calls for s in rows),
            "n_embed_calls": sum(s.n_embed_calls for s in rows),
            "llm_latency_s": sum(s.llm_latency_s for s in rows),
            "wall_latency_s": sum(s.wall_latency_s for s in rows),
            "mean_cost_usd": (sum(s.cost_usd for s in rows) / len(rows)) if rows else 0.0,
            "mean_wall_latency_s": (sum(s.wall_latency_s for s in rows) / len(rows)) if rows else 0.0,
            "n_errors": sum(1 for s in rows if s.error is not None),
        }

    kinds = sorted({s.kind for s in all_stats})
    return {
        "total": _roll(all_stats),
        "by_kind": {k: _roll([s for s in all_stats if s.kind == k]) for k in kinds},
    }
