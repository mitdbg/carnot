"""Build a QDT (anchor + probes) from a user query via LLM. Used by eval_indices summary-QDT pipeline."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from carnot.agents.models import ChatMessage, LiteLLMModel, MessageRole

_DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent / "qdt_probe_prompt.yaml"


def load_qdt_probe_system_prompt(path: Path | None = None) -> str:
    p = path or _DEFAULT_PROMPT_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return str(data["qdt_probe_system"]).strip()


def _extract_json_object(text: str) -> dict:
    raw = text.strip()
    fence = re.match(r"^```(?:json)?\s*([\s\S]*?)```\s*$", raw)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


def generate_qdt_from_query(
    query: str,
    api_key: str,
    model_id: str,
    system_prompt_path: Path | None = None,
) -> tuple[dict, object | None]:
    """Return ``{"anchor": str, "probes": [str, ...]}`` and the LLM call stats."""
    system = load_qdt_probe_system_prompt(system_prompt_path)
    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    user_msg = ChatMessage(
        role=MessageRole.USER,
        content=f"User query:\n\n{query}",
    )
    sys_msg = ChatMessage(role=MessageRole.SYSTEM, content=system)
    try:
        response = model.generate(
            messages=[sys_msg, user_msg],
            response_format={"type": "json_object"},
        )
    except Exception:
        response = model.generate(messages=[sys_msg, user_msg])
    if not response.content:
        raise ValueError("QDT model returned empty content")
    parsed = _extract_json_object(response.content)
    print("QDT IS")
    print(parsed)
    anchor = parsed.get("anchor")
    probes = parsed.get("probes")
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError('QDT JSON must contain non-empty string "anchor"')
    if not isinstance(probes, list):
        raise ValueError('QDT JSON must contain array "probes"')
    probes_clean: list[str] = []
    for p in probes:
        if isinstance(p, str) and p.strip():
            probes_clean.append(p.strip())
    out = {"anchor": anchor.strip(), "probes": probes_clean}
    return out, response.llm_call_stats


def dedupe_qdt_tasks(anchor: str, probes: list[str]) -> list[tuple[str, str]]:
    """Return (label, task) pairs; skip duplicate task strings (anchor first)."""
    seen: set[str] = set()
    tasks: list[tuple[str, str]] = []
    a = anchor.strip()
    if a not in seen:
        seen.add(a)
        tasks.append(("anchor", a))
    pi = 0
    for p in probes:
        t = p.strip()
        if t in seen:
            continue
        seen.add(t)
        tasks.append((f"probe_{pi}", t))
        pi += 1
    return tasks
