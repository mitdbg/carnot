import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence


def chat_json(
    messages: Sequence[Dict[str, str]],
    *,
    provider: str,
    model: str,
    timeout: float = 25.0,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    content = _chat_text(
        messages,
        provider=provider,
        model=model,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_json_object(content)


def parallel_json(
    tasks: Sequence[Dict[str, Any]],
    *,
    provider: str,
    model: str,
    timeout: float = 25.0,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    if not tasks:
        return []
    workers = max(1, min(max_workers, len(tasks)))

    def _run(task: Dict[str, Any]) -> Dict[str, Any]:
        return chat_json(
            task["messages"],
            provider=provider,
            model=model,
            timeout=timeout,
            temperature=task.get("temperature", 0.0),
            max_tokens=task.get("max_tokens", 1200),
        )

    if workers == 1:
        return [_run(task) for task in tasks]

    out: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run, task): index for index, task in enumerate(tasks)}
        for future in as_completed(futures):
            out[futures[future]] = future.result()
    return [item or {} for item in out]


def _chat_text(
    messages: Sequence[Dict[str, str]],
    *,
    provider: str,
    model: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
) -> str:
    provider = (provider or "openrouter").lower()
    if provider == "gemini":
        return _gemini_chat(messages, model=model, timeout=timeout, temperature=temperature, max_tokens=max_tokens)
    return _openrouter_chat(messages, model=model, timeout=timeout, temperature=temperature, max_tokens=max_tokens)


def _openrouter_chat(messages, *, model, timeout, temperature, max_tokens) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required when llm_provider=openrouter.")
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer {}".format(api_key),
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError("OpenRouter request failed: {}".format(exc)) from exc
    return raw["choices"][0]["message"]["content"]


def _gemini_chat(messages, *, model, timeout, temperature, max_tokens) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required when llm_provider=gemini.")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Install google-genai: pip install google-genai") from exc

    client = genai.Client(api_key=api_key)
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    user_parts = [m["content"] for m in messages if m.get("role") != "system"]
    prompt = ""
    if system_parts:
        prompt += "\n\n".join(system_parts) + "\n\n"
    prompt += "\n\n".join(user_parts)

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )
    text = getattr(response, "text", None)
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = candidates[0].content
        parts = getattr(content, "parts", None) or []
        if parts:
            return "".join(getattr(part, "text", "") or "" for part in parts)
    return ""


def _extract_json_object(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}
