import base64
import hashlib
import json
import os
import pickle
import re

import requests
import untruncate_json

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
OPENROUTER_VISION_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"
LLM_CACHE_PATH = os.environ.get(
    "SKUNK_LLM_CACHE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache", "llm_call_cache.pckl")),
)
OPENROUTER_TEXT_SYSTEM_PROMPT = "You extract compact metadata for a semantic document index. Return only the requested text."
OPENROUTER_VISION_SYSTEM_PROMPT = "You match rendered PDF pages to spans of OCR text. Return only valid JSON."


def load_llm_cache() -> dict:
    if not os.path.exists(LLM_CACHE_PATH):
        return {}

    with open(LLM_CACHE_PATH, "rb") as cache_file:
        return pickle.load(cache_file)


def save_llm_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(LLM_CACHE_PATH), exist_ok=True)
    tmp_cache_path = f"{LLM_CACHE_PATH}.tmp"
    with open(tmp_cache_path, "wb") as cache_file:
        pickle.dump(cache, cache_file)
    os.replace(tmp_cache_path, LLM_CACHE_PATH)


def llm_cache_key(request_inputs: dict) -> str:
    request_json = json.dumps(request_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(request_json.encode("utf-8")).hexdigest()


def extract_openrouter_text(response: requests.Response) -> str:
    if not response.ok:
        raise requests.HTTPError(
            f"{response.status_code} {response.reason} from OpenRouter: {response.text}",
            response=response,
        )

    response_json = response.json()
    choice = response_json["choices"][0]
    message = choice.get("message", {})
    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        text = "".join(parts).strip()
        if text:
            return text

    raise ValueError(f"No text content in OpenRouter response choice: {choice}")


def parse_json_response(text: str) -> dict:
    cleaned_text = text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        return json.loads(untruncate_json.complete(cleaned_text))


def call_openrouter(prompt: str, max_tokens: int = 2048, model: str = OPENROUTER_MODEL) -> str:
    request_inputs = {
        "call_type": "openrouter_text",
        "model": model,
        "system_prompt": OPENROUTER_TEXT_SYSTEM_PROMPT,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    cache_key = llm_cache_key(request_inputs)
    cache = load_llm_cache()
    if cache_key in cache:
        return cache[cache_key]

    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": OPENROUTER_TEXT_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response_text = extract_openrouter_text(response)
    cache[cache_key] = response_text
    save_llm_cache(cache)
    return response_text


def call_openrouter_vision(
    prompt: str,
    image_bytes: bytes,
    max_tokens: int = 4096,
    model: str = OPENROUTER_VISION_MODEL,
) -> str:
    request_inputs = {
        "call_type": "openrouter_vision",
        "model": model,
        "system_prompt": OPENROUTER_VISION_SYSTEM_PROMPT,
        "prompt": prompt,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    cache_key = llm_cache_key(request_inputs)
    cache = load_llm_cache()
    if cache_key in cache:
        return cache[cache_key]

    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": OPENROUTER_VISION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    response_text = extract_openrouter_text(response)
    cache[cache_key] = response_text
    save_llm_cache(cache)
    return response_text


def embed_texts(texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    cleaned_texts = [text.strip() or "empty semantic description" for text in texts]
    cache = load_llm_cache()
    embeddings: list[list[float] | None] = [None] * len(cleaned_texts)
    uncached_texts = []
    uncached_positions = []

    for text_idx, text in enumerate(cleaned_texts):
        request_inputs = {
            "call_type": "embedding",
            "model": model,
            "text": text,
        }
        cache_key = llm_cache_key(request_inputs)
        if cache_key in cache:
            embeddings[text_idx] = cache[cache_key]
        else:
            uncached_texts.append(text)
            uncached_positions.append((text_idx, cache_key))

    if uncached_texts:
        import litellm

        response = litellm.embedding(model=model, input=uncached_texts)
        for response_idx, item in enumerate(response.data):
            embedding = item["embedding"] if isinstance(item, dict) else item.embedding
            text_idx, cache_key = uncached_positions[response_idx]
            cache[cache_key] = embedding
            embeddings[text_idx] = embedding
        save_llm_cache(cache)

    return [embedding for embedding in embeddings if embedding is not None]
