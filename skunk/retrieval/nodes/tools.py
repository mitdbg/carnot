import base64
import os

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "google/gemini-3.1-pro-preview"
OPENROUTER_VISION_MODEL = "google/gemini-3.1-pro-preview"


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


def call_openrouter(prompt: str, max_tokens: int = 2048, model: str = OPENROUTER_MODEL) -> str:
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
                    "content": "You extract compact metadata for a semantic document index. Return only the requested text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    return extract_openrouter_text(response)


def call_openrouter_vision(
    prompt: str,
    image_bytes: bytes,
    max_tokens: int = 4096,
    model: str = OPENROUTER_VISION_MODEL,
) -> str:
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
                    "content": "You match rendered PDF pages to spans of OCR text. Return only valid JSON.",
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
    return extract_openrouter_text(response)
