import base64
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import pickle
import random
import re
import threading
import time

import json5
import litellm
import numpy as np
import untruncate_json
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tqdm import tqdm

litellm.suppress_debug_info = True

# DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
# DEFAULT_LLM_VISION_MODEL = "openai/gpt-4o-mini"
# DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# DEFAULT_LLM_MODEL = "openrouter/google/gemini-3.1-pro-preview"
# DEFAULT_LLM_VISION_MODEL = "openrouter/google/gemini-3.1-pro-preview"
# DEFAULT_EMBEDDING_MODEL = "openrouter/google/gemini-embedding-001"

DEFAULT_LLM_MODEL = "gemini/gemini-3.5-flash"
DEFAULT_LLM_VISION_MODEL = "gemini/gemini-3.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini/gemini-embedding-001"

DEFAULT_CACHE_DIR = os.path.expanduser("~/orcd/scratch/skunk_cache/")
LLM_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "llm_wrapper_cache.pckl")
LLM_RATE_LIMIT_STATE_PATH = f"{LLM_CACHE_PATH}.rate_limit.json"
LLM_RATE_LIMIT_LOCK_PATH = f"{LLM_RATE_LIMIT_STATE_PATH}.lock"
LLM_TEXT_SYSTEM_PROMPT = "You extract compact metadata for a semantic document index. Return only the requested text."
LLM_VISION_SYSTEM_PROMPT = (
    "You match rendered PDF pages to spans of OCR text. Return only valid JSON."
)

LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", 32))
LLM_MAX_REQUESTS_PER_MINUTE = int(os.environ.get("LLM_MAX_REQUESTS_PER_MINUTE", 15000))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 15))
LLM_RETRY_BACKOFF_SECONDS = int(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", 30))
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", 90))

def remove_provider(model_name):
    return model_name.split("/")[-1]

def llm_cache_key(request_inputs: dict) -> str:
    provider_agnostic = request_inputs.copy()
    provider_agnostic["model"] = remove_provider(provider_agnostic["model"])
    request_json = json.dumps(provider_agnostic, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(request_json.encode("utf-8")).hexdigest()


def extract_litellm_text(response) -> str:
    choice = (
        response.choices[0] if hasattr(response, "choices") else response["choices"][0]
    )
    message = (
        choice.message if hasattr(choice, "message") else choice.get("message", {})
    )
    content = message.content if hasattr(message, "content") else message.get("content")

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

    print(
        f"Warning: LLM response content is an empty string. Returning empty string.\nResponse: {response}"
    )
    return ""


def extract_google_genai_text(response) -> str:
    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            text = getattr(part, "text", None)
            if text:
                parts.append(text)
    text = "".join(parts).strip()
    if text:
        return text

    # print(f"Warning: Google GenAI response content is an empty string. Returning empty string.\nResponse: {response}")
    return ""


def parse_json_response(text: str) -> dict:
    cleaned_text = text.strip()
    if cleaned_text == '':
        return {}
    elif "```" in cleaned_text:
        matches = re.findall(
            r"```(?:json)?\s*(.*?)\s*```", cleaned_text, flags=re.DOTALL
        )
        if matches:
            cleaned_text = matches[-1].strip()

    try:
        return json.loads(untruncate_json.complete(cleaned_text))
    except json.JSONDecodeError:
        try:
            return json5.loads(cleaned_text)
        except Exception as e:
            # print(f"Failed to parse JSON response: {e}\nOriginal text: {text}")
            # with open("json_parse_error.txt", "w") as f:
            # f.write(text)
            return {}


def extract_google_usage_metadata(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
        "thinking_tokens": getattr(usage, "thoughts_token_count", None),
        "cache_input_tokens": getattr(usage, "cached_content_token_count", None),
    }


def extract_litellm_usage_metadata(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "thinking_tokens": None,
            "cache_input_tokens": None,
        }
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = details.get("cached_tokens") if isinstance(details, dict) else None
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(details, "cached_tokens", None) if details else None
    return {
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "thinking_tokens": None,
        "cache_input_tokens": cached_tokens,
    }


class LLMWrapper:
    def __init__(
        self,
        max_workers: int = LLM_MAX_WORKERS,
        max_requests_per_minute: int = LLM_MAX_REQUESTS_PER_MINUTE,
        max_retries: int = LLM_MAX_RETRIES,
        retry_backoff_seconds: int = LLM_RETRY_BACKOFF_SECONDS,
        text_timeout: int = 60,
        vision_timeout: int = 120,
        cache_path: str = LLM_CACHE_PATH,
        cache_enabled: bool = True,
    ):
        self.max_workers = max(1, max_workers)
        self.max_requests_per_minute = max(1, max_requests_per_minute)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(1, retry_backoff_seconds)
        self.text_timeout = text_timeout
        self.vision_timeout = vision_timeout
        self.cache_path = cache_path
        self.cache_lock_path = f"{self.cache_path}.lock"
        self.cache_enabled = cache_enabled
        self.cache_lock = threading.RLock()
        self.flush_lock = threading.Lock()
        self.dirty_cache_entries = {}
        self.generated_embedding_cache_keys = set()
        if self.cache_enabled:
            with self.locked_cache_file():
                self.cache = self.load_cache_from_disk()
        else:
            self.cache = {}

    @contextlib.contextmanager
    def locked_cache_file(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def load_cache_from_disk(self) -> dict:
        if not os.path.exists(self.cache_path):
            return {}

        with open(self.cache_path, "rb") as cache_file:
            return pickle.load(cache_file)

    def save_cache_to_disk(self, cache: dict) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_cache_path = f"{self.cache_path}.tmp"
        with open(tmp_cache_path, "wb") as cache_file:
            pickle.dump(cache, cache_file)
        os.replace(tmp_cache_path, self.cache_path)

    def get_cached_value(self, cache_key: str):
        if not self.cache_enabled:
            return None

        with self.cache_lock:
            return self.cache.get(cache_key)

    def get_cached_values(self, cache_keys: list[str | None]) -> list:
        if not self.cache_enabled:
            return [None] * len(cache_keys)
        with self.cache_lock:
            return [self.cache.get(key, None) for key in cache_keys]

    def store_cached_value(self, cache_key: str, value) -> None:
        if not self.cache_enabled:
            return

        with self.cache_lock:
            self.cache[cache_key] = value
            self.dirty_cache_entries[cache_key] = value

    def flush_cache(
        self, evict_generated_embeddings: bool = False, wait: bool = True
    ) -> bool:
        """
        Flush dirty in-memory cache entries to disk.

        A process-local flush lock prevents multiple expensive pickle writes from
        running at the same time. When wait is False, this method is non-blocking:
        if another thread is already flushing, it returns False and leaves dirty
        entries in memory for a later flush. When wait is True, it blocks until
        the flush lock is acquired, which is appropriate for final or durability
        flushes.

        If evict_generated_embeddings is True, generated embedding arrays are
        removed from the in-memory cache after they have been persisted.
        """
        if not self.cache_enabled:
            return False
        acquired = self.flush_lock.acquire(blocking=wait)
        if not acquired:
            return False
        try:
            with self.cache_lock:
                if not self.dirty_cache_entries:
                    return False
                dirty_cache_entries = dict(self.dirty_cache_entries)

            with self.locked_cache_file():
                disk_cache = self.load_cache_from_disk()
                disk_cache.update(dirty_cache_entries)
                self.save_cache_to_disk(disk_cache)

            with self.cache_lock:
                for cache_key in dirty_cache_entries:
                    current_dirty_value = self.dirty_cache_entries.get(cache_key)
                    flushed_value = dirty_cache_entries[cache_key]
                    values_match = current_dirty_value is flushed_value
                    if not values_match:
                        if isinstance(current_dirty_value, np.ndarray) or isinstance(
                            flushed_value, np.ndarray
                        ):
                            values_match = np.array_equal(
                                current_dirty_value, flushed_value
                            )
                        else:
                            values_match = current_dirty_value == flushed_value
                    if values_match:
                        self.dirty_cache_entries.pop(cache_key, None)
                        if (
                            evict_generated_embeddings
                            and cache_key in self.generated_embedding_cache_keys
                        ):
                            self.cache.pop(cache_key, None)
                            self.generated_embedding_cache_keys.discard(cache_key)
                        else:
                            self.cache[cache_key] = flushed_value

                if not evict_generated_embeddings:
                    for cache_key, value in disk_cache.items():
                        if cache_key not in self.dirty_cache_entries:
                            self.cache[cache_key] = value
            return True
        finally:
            self.flush_lock.release()

    def is_rate_limit_error(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None) or getattr(
            error, "code", None
        )
        error_text = str(error).lower()
        if status_code == 429:
            return True

        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True

        if isinstance(error, genai_errors.APIError) and error.code == 429:
            return True

        return (
            "429" in error_text
            or "rate limit" in error_text
            or "too many requests" in error_text
            or "resource exhausted" in error_text
            or "quota exceeded" in error_text
        )

    def is_transient_llm_error(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None) or getattr(
            error, "code", None
        )
        if status_code in {408, 500, 502, 503, 504}:
            return True

        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) in {408, 500, 502, 503, 504}:
            return True

        if isinstance(error, genai_errors.APIError) and error.code in {
            408,
            500,
            502,
            503,
            504,
        }:
            return True

        if isinstance(error, litellm.exceptions.ServiceUnavailableError):
            return True

        error_text = str(error).lower()
        return any(
            transient_text in error_text
            for transient_text in [
                "openrouterexception",
                "unexpected_eof_while_reading",
                "unexpected eof while reading",
                "server disconnected",
                "connection reset",
                "connection error",
                "remote protocol error",
                "read timeout",
                "write timeout",
                "temporarily unavailable",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
                "deadline exceeded",
                "internal server error",
                "server error",
            ]
        )

    def run_with_rate_limit_retries(self, request_fn):
        attempt_idx = 0
        while True:
            try:
                return request_fn()
            except Exception as e:
                is_rate_limited = self.is_rate_limit_error(e)
                is_transient = self.is_transient_llm_error(e)
                if (
                    not is_rate_limited and not is_transient
                ) or attempt_idx >= self.max_retries:
                    raise
                sleep_seconds = self.retry_backoff_seconds * (
                    attempt_idx + 1
                ) + random.uniform(0, self.retry_backoff_seconds)
                if is_rate_limited:
                    # print(
                    #     f"LLM error hit limit hit ({type(e).__name__}: {e}); ",
                    #     f"{e.with_traceback(None)}",
                    #     f"retrying in {sleep_seconds:.1f} seconds.",
                    # )
                    self.wait_for_rate_limit_slot()
                # else:
                # print(
                #     f"Transient LLM error ({type(e).__name__}: {e}); "
                #     f"retrying in {sleep_seconds:.1f} seconds."
                # )
                # self.flush_cache(evict_generated_embeddings=True)
                time.sleep(sleep_seconds)
                attempt_idx += 1

    def wait_for_rate_limit_slot(self) -> None:
        os.makedirs(os.path.dirname(LLM_RATE_LIMIT_STATE_PATH), exist_ok=True)

        while True:
            with open(LLM_RATE_LIMIT_LOCK_PATH, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                now = time.time()
                request_times = []
                if os.path.exists(LLM_RATE_LIMIT_STATE_PATH):
                    try:
                        with open(LLM_RATE_LIMIT_STATE_PATH) as state_file:
                            request_times = json.load(state_file).get(
                                "request_times", []
                            )
                    except (json.JSONDecodeError, OSError):
                        request_times = []

                request_times = [
                    request_time
                    for request_time in request_times
                    if now - request_time < 60
                ]
                if len(request_times) < self.max_requests_per_minute:
                    request_times.append(now)
                    tmp_state_path = f"{LLM_RATE_LIMIT_STATE_PATH}.tmp"
                    with open(tmp_state_path, "w") as state_file:
                        json.dump({"request_times": request_times}, state_file)
                    os.replace(tmp_state_path, LLM_RATE_LIMIT_STATE_PATH)
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    return

                sleep_seconds = max(0.05, 60 - (now - min(request_times)))
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            time.sleep(sleep_seconds)

    def generate_google(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        image_bytes: bytes = None,
        max_tokens: int = 32000,
    ) -> str:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set"
            )
        client = genai.Client(api_key=api_key)
        if image_bytes is not None:
            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt,
            ]
        else:
            contents = prompt
        response = client.models.generate_content(
            model=remove_provider(model),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=max_tokens,
            ),
        )
        return extract_google_genai_text(response)

    def generate_google_with_usage(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        image_bytes: bytes = None,
        max_tokens: int = 32000,
    ) -> dict:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set"
            )
        client = genai.Client(api_key=api_key)
        if image_bytes is not None:
            contents = [
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                prompt,
            ]
        else:
            contents = prompt
        response = client.models.generate_content(
            model=remove_provider(model),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=max_tokens,
            ),
        )
        return {
            "text": extract_google_genai_text(response),
            "usage": extract_google_usage_metadata(response),
            "cache_hit": False,
        }

    def generate_litellm(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        image_bytes: bytes = None,
        max_tokens: int = 32000,
    ) -> str:

        if image_bytes is not None:
            image_base64 = base64.b64encode(image_bytes).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                },
            ]
        else:
            content = prompt
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=self.text_timeout,
        )

        return extract_litellm_text(response)

    def generate_litellm_with_usage(
        self,
        prompt: str,
        model: str,
        system_prompt: str,
        image_bytes: bytes = None,
        max_tokens: int = 32000,
    ) -> dict:

        if image_bytes is not None:
            image_base64 = base64.b64encode(image_bytes).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                },
            ]
        else:
            content = prompt
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": content},
            ],
            temperature=0,
            max_tokens=max_tokens,
            timeout=self.text_timeout,
        )

        return {
            "text": extract_litellm_text(response),
            "usage": extract_litellm_usage_metadata(response),
            "cache_hit": False,
        }

    def embed_google(self, documents: list[str], model: str):
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable is not set"
            )
        client = genai.Client(api_key=api_key)
        google_model = remove_provider(model)
        embeddings = []
        if "embedding-001" in google_model:
            result = client.models.embed_content(
                model=google_model,
                contents=documents,  # type: ignore
                config=types.EmbedContentConfig(output_dimensionality=768),
            )

            for e in result.embeddings:
                np_emb = np.array(e.values, dtype=np.float32)
                normed_embedding = np_emb / np.linalg.norm(np_emb)
                embeddings.append(normed_embedding)

        elif "embedding-2" in google_model:
            embeddings = []
            for doc in documents:
                result = client.models.embed_content(
                    model=google_model,
                    contents=doc,
                    config=types.EmbedContentConfig(output_dimensionality=768),
                )  # type: ignore
                embeddings.append(result.embeddings[0].values)  # type: ignore

        return np.asarray(embeddings)

    def call_llm(
        self,
        prompt: str,
        max_tokens: int = 2048,
        model: str = DEFAULT_LLM_MODEL,
        system_prompt: str = LLM_TEXT_SYSTEM_PROMPT,
        use_cache: bool = True,
    ) -> str:
        request_inputs = {
            "call_type": "litellm_text",
            "model": model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if use_cache:
            cache_key = llm_cache_key(request_inputs)
            cached_value = self.get_cached_value(cache_key)
            if cached_value is not None:
                return cached_value

        if "gemini" in model:
            llm = self.generate_google
        else:
            llm = self.generate_litellm
        response_text = self.run_with_rate_limit_retries(
            lambda: llm(
                prompt=prompt,
                model=model,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
        )
        if use_cache:
            self.store_cached_value(cache_key, response_text)  # type: ignore
        return response_text

    def call_llm_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        max_tokens: int = 4096,
        model: str = DEFAULT_LLM_VISION_MODEL,
        use_cache: bool = True,
    ) -> str:
        request_inputs = {
            "call_type": "litellm_vision",
            "model": model,
            "system_prompt": LLM_VISION_SYSTEM_PROMPT,
            "prompt": prompt,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if use_cache:
            cache_key = llm_cache_key(request_inputs)
            cached_value = self.get_cached_value(cache_key)
            if cached_value is not None:
                return cached_value

        if "gemini" in model:
            llm = self.generate_google
        else:
            llm = self.generate_litellm
        response_text = self.run_with_rate_limit_retries(
            lambda: llm(
                prompt=prompt,
                model=model,
                system_prompt=LLM_VISION_SYSTEM_PROMPT,
                image_bytes=image_bytes,
                max_tokens=max_tokens,
            )
        )
        if use_cache:
            self.store_cached_value(cache_key, response_text)  # type: ignore
        return response_text

    def call_llm_vision_with_usage(
        self,
        prompt: str,
        image_bytes: bytes,
        max_tokens: int = 4096,
        model: str = DEFAULT_LLM_VISION_MODEL,
        use_cache: bool = True,
    ) -> dict:
        request_inputs = {
            "call_type": "litellm_vision_with_usage",
            "model": model,
            "system_prompt": LLM_VISION_SYSTEM_PROMPT,
            "prompt": prompt,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if use_cache:
            cache_key = llm_cache_key(request_inputs)
            cached_value = self.get_cached_value(cache_key)
            if cached_value is not None:
                if isinstance(cached_value, dict):
                    cached_copy = cached_value.copy()
                    cached_copy["cache_hit"] = True
                    return cached_copy
                return {
                    "text": str(cached_value),
                    "usage": {},
                    "cache_hit": True,
                }

        if "gemini" in model:
            llm = self.generate_google_with_usage
        else:
            llm = self.generate_litellm_with_usage
        response_payload = self.run_with_rate_limit_retries(
            lambda: llm(
                prompt=prompt,
                model=model,
                system_prompt=LLM_VISION_SYSTEM_PROMPT,
                image_bytes=image_bytes,
                max_tokens=max_tokens,
            )
        )
        if use_cache:
            self.store_cached_value(cache_key, response_payload)  # type: ignore
        return response_payload

    def batch_call_llm(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        model: str = DEFAULT_LLM_MODEL,
        desc: str = "LLM text batch",
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> list[str]:
        if not prompts:
            return []

        results: list[str | None] = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="llm-text"
        ) as executor:
            futures = {
                executor.submit(
                    self.call_llm,
                    prompt,
                    max_tokens=max_tokens,
                    model=model,
                    use_cache=use_cache,
                ): prompt_idx
                for prompt_idx, prompt in enumerate(prompts)
            }
            completed_futures = concurrent.futures.as_completed(futures)
            if show_progress:
                completed_futures = tqdm(
                    completed_futures,
                    total=len(futures),
                    desc=desc,
                    unit="request",
                )
            for future in completed_futures:
                prompt_idx = futures.pop(future)
                results[prompt_idx] = future.result()

        return [result or "" for result in results]

    def batch_call_llm_vision(
        self,
        requests: list[tuple[str, bytes]],
        max_tokens: int = 4096,
        model: str = DEFAULT_LLM_VISION_MODEL,
        desc: str = "LLM vision batch",
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> list[str]:
        if not requests:
            return []

        results: list[str | None] = [None] * len(requests)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="llm-vision"
        ) as executor:
            futures = {
                executor.submit(
                    self.call_llm_vision,
                    prompt,
                    image_bytes,
                    max_tokens=max_tokens,
                    model=model,
                    use_cache=use_cache,
                ): request_idx
                for request_idx, (prompt, image_bytes) in enumerate(requests)
            }
            completed_futures = concurrent.futures.as_completed(futures)
            if show_progress:
                completed_futures = tqdm(
                    completed_futures,
                    total=len(futures),
                    desc=desc,
                    unit="request",
                )
            for future in completed_futures:
                request_idx = futures.pop(future)
                results[request_idx] = future.result()

        return [result or "" for result in results]

    def batch_call_llm_vision_with_usage(
        self,
        requests: list[tuple[str, bytes]],
        max_tokens: int = 4096,
        model: str = DEFAULT_LLM_VISION_MODEL,
        desc: str = "LLM vision batch",
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> list[dict]:
        if not requests:
            return []

        results: list[dict | None] = [None] * len(requests)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="llm-vision"
        ) as executor:
            futures = {
                executor.submit(
                    self.call_llm_vision_with_usage,
                    prompt,
                    image_bytes,
                    max_tokens=max_tokens,
                    model=model,
                    use_cache=use_cache,
                ): request_idx
                for request_idx, (prompt, image_bytes) in enumerate(requests)
            }
            completed_futures = concurrent.futures.as_completed(futures)
            if show_progress:
                completed_futures = tqdm(
                    completed_futures,
                    total=len(futures),
                    desc=desc,
                    unit="request",
                )
            for future in completed_futures:
                request_idx = futures.pop(future)
                results[request_idx] = future.result()

        return [
            result
            or {
                "text": "",
                "usage": {},
                "cache_hit": False,
            }
            for result in results
        ]

    def embed_batch(
        self, model, batch_texts: list[str], use_cache=True
    ) -> list[tuple[str, np.ndarray]]:
        if "gemini" in model:
            response_embeddings = self.run_with_rate_limit_retries(
                lambda batch_texts=batch_texts: self.embed_google(
                    documents=batch_texts, model=remove_provider(model)
                )
            )
        else:
            response = self.run_with_rate_limit_retries(
                lambda batch_texts=batch_texts: litellm.embedding(
                    model=model, input=batch_texts
                )
            )
            response_embeddings = []
            for item in response.data:
                raw_embedding = (
                    item["embedding"] if isinstance(item, dict) else item.embedding
                )
                response_embeddings.append(np.asarray(raw_embedding, dtype=np.float32))

        batch_embeddings = []
        new_embeddings = {}
        for response_idx, embedding in enumerate(response_embeddings):
            text = batch_texts[response_idx]
            request_inputs = {
                "call_type": "embedding",
                "model": model,
                "text": text,
            }
            cache_key = llm_cache_key(request_inputs) if use_cache else text
            batch_embeddings.append((cache_key, embedding))
            if use_cache:
                new_embeddings[cache_key] = embedding
        if use_cache and self.cache_enabled and new_embeddings:
            with self.cache_lock:
                self.cache.update(new_embeddings)
                self.dirty_cache_entries.update(new_embeddings)
                self.generated_embedding_cache_keys.update(new_embeddings)
        return batch_embeddings

    def embed_texts(
        self,
        texts: list[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        use_cache: bool = True,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        n_workers: int = 1,
    ) -> np.ndarray:
        cleaned_texts = [text.strip() or "empty semantic description" for text in texts]
        embeddings: list[np.ndarray | None] = [None] * len(cleaned_texts)
        uncached_texts: list[str] = []
        uncached_positions_by_key: dict[str, list[int]] = {}

        if use_cache:
            cache_keys = []
            for text in cleaned_texts:
                request_inputs = {
                    "call_type": "embedding",
                    "model": model,
                    "text": text,
                }
                cache_keys.append(llm_cache_key(request_inputs))

            cached_values = self.get_cached_values(cache_keys)
            for text_idx, cached_value in enumerate(cached_values):
                if cached_value is not None and len(cached_value) == 768:
                    embeddings[text_idx] = np.asarray(cached_value, dtype=np.float32)
                else:
                    cache_key = cache_keys[text_idx]
                    if cache_key not in uncached_positions_by_key:
                        uncached_positions_by_key[cache_key] = []
                        uncached_texts.append(cleaned_texts[text_idx])
                    uncached_positions_by_key[cache_key].append(text_idx)
        else:
            for text_idx, text in enumerate(cleaned_texts):
                if text not in uncached_positions_by_key:
                    uncached_positions_by_key[text] = []
                    uncached_texts.append(text)
                uncached_positions_by_key[text].append(text_idx)

        print(f"{len(uncached_texts)} uncached texts to embed.")
        if uncached_texts:
            n_embedding_workers = max(1, n_workers)
            batch_text_groups = [
                uncached_texts[batch_start : batch_start + batch_size]
                for batch_start in range(0, len(uncached_texts), batch_size)
            ]
            n_embedding_workers = min(n_embedding_workers, len(batch_text_groups))

            batches_since_cache_flush = 0
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=n_embedding_workers, thread_name_prefix="llm-embedding"
            ) as executor:
                futures = {
                    executor.submit(
                        self.embed_batch,
                        model,
                        batch_texts,
                        use_cache=use_cache,
                    ): batch_idx
                    for batch_idx, batch_texts in enumerate(batch_text_groups)
                }
                completed_futures = tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Embedding texts",
                    unit="batch",
                )
                for future in completed_futures:
                    embedded_batch = future.result()
                    for cache_key, embedding in embedded_batch:
                        for text_idx in uncached_positions_by_key[cache_key]:
                            embeddings[text_idx] = embedding
                    # if use_cache and self.cache_enabled:
                    # batches_since_cache_flush += 1
                    # if batches_since_cache_flush >= 200:
                    #     self.flush_cache(evict_generated_embeddings=True)
                    #     batches_since_cache_flush = 0

        if use_cache:
            self.flush_cache(evict_generated_embeddings=True, wait=True)

        if len(embeddings) == 0:
            return np.empty((0, 0), dtype=np.float32)
        else:
            embedding_dim = len(embeddings[0])

        embedding_matrix = np.empty((len(embeddings), embedding_dim), dtype=np.float32)

        for embedding_idx, embedding in enumerate(embeddings):
            if embedding is None:
                raise ValueError(f"Missing embedding at position {embedding_idx}")
            embedding_matrix[embedding_idx] = embedding
            embeddings[embedding_idx] = None

        return embedding_matrix


_PROCESS_LLM_WRAPPER = None


def get_llm_wrapper() -> LLMWrapper:
    global _PROCESS_LLM_WRAPPER
    if _PROCESS_LLM_WRAPPER is None:
        _PROCESS_LLM_WRAPPER = LLMWrapper()
    return _PROCESS_LLM_WRAPPER


class LazyLLMWrapper:
    def __getattr__(self, name: str):
        return getattr(get_llm_wrapper(), name)


DEFAULT_LLM_WRAPPER = LazyLLMWrapper()
