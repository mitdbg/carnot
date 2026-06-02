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

DEFAULT_LLM_MODEL = "vertex_ai/gemini-3.5-flash"
DEFAULT_LLM_VISION_MODEL = "vertex_ai/gemini-3.5-flash"
DEFAULT_EMBEDDING_MODEL = "vertex_ai/gemini-embedding-001"

DEFAULT_CACHE_DIR = os.path.expanduser("~/orcd/scratch/skunk_cache/")
LLM_CACHE_PATH = os.path.join(DEFAULT_CACHE_DIR, "llm_wrapper_cache.pckl")
LLM_RATE_LIMIT_STATE_PATH = f"{LLM_CACHE_PATH}.rate_limit.json"
LLM_RATE_LIMIT_LOCK_PATH = f"{LLM_RATE_LIMIT_STATE_PATH}.lock"
LLM_TEXT_SYSTEM_PROMPT = "You extract compact metadata for a semantic document index. Return only the requested text."
LLM_VISION_SYSTEM_PROMPT = (
    "You match rendered PDF pages to spans of OCR text. Return only valid JSON."
)

LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", 32))
LLM_MAX_REQUESTS_PER_MINUTE = int(os.environ.get("LLM_MAX_REQUESTS_PER_MINUTE", 100))
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", 15))
LLM_RETRY_BACKOFF_SECONDS = int(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", 5))
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
    elif cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        return json.loads(untruncate_json.complete(cleaned_text))
    except json.JSONDecodeError:
        try:
            return json5.loads(cleaned_text)
        except Exception as e:
            print(f"Failed to parse JSON response: {e}\nOriginal text: {text}")
            return {}


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

    def flush_cache(self, evict_generated_embeddings: bool = False) -> None:
        if not self.cache_enabled:
            return

        with self.cache_lock:
            if not self.dirty_cache_entries:
                return
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
                    print(
                        f"LLM error hit limit hit ({type(e).__name__}: {e}); ",
                        f"{e.with_traceback(None)}",
                        f"retrying in {sleep_seconds:.1f} seconds.",
                    )
                    self.wait_for_rate_limit_slot()
                else:
                    print(
                        f"Transient LLM error ({type(e).__name__}: {e}); "
                        f"retrying in {sleep_seconds:.1f} seconds."
                    )
                self.flush_cache(evict_generated_embeddings=True)
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
        client = genai.Client(vertexai=True, project="mit-grc-free-tier")
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

    def embed_google(self, documents: list[str], model: str):
        client = genai.Client(vertexai=True, project="mit-grc-free-tier")
        if "embedding-001" in model:
            result = client.models.embed_content(
                model=model, contents=documents  # type: ignore
            )
            embeddings = [e.values for e in result.embeddings]  # type: ignore
        elif "embedding-2" in model:
            embeddings = []
            for doc in documents:
                result = client.models.embed_content(model=model, contents=documents)  # type: ignore
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
                prompt_idx = futures[future]
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
                request_idx = futures[future]
                results[request_idx] = future.result()

        return [result or "" for result in results]

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
                if cached_value is not None:
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
                    if use_cache and self.cache_enabled:
                        batches_since_cache_flush += 1
                        if batches_since_cache_flush >= 500:
                            self.flush_cache(evict_generated_embeddings=True)
                            batches_since_cache_flush = 0

        self.flush_cache(evict_generated_embeddings=True)

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
