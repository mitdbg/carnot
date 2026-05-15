import base64
import concurrent.futures
import contextlib
import fcntl
import hashlib
import json
import os
import pickle
import re
import threading
import time

import litellm
import untruncate_json
from tqdm import tqdm

litellm.suppress_debug_info = True

# DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
# DEFAULT_LLM_VISION_MODEL = "openai/gpt-4o-mini"
# DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_LLM_MODEL = "openrouter/google/gemini-2.5-flash"
DEFAULT_LLM_VISION_MODEL = "openrouter/google/gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "openrouter/google/gemini-embedding-001"

EMBEDDING_BATCH_SIZE = 64

LLM_CACHE_PATH = os.environ.get(
    "SKUNK_LLM_CACHE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache", "llm_call_cache.pckl")),
)
LLM_RATE_LIMIT_STATE_PATH = f"{LLM_CACHE_PATH}.rate_limit.json"
LLM_RATE_LIMIT_LOCK_PATH = f"{LLM_RATE_LIMIT_STATE_PATH}.lock"
LLM_TEXT_SYSTEM_PROMPT = "You extract compact metadata for a semantic document index. Return only the requested text."
LLM_VISION_SYSTEM_PROMPT = "You match rendered PDF pages to spans of OCR text. Return only valid JSON."

DEFAULT_LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "32"))
DEFAULT_LLM_MAX_REQUESTS_PER_MINUTE = int(os.environ.get("LLM_MAX_REQUESTS_PER_MINUTE", "500"))
DEFAULT_LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "10"))
DEFAULT_LLM_RETRY_BACKOFF_SECONDS = int(os.environ.get("LLM_RETRY_BACKOFF_SECONDS", "10"))


def llm_cache_key(request_inputs: dict) -> str:
    request_json = json.dumps(request_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(request_json.encode("utf-8")).hexdigest()


def extract_litellm_text(response) -> str:
    choice = response.choices[0] if hasattr(response, "choices") else response["choices"][0]
    message = choice.message if hasattr(choice, "message") else choice.get("message", {})
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

    raise ValueError(f"No text content in LiteLLM response choice: {choice}")


def parse_json_response(text: str) -> dict:
    cleaned_text = text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = re.sub(r"^```(?:json)?\s*", "", cleaned_text)
        cleaned_text = re.sub(r"\s*```$", "", cleaned_text)

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        try:
            return json.loads(untruncate_json.complete(cleaned_text))
        except Exception as e:
            print(f"Failed to parse JSON response: {e}\nOriginal text: {text}")
            return {}


class LLMWrapper:
    def __init__(
        self,
        max_workers: int = DEFAULT_LLM_MAX_WORKERS,
        max_requests_per_minute: int = DEFAULT_LLM_MAX_REQUESTS_PER_MINUTE,
        max_retries: int = DEFAULT_LLM_MAX_RETRIES,
        retry_backoff_seconds: int = DEFAULT_LLM_RETRY_BACKOFF_SECONDS,
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

    def store_cached_value(self, cache_key: str, value) -> None:
        if not self.cache_enabled:
            return

        with self.cache_lock:
            self.cache[cache_key] = value
            self.dirty_cache_entries[cache_key] = value

        self.flush_cache()

    def flush_cache(self) -> None:
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
            for cache_key, value in disk_cache.items():
                if cache_key not in self.dirty_cache_entries:
                    self.cache[cache_key] = value
                elif self.dirty_cache_entries[cache_key] == dirty_cache_entries.get(cache_key):
                    self.cache[cache_key] = value
            for cache_key in dirty_cache_entries:
                if self.dirty_cache_entries.get(cache_key) == dirty_cache_entries[cache_key]:
                    self.dirty_cache_entries.pop(cache_key, None)

    def is_rate_limit_error(self, error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code == 429:
            return True

        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True

        error_text = str(error).lower()
        return "429" in error_text or "rate limit" in error_text or "too many requests" in error_text

    def run_with_rate_limit_retries(self, request_fn):
        attempt_idx = 0
        while True:
            try:
                self.wait_for_rate_limit_slot()
                return request_fn()
            except Exception as e:
                if not self.is_rate_limit_error(e) or attempt_idx >= self.max_retries:
                    raise
                sleep_seconds = self.retry_backoff_seconds * (attempt_idx + 1)
                print(f"LLM rate limit hit; retrying in {sleep_seconds} seconds.")
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
                            request_times = json.load(state_file).get("request_times", [])
                    except (json.JSONDecodeError, OSError):
                        request_times = []

                request_times = [request_time for request_time in request_times if now - request_time < 60]
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

    def call_llm(self, prompt: str, max_tokens: int = 2048, model: str = DEFAULT_LLM_MODEL) -> str:
        request_inputs = {
            "call_type": "litellm_text",
            "model": model,
            "system_prompt": LLM_TEXT_SYSTEM_PROMPT,
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        cache_key = llm_cache_key(request_inputs)
        cached_value = self.get_cached_value(cache_key)
        if cached_value is not None:
            return cached_value

        response = self.run_with_rate_limit_retries(
            lambda: litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": LLM_TEXT_SYSTEM_PROMPT,
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max_tokens,
                timeout=self.text_timeout,
            )
        )
        response_text = extract_litellm_text(response)
        self.store_cached_value(cache_key, response_text)
        return response_text

    def call_llm_vision(
        self,
        prompt: str,
        image_bytes: bytes,
        max_tokens: int = 4096,
        model: str = DEFAULT_LLM_VISION_MODEL,
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
        cache_key = llm_cache_key(request_inputs)
        cached_value = self.get_cached_value(cache_key)
        if cached_value is not None:
            return cached_value

        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        response = self.run_with_rate_limit_retries(
            lambda: litellm.completion(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": LLM_VISION_SYSTEM_PROMPT,
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
                temperature=0,
                max_tokens=max_tokens,
                timeout=self.vision_timeout,
            )
        )
        response_text = extract_litellm_text(response)
        self.store_cached_value(cache_key, response_text)
        return response_text

    def batch_call_llm(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        model: str = DEFAULT_LLM_MODEL,
        desc: str = "LLM text batch",
        show_progress: bool = True,
    ) -> list[str]:
        if not prompts:
            return []

        results: list[str | None] = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="llm-text") as executor:
            futures = {
                executor.submit(self.call_llm, prompt, max_tokens, model): prompt_idx
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
    ) -> list[str]:
        if not requests:
            return []

        results: list[str | None] = [None] * len(requests)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="llm-vision") as executor:
            futures = {
                executor.submit(self.call_llm_vision, prompt, image_bytes, max_tokens, model): request_idx
                for request_idx, (prompt, image_bytes) in enumerate(requests)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="request",
            ):
                request_idx = futures[future]
                results[request_idx] = future.result()

        return [result or "" for result in results]

    def embed_texts(self, texts: list[str], model: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
        cleaned_texts = [text.strip() or "empty semantic description" for text in texts]
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
            cached_value = self.get_cached_value(cache_key)
            if cached_value is not None:
                embeddings[text_idx] = cached_value
            else:
                uncached_texts.append(text)
                uncached_positions.append((text_idx, cache_key))

        if uncached_texts:
            new_embeddings = {}
            for batch_start in range(0, len(uncached_texts), EMBEDDING_BATCH_SIZE):
                batch_texts = uncached_texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                batch_positions = uncached_positions[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                response = self.run_with_rate_limit_retries(lambda: litellm.embedding(model=model, input=batch_texts))
                for response_idx, item in enumerate(response.data):
                    embedding = item["embedding"] if isinstance(item, dict) else item.embedding
                    text_idx, cache_key = batch_positions[response_idx]
                    new_embeddings[cache_key] = embedding
                    embeddings[text_idx] = embedding
            if self.cache_enabled:
                with self.cache_lock:
                    self.cache.update(new_embeddings)
                    self.dirty_cache_entries.update(new_embeddings)
                self.flush_cache()

        return [embedding for embedding in embeddings if embedding is not None]


DEFAULT_LLM_WRAPPER = LLMWrapper()
