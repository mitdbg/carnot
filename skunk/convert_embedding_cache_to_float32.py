import fcntl
import os
import pickle
import shutil

import numpy as np


CACHE_PATH = os.environ.get(
    "SKUNK_LLM_CACHE_PATH",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "cache",
            "llm_call_cache.pckl",
        )
    ),
)
LOCK_PATH = f"{CACHE_PATH}.lock"
BACKUP_PATH = f"{CACHE_PATH}.bak"
TMP_PATH = f"{CACHE_PATH}.tmp"


os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
with open(LOCK_PATH, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        if not os.path.exists(CACHE_PATH):
            raise FileNotFoundError(f"No cache found at {CACHE_PATH}")

        with open(CACHE_PATH, "rb") as cache_file:
            cache = pickle.load(cache_file)

        converted_count = 0
        skipped_count = 0
        for cache_key, cached_value in list(cache.items()):
            if isinstance(cached_value, np.ndarray):
                if cached_value.ndim == 1 and cached_value.dtype != np.float32:
                    cache[cache_key] = cached_value.astype(np.float32, copy=False)
                    converted_count += 1
                else:
                    skipped_count += 1
                continue

            if not isinstance(cached_value, (list, tuple)) or not cached_value:
                skipped_count += 1
                continue

            try:
                embedding = np.asarray(cached_value, dtype=np.float32)
            except (TypeError, ValueError):
                skipped_count += 1
                continue

            if embedding.ndim != 1:
                skipped_count += 1
                continue

            cache[cache_key] = embedding
            converted_count += 1

        if converted_count:
            shutil.copy2(CACHE_PATH, BACKUP_PATH)
            with open(TMP_PATH, "wb") as cache_file:
                pickle.dump(cache, cache_file)
            os.replace(TMP_PATH, CACHE_PATH)
        elif os.path.exists(TMP_PATH):
            os.remove(TMP_PATH)

        print(f"Cache path: {CACHE_PATH}")
        print(f"Converted embedding entries: {converted_count}")
        print(f"Skipped non-embedding entries: {skipped_count}")
        if converted_count:
            print(f"Backup path: {BACKUP_PATH}")
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
