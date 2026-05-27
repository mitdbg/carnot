import fcntl
import hashlib
import os
import pickle
import shutil


CACHE_PATH = os.path.expanduser("~/orcd/scratch/skunk_cache/semantic_document_index.pckl")
LOCK_PATH = f"{CACHE_PATH}.lock"
BACKUP_PATH = f"{CACHE_PATH}.raw_sha.bak"
TMP_PATH = f"{CACHE_PATH}.tmp"
OVERWRITE_CACHE = False
MIGRATED_CACHE_PATH = f"{CACHE_PATH}.path_sha"


if not os.path.exists(CACHE_PATH):
    raise FileNotFoundError(f"No cache found at {CACHE_PATH}")

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
with open(LOCK_PATH, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        with open(CACHE_PATH, "rb") as cache_file:
            cache = pickle.load(cache_file)

        documents = cache.get("documents", {})
        if not isinstance(documents, dict):
            raise TypeError("Semantic index cache field 'documents' must be a dict.")

        migrated_sha_to_document_id = {}
        skipped_documents = []
        duplicate_paths = []

        for document_id, document in documents.items():
            filename = getattr(document, "filename", "")
            if not isinstance(filename, str) or not filename.strip():
                skipped_documents.append(document_id)
                continue

            path_sha = hashlib.sha256(os.path.abspath(filename).encode("utf-8")).hexdigest()
            existing_document_id = migrated_sha_to_document_id.get(path_sha)
            if existing_document_id is not None and existing_document_id != document_id:
                duplicate_paths.append((filename, existing_document_id, document_id))
                continue

            migrated_sha_to_document_id[path_sha] = document_id

        if skipped_documents:
            skipped_text = ", ".join(skipped_documents[:10])
            if len(skipped_documents) > 10:
                skipped_text = f"{skipped_text}, ..."
            raise ValueError(f"Cannot migrate documents with missing filenames: {skipped_text}")

        if duplicate_paths:
            duplicate_text = ", ".join(
                f"{filename}: {first_id} / {second_id}"
                for filename, first_id, second_id in duplicate_paths[:10]
            )
            if len(duplicate_paths) > 10:
                duplicate_text = f"{duplicate_text}, ..."
            raise ValueError(f"Cannot migrate duplicate path hashes: {duplicate_text}")

        cache["sha_to_document_id"] = migrated_sha_to_document_id
        cache["document_identity"] = "absolute_path_sha256"

        output_path = CACHE_PATH if OVERWRITE_CACHE else MIGRATED_CACHE_PATH
        if OVERWRITE_CACHE:
            shutil.copy2(CACHE_PATH, BACKUP_PATH)

        with open(TMP_PATH, "wb") as cache_file:
            pickle.dump(cache, cache_file)
        os.replace(TMP_PATH, output_path)

        print(f"Source cache path: {CACHE_PATH}")
        print(f"Migrated cache path: {output_path}")
        print(f"Documents migrated: {len(migrated_sha_to_document_id)}")
        print("Document identity: absolute_path_sha256")
        if OVERWRITE_CACHE:
            print(f"Backup path: {BACKUP_PATH}")
    finally:
        if os.path.exists(TMP_PATH):
            os.remove(TMP_PATH)
        fcntl.flock(lock_file, fcntl.LOCK_UN)
