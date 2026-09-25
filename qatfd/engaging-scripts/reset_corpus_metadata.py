"""Strip agent-written chunk metadata from a `-v1` corpus collection between experiments.

The Bootstrap / Enrich agents may run `map_collections` / `semantic_map` over the base collection of a
run (nothing in the tools or prompts stops them). Such a map writes new metadata keys onto every chunk
and merges the field names into the collection's `fields` schema, and those keys would then leak into
every later run on the same store. This script puts a corpus collection back to its pristine state:

  1. detect: one read-only query over the store's chroma.sqlite3 finds every chunk that carries a
     metadata key outside the collection's canonical schema (`get(offset=)` is O(N) per call, so the
     rows are never paged through the server);
  2. strip: those keys are deleted through the chroma SERVER (`update(metadatas={key: None})` deletes
     a key in chromadb >= 1.x), in batches, so the server's own caches stay consistent;
  3. reset the schema: `collection.metadata["fields"]` is rewritten to the canonical schema (every
     other collection-metadata key is preserved and reported, never removed);
  4. verify: the sqlite scan is repeated and a random sample of rows is re-read through the server.

ONLY the `-v1` copies (copy_collection_v1.V1_NAMES) may be stripped: their canonical schema is exactly
`doc_id` / `chunk_id` / `element_id`, so everything else on a chunk is agent-written by construction.
Any other collection known to set_collection_fields.TARGETS (the original base collections) is only
ever CHECKED against its own native schema; if it turns out to be polluted the script reports it and
exits non-zero, because there is no safe way to tell an agent-written key from a native one there.

The store must be on this machine (the sqlite scan reads it directly, read-only, which is safe next to
a running server — copy_collection_v1.py / clean_chroma_orphans.py do the same). Writes go through
`--server` (the eval's warm server); without it the embedded PersistentClient is used and the script
refuses to start if a `chroma run` holds the store.

Usage (from the qatfd dir):
  python3 engaging-scripts/reset_corpus_metadata.py --collection officeqa-qwen-8b-v1 --server 127.0.0.1:8001
  python3 engaging-scripts/reset_corpus_metadata.py --collection officeqa-qwen-8b-v1 --server 127.0.0.1:8001 --check
  python3 engaging-scripts/reset_corpus_metadata.py --collection officeqa-qwen-8b --server 127.0.0.1:8001   # check only
  python3 engaging-scripts/reset_corpus_metadata.py --list

Exit status: 0 iff every requested collection ends up clean (no stray chunk keys, canonical schema).
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copy_collection_v1 import KEEP, V1_NAMES, make_client, v1_fields  # noqa: E402
from set_collection_fields import TARGETS  # noqa: E402

# every collection this script knows: name -> (chroma_dir relative to qatfd/, canonical `fields` schema)
KNOWN: dict[str, tuple[str, str]] = {}
for _bench, _targets in TARGETS.items():
    for _chroma_dir, _base_name, _fields in _targets:
        KNOWN[_base_name] = (_chroma_dir, _fields)
        if _base_name in V1_NAMES:
            KNOWN[V1_NAMES[_base_name]] = (_chroma_dir, v1_fields(_fields))

# the explicit allowlist of collections whose stray chunk metadata may be deleted: the `-v1` copies only
STRIPPABLE: frozenset[str] = frozenset(V1_NAMES.values())
assert all(name.endswith("-v1") for name in STRIPPABLE), sorted(STRIPPABLE)
assert STRIPPABLE <= KNOWN.keys(), sorted(STRIPPABLE - KNOWN.keys())

# metadata keys chroma itself stores in embedding_metadata (the document text, uri, ...): never touched
_INTERNAL_PREFIX = "chroma:"
_FIELD_LINE_RE = re.compile(r"^\s*-\s*([^\s(]+)")
_DEFAULT_BATCH = 5000
_VERIFY_RETRIES = 10  # sqlite re-scans after the strip (the server's write may land a moment later)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def field_names(fields: str) -> list[str]:
    """The key names of a `fields` schema string ("  - name (type): desc" lines)."""
    return [m.group(1) for ln in fields.splitlines() if (m := _FIELD_LINE_RE.match(ln))]


def allowed_keys(name: str) -> frozenset[str]:
    """The chunk-metadata keys that legitimately exist on `name`: exactly KEEP for a -v1 copy, the
    native schema for an original base collection."""
    if name in STRIPPABLE:
        return frozenset(KEEP)
    return frozenset(field_names(KNOWN[name][1]))


def find_stray_keys(chroma_dir: str, name: str, allowed: frozenset[str]) -> dict[str, list[str]]:
    """embedding_id -> its metadata keys outside `allowed`, read straight from chroma.sqlite3 (read-only)."""
    con = sqlite3.connect(f"file:{chroma_dir}/chroma.sqlite3?mode=ro", uri=True)
    try:
        row = con.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
        if row is None:
            raise SystemExit(f"collection {name!r} not found in {chroma_dir}")
        seg = con.execute("SELECT id FROM segments WHERE collection=? AND scope='METADATA'", (row[0],)).fetchone()[0]
        placeholders = ",".join("?" * len(allowed))
        rows = con.execute(
            "SELECT e.embedding_id, m.key FROM embeddings e JOIN embedding_metadata m ON m.id = e.id "
            f"WHERE e.segment_id=? AND m.key NOT IN ({placeholders}) AND m.key NOT LIKE ?",
            (seg, *sorted(allowed), f"{_INTERNAL_PREFIX}%"),
        ).fetchall()
    finally:
        con.close()
    stray: dict[str, list[str]] = {}
    for embedding_id, key in rows:
        stray.setdefault(embedding_id, []).append(key)
    return stray


def strip_keys(collection, stray: dict[str, list[str]], batch: int) -> None:
    """Delete the listed keys from each chunk through the server: a `None` value in `update` deletes it."""
    ids = sorted(stray)
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        collection.update(ids=chunk, metadatas=[dict.fromkeys(stray[i], None) for i in chunk])


def reset_schema(collection, canonical: str) -> list[str]:
    """Rewrite `fields` to `canonical`, keeping every other collection-metadata key; returns those keys."""
    meta = dict(collection.metadata or {})
    others = sorted(k for k in meta if k != "fields")
    if meta.get("fields") != canonical:
        collection.modify(metadata={**meta, "fields": canonical})
    return others


def _sample_via_server(collection, ids: list[str], allowed: frozenset[str], n: int) -> list[str]:
    """Ids among a random sample of `ids` that still carry a disallowed key when read through the server."""
    if not ids:
        return []
    sample = random.Random(0).sample(ids, min(n, len(ids)))
    got = collection.get(ids=sample, include=["metadatas"])
    return [i for i, m in zip(got["ids"], got["metadatas"], strict=True) if set(m or {}) - allowed]


def reset_one(client, chroma_dir: str, name: str, *, check: bool, batch: int = _DEFAULT_BATCH, sample: int = 200) -> bool:
    """Detect (and unless `check`, strip) stray chunk keys on `name`; True iff it ends up clean."""
    if name not in KNOWN:
        raise SystemExit(f"unknown collection {name!r}; known: {sorted(KNOWN)}")
    canonical = KNOWN[name][1]
    allowed = allowed_keys(name)
    strippable = name in STRIPPABLE
    mode = "check" if check else ("strip" if strippable else "check (not in the -v1 allowlist: never stripped)")
    log(f"{name} [{mode}] in {chroma_dir}: allowed keys {sorted(allowed)}")

    t0 = time.time()
    stray = find_stray_keys(chroma_dir, name, allowed)
    keys = sorted({k for ks in stray.values() for k in ks})
    collection = client.get_collection(name)
    stored_fields = (collection.metadata or {}).get("fields")
    schema_ok = stored_fields == canonical
    log(f"  scan {time.time() - t0:.1f}s: {len(stray):,} chunk(s) with stray key(s) {keys}; schema {'canonical' if schema_ok else 'DIFFERS'}")
    if not schema_ok:
        log(f"  stored schema fields: {field_names(stored_fields or '')}")

    if check or not strippable:
        clean = not stray and schema_ok
        if not clean and not strippable and not check:
            log(f"  ERROR: {name} is polluted but only the -v1 copies may be stripped; rebuild it or strip by hand")
        return clean

    if stray:
        t0 = time.time()
        strip_keys(collection, stray, batch)
        log(f"  stripped {len(keys)} key(s) from {len(stray):,} chunk(s) in {time.time() - t0:.1f}s")
    others = reset_schema(collection, canonical)
    if not schema_ok:
        log(f"  schema reset to {field_names(canonical)}; other collection-metadata keys kept: {others}")

    # verify: the server must show the keys gone on a sample, and the sqlite scan must come back empty
    # (the server's write can land in sqlite a moment after the call returns, hence the retries)
    bad = _sample_via_server(collection, sorted(stray), allowed, sample)
    if bad:
        log(f"  ERROR: {len(bad)} sampled chunk(s) still carry stray keys via the server, e.g. {bad[:3]}")
        return False
    for attempt in range(_VERIFY_RETRIES):
        left = find_stray_keys(chroma_dir, name, allowed)
        if not left:
            break
        if attempt == _VERIFY_RETRIES - 1:
            log(f"  ERROR: {len(left):,} chunk(s) still carry stray keys in sqlite after the strip, e.g. {sorted(left)[:3]}")
            return False
        time.sleep(1)
    schema_ok = (client.get_collection(name).metadata or {}).get("fields") == canonical
    if not schema_ok:
        log("  ERROR: schema still differs from the canonical fields after the reset")
    log(f"  {'CLEAN' if schema_ok else 'FAILED'}: {name}")
    return schema_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", action="append", default=[], help="collection to reset (repeatable); see --list")
    parser.add_argument("--server", help="HOST:PORT of the chroma server holding the store (else embedded)")
    parser.add_argument("--store", help="override the collection's chroma dir (one collection only)")
    parser.add_argument("--check", action="store_true", help="detect and report only; write nothing")
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH, help="chunks per update call")
    parser.add_argument("--sample", type=int, default=200, help="stripped chunks re-read through the server to verify")
    parser.add_argument("--list", action="store_true", help="print the known collections and exit")
    args = parser.parse_args()

    if args.list:
        for name in sorted(KNOWN):
            print(f"{name}: {KNOWN[name][0]}  ({'STRIPPABLE' if name in STRIPPABLE else 'check only'}; keys {sorted(allowed_keys(name))})")
        return 0
    names = list(dict.fromkeys(args.collection))
    if not names:
        parser.error("pass --collection NAME (repeatable) or --list")
    unknown = [n for n in names if n not in KNOWN]
    if unknown:
        parser.error(f"unknown collection(s) {unknown}; known: {sorted(KNOWN)}")
    if args.store and len(names) != 1:
        parser.error("--store applies to exactly one collection")

    ok = True
    for name in names:
        chroma_dir = args.store or KNOWN[name][0]
        if not os.path.isfile(os.path.join(chroma_dir, "chroma.sqlite3")):
            log(f"ERROR: no chroma.sqlite3 under {chroma_dir} (the store must be on this machine; see --store)")
            ok = False
            continue
        client = make_client(chroma_dir, args.server)
        ok &= reset_one(client, chroma_dir, name, check=args.check, batch=args.batch, sample=args.sample)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
