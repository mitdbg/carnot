"""Copy each benchmark's BASE chroma collection into a sibling `<benchmark>-<embed_model>-v1` collection
(same store) that keeps ONLY the `doc_id` / `chunk_id` / `element_id` metadata keys.

Everything else is carried over unchanged: row ids, documents, embeddings, and the HNSW config of the
base collection (l2 / ef_construction 100 / max_neighbors 16 / ef_search 100). The copy's collection
metadata gets a `fields` schema restricted to the three kept keys (taken from set_collection_fields.py,
the single source of truth) and, like the base, NO description.

There is no storage-level clone in chroma, so this re-reads every row from the base collection (ids
enumerated straight from chroma.sqlite3 — `get(offset=)` is O(N) per call) and upserts it into the
copy, which rebuilds the HNSW index. Resumable: rows already present in the copy are skipped.

Client: pass `--server HOST:PORT` when a chroma server already holds the store (e.g. the eval's warm
server on 8001 — a second client on one directory is the cross-process lock that hangs both); otherwise
the embedded PersistentClient is used and the script refuses to start if a `chroma run` over that dir
is running.

Usage (from the qatfd dir):
  python3 engaging-scripts/copy_collection_v1.py --benchmark financebench
  python3 engaging-scripts/copy_collection_v1.py --benchmark officeqa --server 127.0.0.1:8001
  python3 engaging-scripts/copy_collection_v1.py --benchmark officeqa --server 127.0.0.1:8001 --verify-only
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from set_collection_fields import TARGETS, _stored_fields  # noqa: E402

KEEP = ("doc_id", "chunk_id", "element_id")
BATCH = 5461  # chroma's max batch size for these stores (client.get_max_batch_size())

# base collection name -> v1 copy name (`{benchmark}-{embed_model}-v1`)
V1_NAMES: dict[str, str] = {
    "officeqa-qwen-8b": "officeqa-qwen-8b-v1",
    "browsecomp-plus-qwen-8b": "browsecomp-plus-qwen-8b-v1",
    "financebench-qwen-8b": "financebench-qwen-8b-v1",
    "freshstack-laravel-qwen-0.6b": "freshstack-laravel-qwen-0.6b-v1",
    "freshstack-langchain-qwen-0.6b": "freshstack-langchain-qwen-0.6b-v1",
    "qwen-qampari-0.6b": "qampari-qwen-0.6b-v1",
}

# HNSW config shared by every base collection (see collections.schema_str); replicated on the copies.
HNSW = {"space": "l2", "ef_construction": 100, "max_neighbors": 16, "ef_search": 100}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def v1_fields(fields: str) -> str:
    """The base `fields` schema restricted to the kept keys (one '  - name (type): ...' line each)."""
    lines = [ln for ln in fields.splitlines() if ln.strip().split(" ")[1] in KEEP]
    assert len(lines) == len(KEEP), lines
    return "".join(ln + "\n" for ln in lines)


def collection_ids(chroma_dir: str, name: str) -> list[str] | None:
    """All row ids of `name`, read straight from chroma.sqlite3 (None if the collection doesn't exist).
    Materialized and the connection closed so no long-lived read snapshot pins the WAL."""
    con = sqlite3.connect(f"file:{chroma_dir}/chroma.sqlite3?mode=ro", uri=True)
    try:
        row = con.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        seg = con.execute("SELECT id FROM segments WHERE collection=? AND scope='METADATA'", (row[0],)).fetchone()[0]
        return [r[0] for r in con.execute("SELECT embedding_id FROM embeddings WHERE segment_id=? ORDER BY id", (seg,))]
    finally:
        con.close()


def make_client(chroma_dir: str, server: str | None):
    import chromadb

    if server:
        host, port = server.rsplit(":", 1)
        return chromadb.HttpClient(host=host, port=int(port))
    probe = subprocess.run(["pgrep", "-af", "chroma run"], capture_output=True, text=True).stdout
    if os.path.abspath(chroma_dir) in probe:
        raise SystemExit(f"a chroma server holds {chroma_dir}; pass --server HOST:PORT instead of opening it embedded")
    return chromadb.PersistentClient(path=chroma_dir)


def copy_one(chroma_dir: str, base_name: str, fields: str, server: str | None, threads: int, batch: int) -> None:
    target_name = V1_NAMES[base_name]
    client = make_client(chroma_dir, server)
    base = client.get_collection(base_name)
    target = client.get_or_create_collection(
        name=target_name,
        configuration={"hnsw": {**HNSW, "num_threads": threads}},
        metadata={"fields": v1_fields(fields)},
    )
    log(f"{base_name} -> {target_name} in {chroma_dir} ({'server ' + server if server else 'embedded'})")

    all_ids = collection_ids(chroma_dir, base_name)
    assert all_ids is not None
    done = set(collection_ids(chroma_dir, target_name) or [])
    todo = [i for i in all_ids if i not in done] if done else all_ids
    log(f"base rows={len(all_ids):,}  already copied={len(done):,}  to copy={len(todo):,}")
    del done

    t0 = time.time()
    copied = 0
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        got = base.get(ids=chunk, include=["embeddings", "documents", "metadatas"])
        assert len(got["ids"]) == len(chunk), (len(got["ids"]), len(chunk))
        metas = []
        for m in got["metadatas"]:
            missing = [k for k in KEEP if k not in m]
            assert not missing, f"row missing {missing}: {m}"
            metas.append({k: m[k] for k in KEEP})
        target.upsert(
            ids=got["ids"],
            embeddings=np.asarray(got["embeddings"], dtype=np.float32),
            documents=got["documents"],
            metadatas=metas,
        )
        copied += len(chunk)
        if (start // batch) % max(1, 30_000 // batch) == 0 or copied == len(todo):
            el = time.time() - t0
            rate = copied / el if el else 0.0
            eta = (len(todo) - copied) / rate / 60 if rate else float("nan")
            log(f"  {copied:,}/{len(todo):,} rows  {rate:,.0f} rows/s  eta {eta:.1f} min  free {shutil.disk_usage(chroma_dir).free / 1e9:.0f} GB")
    log(f"copy done in {(time.time() - t0) / 60:.1f} min")


def verify_one(chroma_dir: str, base_name: str, fields: str, server: str | None, n_sample: int = 500, n_query: int = 20) -> bool:
    target_name = V1_NAMES[base_name]
    client = make_client(chroma_dir, server)
    base, target = client.get_collection(base_name), client.get_collection(target_name)
    ok = True

    all_ids = collection_ids(chroma_dir, base_name) or []
    tgt_ids = collection_ids(chroma_dir, target_name) or []
    same_ids = set(all_ids) == set(tgt_ids)
    log(f"[verify] {target_name}: base rows={len(all_ids):,} copy rows={len(tgt_ids):,} count()={target.count():,} same id set={same_ids}")
    ok &= same_ids and target.count() == len(all_ids)

    stored, meta = _stored_fields(chroma_dir, target_name)
    fields_ok = stored == v1_fields(fields) and "description" not in meta
    log(f"[verify] fields metadata ok={fields_ok}; other metadata keys={sorted(k for k in meta if k != 'fields')}")
    ok &= fields_ok
    log(f"[verify] hnsw config: {target.configuration_json.get('hnsw')}")

    rng = random.Random(0)
    sample = rng.sample(all_ids, min(n_sample, len(all_ids)))
    b = base.get(ids=sample, include=["embeddings", "documents", "metadatas"])
    t = target.get(ids=sample, include=["embeddings", "documents", "metadatas"])
    bi = {i: k for k, i in enumerate(b["ids"])}
    ti = {i: k for k, i in enumerate(t["ids"])}
    bad = 0
    for i in sample:
        bb, tt = bi[i], ti[i]
        if not np.allclose(b["embeddings"][bb], t["embeddings"][tt], atol=1e-6):
            bad += 1
        elif b["documents"][bb] != t["documents"][tt]:
            bad += 1
        elif {k: b["metadatas"][bb][k] for k in KEEP} != t["metadatas"][tt] or set(t["metadatas"][tt]) != set(KEEP):
            bad += 1
    log(f"[verify] sampled {len(sample)} rows: {len(sample) - bad} identical (embedding+document+kept metadata), {bad} mismatched")
    ok &= bad == 0

    overlaps = []
    for i in rng.sample(all_ids, min(n_query, len(all_ids))):
        q = base.get(ids=[i], include=["embeddings"])["embeddings"][0]
        rb = base.query(query_embeddings=[q], n_results=10, include=[])["ids"][0]
        rt = target.query(query_embeddings=[q], n_results=10, include=[])["ids"][0]
        overlaps.append(len(set(rb) & set(rt)) / 10)
    log(f"[verify] top-10 overlap base vs copy over {len(overlaps)} self-queries: mean {np.mean(overlaps):.2f} min {min(overlaps):.2f}")
    log(f"[verify] {'PASS' if ok else 'FAIL'}: {target_name}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=sorted(k for k in TARGETS if k != "trec_biogen"), action="append", required=True)
    parser.add_argument("--server", help="HOST:PORT of a chroma server already holding the store (else embedded)")
    parser.add_argument("--threads", type=int, default=16, help="hnsw num_threads for the copy's index build")
    parser.add_argument("--batch", type=int, default=BATCH, help="rows per upsert; the server rejects >~32MB bodies, so use ~800 for 4096-d via --server")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--store", help="override the benchmark's chroma dir (e.g. a /dev/shm copy of the store for a fast build); one benchmark only")
    args = parser.parse_args()

    if args.store and (len(args.benchmark) != 1 or len(TARGETS[args.benchmark[0]]) != 1):
        parser.error("--store applies to exactly one benchmark with a single collection")
    ok = True
    for bench in args.benchmark:
        for chroma_dir, name, fields in TARGETS[bench]:
            chroma_dir = args.store or chroma_dir
            if not args.verify_only:
                copy_one(chroma_dir, name, fields, args.server, args.threads, args.batch)
            ok &= verify_one(chroma_dir, name, fields, args.server)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
