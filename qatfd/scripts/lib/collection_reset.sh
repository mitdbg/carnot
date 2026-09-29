# Shared by the sweep drivers whose runs build collections on a shared chroma server (the Bootstrap / Enrich
# collection agents): put the server back to "only the corpus" before a fresh run. Source it AFTER setting
#   PYTHON, QATFD_DIR, CHROMA_HOST, CHROMA_PORT, BASE_COLLECTION, KEEP_COLLECTIONS
# (KEEP_COLLECTIONS: the other corpus copies served from the same store, never deleted).

# delete every collection except the corpus collections, so a run only sees the collections it builds itself
wipe_collections() {
    "$PYTHON" - "$CHROMA_HOST" "$CHROMA_PORT" "$BASE_COLLECTION" $KEEP_COLLECTIONS <<'PY'
import sys, time
from skunk.chroma_client import make_chroma_client
from chromadb.errors import NotFoundError
host, port, base = sys.argv[1], int(sys.argv[2]), sys.argv[3]
keep = {base, *sys.argv[4:]}  # the active corpus + every other corpus copy served from this store
client = make_chroma_client(host, port)
client.get_collection(base)  # raises if the base corpus is not served here
names, offset = [], 0
while True:
    page = client.list_collections(limit=100, offset=offset)
    names.extend(c.name for c in page if c.name not in keep)
    if len(page) < 100:
        break
    offset += 100
failed = []
for name in names:
    for attempt in range(3):  # a delete can 500 transiently (e.g. the server is still compacting it)
        try:
            client.delete_collection(name)
            break
        except NotFoundError:
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                failed.append((name, f"{type(e).__name__}: {e}"))
            else:
                time.sleep(5)
left = [c.name for c in client.list_collections(limit=100) if c.name not in keep]
print(f"[sweep] wiped {len(names) - len(failed)} collection(s); kept {sorted(keep)}; {len(left)} other collection(s) remain")
for name, err in failed:
    print(f"[sweep] delete failed for {name!r}: {err}", file=sys.stderr)
sys.exit(1 if left else 0)
PY
}

# strip agent-written chunk metadata (a map / semantic_map over the corpus) from BASE_COLLECTION: a -v1
# copy is stripped, any other known collection is only checked (see engaging-scripts/reset_corpus_metadata.py)
reset_corpus_metadata() {
    "$PYTHON" "$QATFD_DIR/engaging-scripts/reset_corpus_metadata.py" --collection "$BASE_COLLECTION" --server "$CHROMA_HOST:$CHROMA_PORT"
}
