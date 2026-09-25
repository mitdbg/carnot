"""Set (or check) the `fields` metadata schema on each benchmark's BASE chroma collection.

The Bootstrap / Enrich / Search agents read `collection.metadata["fields"]` to learn which chunk-metadata
keys they can filter on (one "  - name (type): description" line per key; chroma collection metadata
cannot hold lists). Only `fields` is written — base collections deliberately carry NO description:
everything an agent learns about a corpus must come from the prompts' additional notes.

Runs the embedded PersistentClient, so no chroma server may hold the store while this runs. For the
sharded TREC-BioGen corpus every per-rank shard collection gets the same schema (MergedCollection reads
metadata from its first shard).

Usage (from the qatfd dir):
  python3 engaging-scripts/set_collection_fields.py --benchmark officeqa           # write
  python3 engaging-scripts/set_collection_fields.py --benchmark officeqa --check   # verify only
  python3 engaging-scripts/set_collection_fields.py --all --check
"""

from __future__ import annotations

import argparse
import sys

FRESHSTACK_FIELDS = (
    '  - doc_id (str): the source file the chunk belongs to, "<repo>/<path>" (e.g. "breeze/CHANGELOG.md"); every chunk of a file shares it\n'
    '  - chunk_id (str): "<doc_id>_<start_byte>_<end_byte>", the byte-range slice of the source file that this chunk holds\n'
    "  - file_id (str): the source file, identical to doc_id\n"
    "  - url (str): the GitHub URL of the source file (exact-match / $in filtering only)\n"
    "  - element_id (int): the chunk's start byte offset within its file; orders the chunks of a file\n"
)

# benchmark -> list of (chroma_dir relative to qatfd/, collection name, fields)
TARGETS: dict[str, list[tuple[str, str, str]]] = {
    "officeqa": [(
        "benchmarks/officeqa/chromadb",
        "officeqa-qwen-8b",
        '  - doc_id (str): the page key "<year>_<month>_<page_id>" (e.g. "2002_12_26" = page 26 of the December 2002 bulletin); every chunk on a page shares it\n'
        '  - chunk_id (str): "treasury_bulletin_<year>_<month>_<page_id>_<element_id>", the id of this chunk (page element)\n'
        '  - file_id (str): the source bulletin, "treasury_bulletin_<year>_<month>" (e.g. "treasury_bulletin_1939_01")\n'
        '  - year (str): zero-padded publication year of the bulletin, e.g. "2010"\n'
        '  - month (str): zero-padded publication month of the bulletin, e.g. "03" (only "03", "06", "09", "12" from 1983 onward)\n'
        "  - page_id (int): the index of the page within its bulletin (not the printed page number)\n"
        "  - element_id (int): 0-based index of the chunk within its page\n"
        '  - type (str): the element type, e.g. "text", "title", "section_header", "table", "footnote"\n',
    )],
    "browsecomp_plus": [(
        "benchmarks/browsecomp-plus/chromadb",
        "browsecomp-plus-qwen-8b",
        '  - doc_id (str): the numeric id of the scraped web page, e.g. "4217"; every chunk of a page shares it\n'
        '  - chunk_id (str): "<doc_id>_<element_id>", the id of this chunk (passage)\n'
        "  - url (str): the source URL of the web page (exact-match / $in filtering only)\n"
        "  - element_id (int): 0-based index of the chunk (a ~1024-token passage) within its web page\n",
    )],
    "financebench": [(
        "benchmarks/financebench/chromadb",
        "financebench-qwen-8b",
        '  - doc_id (str): the page key "<doc_name>::p<page_num>" (zero-indexed page of an SEC filing), e.g. "3M_2015_10K::p59"; every chunk on a page shares it\n'
        '  - chunk_id (str): "<doc_id>::e<element_id>", the id of this chunk (page element)\n'
        '  - doc_name (str): the SEC filing the page belongs to, "<COMPANY>_<YEAR>_<FORM>" (e.g. "3M_2015_10K"; forms include 10K, 10Q, 8K, EARNINGS)\n'
        "  - page_num (int): zero-indexed page number within the filing\n"
        "  - element_id (int): 0-based index of the chunk within its page\n"
        '  - type (str): the element type, e.g. "text", "table", "figure"\n',
    )],
    "freshstack": [
        ("benchmarks/freshstack/laravel/chromadb", "freshstack-laravel-qwen-0.6b", FRESHSTACK_FIELDS),
        ("benchmarks/freshstack/langchain/chromadb", "freshstack-langchain-qwen-0.6b", FRESHSTACK_FIELDS),
    ],
    "qampari": [(
        "benchmarks/qampari/chromadb",
        "qwen-qampari-0.6b",
        '  - doc_id (str): the numeric Wikipedia page id of the article, e.g. "10000001"; every passage of an article shares it\n'
        '  - chunk_id (str): "<page_id>__<n>", the id of this chunk (the n-th ~100-word passage of the article)\n'
        "  - page_id (str): the article's numeric Wikipedia page id, identical to doc_id\n"
        "  - title (str): the title of the Wikipedia article\n"
        '  - url (str): the article URL, "https://en.wikipedia.org/wiki?curid=<page_id>" (exact-match / $in filtering only)\n'
        "  - element_id (int): 0-based index of the passage within its article\n",
    )],
    # one collection per embedding rank, each in its own chroma dir (see trec_biogen.yaml); needs a
    # machine that can hold the shards — run per shard, or all four, with --benchmark trec_biogen.
    "trec_biogen": [(
        f"benchmarks/trec-biogen/chromadb/r{i}",
        f"qwen-biogen-0.6b_r{i}",
        '  - doc_id (str): the PubMed id (PMID) of the abstract, e.g. "31234567"\n'
        '  - chunk_id (str): "<doc_id>_0", the id of this chunk (one chunk per abstract)\n'
        "  - element_id (int): always 0 (one chunk per abstract)\n",
    ) for i in range(4)],
}


def _stored_fields(chroma_dir: str, name: str) -> tuple[str | None, dict]:
    """(fields, full metadata) straight from chroma.sqlite3 — read-only, no HNSW load."""
    import sqlite3

    con = sqlite3.connect(f"file:{chroma_dir}/chroma.sqlite3?mode=ro", uri=True)
    try:
        row = con.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
        if row is None:
            raise SystemExit(f"collection {name!r} not found in {chroma_dir}")
        rows = con.execute(
            "SELECT key, str_value, int_value, float_value, bool_value FROM collection_metadata WHERE collection_id=?", (row[0],)
        ).fetchall()
    finally:
        con.close()
    meta = {k: next((v for v in (s, i, f, b) if v is not None), None) for k, s, i, f, b in rows}
    return meta.get("fields"), meta


def set_fields(chroma_dir: str, name: str, fields: str) -> None:
    """Write `fields`, preserving every other collection-metadata key (e.g. legacy hnsw:* settings)."""
    import chromadb

    client = chromadb.PersistentClient(path=chroma_dir)
    c = client.get_collection(name)
    c.modify(metadata={**(c.metadata or {}), "fields": fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=sorted(TARGETS), action="append", default=[])
    parser.add_argument("--all", action="store_true", help="every benchmark except trec_biogen (which needs the large instance)")
    parser.add_argument("--check", action="store_true", help="only compare the stored schema against this file; write nothing")
    args = parser.parse_args()
    benchmarks = args.benchmark or ([b for b in TARGETS if b != "trec_biogen"] if args.all else [])
    if not benchmarks:
        parser.error("pass --benchmark <name> (repeatable) or --all")

    bad = 0
    for bench in benchmarks:
        for chroma_dir, name, fields in TARGETS[bench]:
            if not args.check:
                set_fields(chroma_dir, name, fields)
            stored, meta = _stored_fields(chroma_dir, name)
            ok = stored == fields
            bad += not ok
            extra = sorted(k for k in meta if k != "fields")
            print(f"[{bench}] {name}: {'OK' if ok else 'MISMATCH'} ({len(fields.splitlines())} fields; other metadata keys: {extra})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
