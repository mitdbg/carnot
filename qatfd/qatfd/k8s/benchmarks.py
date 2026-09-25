"""What a pod needs from the data bucket for each benchmark: the chroma store prefix and the benchmark
files, expressed as the `data` entries the chart's fetch step understands (deploy_exp/helm/experiments/
files/fetch.sh): a plain "prefix/" or "prefix/file" (same path under /data/benchmarks), or a dict with
`src`, optional `dest` (local layout differs from S3), `include` / `exclude` (aws s3 sync globs).

Paths mirror qatfd/configs/benchmarks/<benchmark>.yaml, which resolve relative to QATFD_BENCHMARKS_DIR
(/data/benchmarks in the pod). PDFs (the `pdf_dir` figure tool) are pulled only on request: they are GBs
and the search-agent sweeps run with `benchmarks.pdf_dir=null`."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkData:
    collection: str
    store_prefix: str
    data: list = field(default_factory=list)
    # prefix of the corpus PDFs (the view_figure tool's pdf_dir); None = the benchmark has none
    pdfs: str | None = None


def _meta(prefix: str) -> dict:
    """Only the `metadata_rank*.json` files of an element-embeddings prefix (the loaders read those; the
    embedding shards next to them are GBs the pod never touches)."""
    return {"src": prefix, "include": "metadata_rank*.json"}


BENCHMARKS: dict[str, BenchmarkData] = {
    "officeqa": BenchmarkData(
        collection="officeqa-qwen-8b-v1",
        store_prefix="officeqa/chromadb",
        data=[
            "officeqa/officeqa_pro.csv",
            "officeqa/officeqa_splits.json",
            "officeqa/answer_schema.json",
            "officeqa/codex/",
            "officeqa/treasury_bulletins_cleaned/",   # clean_page_map.json + the page texts (the document map)
        ],
        pdfs="officeqa/treasury_bulletin_pdfs/",
    ),
    "browsecomp_plus": BenchmarkData(
        collection="browsecomp-plus-qwen-8b-v1",
        store_prefix="browsecomp-plus/chromadb",
        data=[
            "browsecomp-plus/browsecomp_plus_decrypted.jsonl",
            "browsecomp-plus/browsecomp_plus_splits.json",
            "browsecomp-plus/answer_schema.json",
            "browsecomp-plus/codex/",
            _meta("browsecomp-plus/browsecomp-plus-element-embeddings/"),
        ],
    ),
    "financebench": BenchmarkData(
        collection="financebench-qwen-8b-v1",
        store_prefix="financebench/chromadb",
        data=[
            "financebench/financebench_open_source.jsonl",
            "financebench/financebench_splits.json",
            "financebench/answer_schema.json",
            "financebench/codex/",
            _meta("financebench/financebench-element-embeddings/"),
        ],
        pdfs="financebench/pdfs/",
    ),
}

# freshstack is one benchmark per topic: the store, collection and data live under freshstack/<topic>/
FRESHSTACK_TOPICS = ("laravel", "langchain")


def freshstack(topic: str) -> BenchmarkData:
    if topic not in FRESHSTACK_TOPICS:
        raise ValueError(f"unknown freshstack topic {topic!r} (one of {FRESHSTACK_TOPICS})")
    return BenchmarkData(
        collection=f"freshstack-{topic}-qwen-0.6b-v1",
        store_prefix=f"freshstack/{topic}/chromadb",
        data=[
            "freshstack/freshstack_splits.json",
            "freshstack/answer_schema.json",
            "freshstack/codex/",
            f"freshstack/{topic}/queries.jsonl",
            f"freshstack/{topic}/corpus.jsonl",
        ],
    )


def benchmark_data(benchmark: str, topic: str | None = None, with_pdfs: bool = False) -> BenchmarkData:
    """The pull list for `benchmark`; `with_pdfs` adds the corpus PDFs (only when the run keeps `pdf_dir`)."""
    if benchmark == "freshstack":
        bd = freshstack(topic or FRESHSTACK_TOPICS[0])
    elif benchmark in BENCHMARKS:
        bd = BENCHMARKS[benchmark]
    else:
        raise ValueError(f"no pod data recipe for benchmark {benchmark!r}; add one to qatfd/qatfd/k8s/benchmarks.py")
    if with_pdfs and bd.pdfs:
        return BenchmarkData(bd.collection, bd.store_prefix, [*bd.data, bd.pdfs], bd.pdfs)
    return bd
