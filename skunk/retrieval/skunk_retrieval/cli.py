import argparse
import json
import os
import sys
import time
from pathlib import Path

from .config import PipelineConfig, default_pipeline_config
from .corpus import DEFAULT_RECORD_SOURCES, build_records
from .dataset import discover_layout
from .display import make_display
from .factory import DEFAULT_LATEON_INDEX_NAME, build_retriever
from .format_selector import recommend_canonical_format, run_format_selection
from .lateon import LateOnIndex
from .officeqa import (
    OfficeQARow,
    evaluate_csv,
    evaluate_row,
    extract_oracle,
    load_officeqa_rows,
)
from .page_table import (
    add_retrieval_metrics,
    build_page_table_index,
    build_page_table_lateon_records,
)

DEFAULT_SEARCH_K = 25
DEFAULT_EVAL_K = 100
DEFAULT_LATEON_BATCH_SIZE = 32
DEFAULT_LATEON_SEARCH_BATCH_SIZE = 2048
DEFAULT_LATEON_N_FULL_SCORES = 1024
DEFAULT_LATEON_N_IVF_PROBE = 8
DEFAULT_PAGE_TABLE_LATEON_CANDIDATES = 500
DEFAULT_PAGE_TABLE_LATEON_RERANK_K = 1000


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="skunk_retrieval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Offline: format choice, FTS index, LateOn index.")
    preprocess.add_argument("--data-dir", required=True)
    preprocess.add_argument("--rich", action="store_true")
    preprocess.add_argument("--skip-format-choice", action="store_true")
    preprocess.add_argument("--non-interactive", action="store_true", help="Accept LLM format recommendation.")
    _add_llm_args(preprocess)

    choose_format = subparsers.add_parser("choose-format", help="LLM format judges + interactive choice.")
    choose_format.add_argument("--data-dir", required=True)
    choose_format.add_argument("--rich", action="store_true")
    choose_format.add_argument("--non-interactive", action="store_true")
    _add_llm_args(choose_format)

    build_lateon = subparsers.add_parser("build-lateon")
    build_lateon.add_argument("--data-dir", required=True)
    build_lateon.add_argument("--index-folder", required=True)
    build_lateon.add_argument(
        "--record-source",
        choices=("corpus", "page-table"),
        default="corpus",
        help="Records to encode in the LateOn index.",
    )
    build_lateon.add_argument(
        "--include-row-records",
        action="store_true",
        help="Include page-table row-window records when --record-source=page-table.",
    )
    build_lateon.add_argument("--rich", action="store_true")
    _add_lateon_args(build_lateon)

    build_page_table = subparsers.add_parser("build-page-table")
    build_page_table.add_argument("--data-dir", required=True)
    build_page_table.add_argument("--index-file", required=True)
    build_page_table.add_argument("--rich", action="store_true")

    search_page_table = subparsers.add_parser("search-page-table")
    search_page_table.add_argument("--data-dir")
    search_page_table.add_argument("--index-file", required=True)
    search_page_table.add_argument("--query", required=True)
    search_page_table.add_argument("--k", type=int, default=DEFAULT_SEARCH_K)
    search_page_table.add_argument("--lateon-folder")
    search_page_table.add_argument("--rich", action="store_true")
    _add_lateon_args(search_page_table)
    _add_page_table_args(search_page_table)

    eval_page_table = subparsers.add_parser("eval-page-table")
    eval_page_table.add_argument("--data-dir", help="Corpus root (default: parent of --index-file or SKUNK_DATA_DIR).")
    eval_page_table.add_argument("--index-file", required=True)
    eval_page_table_source = eval_page_table.add_mutually_exclusive_group(required=True)
    eval_page_table_source.add_argument("--benchmark-csv")
    eval_page_table_source.add_argument("--hf-data-file")
    eval_page_table.add_argument("--k", type=int, default=DEFAULT_EVAL_K)
    eval_page_table.add_argument("--limit", type=int)
    eval_page_table.add_argument("--rows-jsonl")
    eval_page_table.add_argument("--no-resume", action="store_true", help="Ignore existing rows-jsonl and overwrite it.")
    eval_page_table.add_argument("--lateon-folder")
    eval_page_table.add_argument("--rich", action="store_true")
    _add_lateon_args(eval_page_table)
    _add_page_table_args(eval_page_table)

    oracle_cmd = subparsers.add_parser("oracle")
    oracle_cmd.add_argument("--source-docs", required=True)
    oracle_cmd.add_argument("--source-files", default="")
    oracle_cmd.add_argument("--rich", action="store_true")

    args = parser.parse_args(argv)
    start = time.perf_counter()
    display = make_display(args.rich)

    if args.command == "preprocess":
        payload = _run_preprocess(args, display, start)
        _print(payload)
        return 0

    if args.command == "choose-format":
        config = _config_from_args(args)
        if args.non_interactive:
            recommendation = recommend_canonical_format(args.data_dir, config)
            layout = discover_layout(args.data_dir)
            layout.save_canonical(recommendation.recommended, rationale="auto")
            chosen = recommendation.recommended
        else:
            chosen = run_format_selection(args.data_dir, config, interactive=True)
        _print(_timed({"data_dir": args.data_dir, "canonical_source": chosen}, start))
        return 0

    if args.command == "build-lateon":
        display.build_start("LateOn index", args.data_dir, _sources_label(args.record_source, args.include_row_records))
        records = _lateon_build_records(args)
        _require_records(records, args.data_dir)
        records_jsonl = _default_lateon_records_file(args.index_folder)
        LateOnIndex.build(
            records=records,
            index_folder=args.index_folder,
            index_name=DEFAULT_LATEON_INDEX_NAME,
            records_file=records_jsonl,
            batch_size=DEFAULT_LATEON_BATCH_SIZE,
            search_batch_size=args.lateon_search_batch_size,
            n_full_scores=args.lateon_n_full_scores,
            n_ivf_probe=args.lateon_n_ivf_probe,
            device=_parse_lateon_device(args.lateon_device),
        )
        payload = _timed(
            {
                "data_dir": args.data_dir,
                "records": len(records),
                "index_folder": args.index_folder,
                "index_name": DEFAULT_LATEON_INDEX_NAME,
                "records_jsonl": records_jsonl,
                "record_source": args.record_source,
                "include_row_records": args.include_row_records,
            },
            start,
        )
        display.build_done(payload)
        _print(payload)
        return 0

    if args.command == "build-page-table":
        layout = discover_layout(args.data_dir)
        display.build_start("Page/table SQLite FTS index", args.data_dir, layout.canonical_source)
        payload = build_page_table_index(args.data_dir, args.index_file, canonical_source=layout.canonical_source)
        display.build_done(payload)
        _print(payload)
        return 0

    if args.command == "search-page-table":
        payload = _run_page_table_search(args, display, start)
        _print(payload)
        return 0

    if args.command == "eval-page-table":
        payload = _run_page_table_eval(args, display, start)
        _print(payload)
        return 0

    if args.command == "oracle":
        row = OfficeQARow(
            uid="",
            question="",
            answer="",
            source_docs=args.source_docs,
            source_files=args.source_files,
            difficulty="",
            raw={},
        )
        oracle = extract_oracle(row)
        payload = _timed(
            {
                "source_files": oracle.source_files,
                "source_refs": [ref.__dict__ for ref in oracle.source_refs],
                "pages_by_source_file": oracle.pages_by_source_file,
            },
            start,
        )
        display.oracle_done(payload, time.perf_counter() - start)
        _print(payload)
        return 0

    return 1


def _print(value) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _add_lateon_args(parser) -> None:
    parser.add_argument(
        "--lateon-device",
        help="LateOn/Plaid device, e.g. cuda:1 or cuda:0,cuda:1. Defaults to PyLate auto-detect.",
    )
    parser.add_argument(
        "--lateon-search-batch-size",
        type=int,
        default=DEFAULT_LATEON_SEARCH_BATCH_SIZE,
        help="Internal FastPlaid search batch size.",
    )
    parser.add_argument(
        "--lateon-n-full-scores",
        type=int,
        default=DEFAULT_LATEON_N_FULL_SCORES,
        help="FastPlaid full-score candidate count.",
    )
    parser.add_argument(
        "--lateon-n-ivf-probe",
        type=int,
        default=DEFAULT_LATEON_N_IVF_PROBE,
        help="FastPlaid IVF probe count.",
    )


def _add_page_table_args(parser) -> None:
    parser.add_argument("--page-k", type=int, default=800, help="Page FTS candidates.")
    parser.add_argument("--table-k", type=int, default=800, help="Table FTS candidates.")
    parser.add_argument("--row-k", type=int, default=200, help="Row-window FTS candidates; 0 disables row search.")
    parser.add_argument("--file-k", type=int, default=80, help="File/TOC FTS candidates.")
    parser.add_argument(
        "--lateon-candidate-k",
        type=int,
        default=DEFAULT_PAGE_TABLE_LATEON_CANDIDATES,
        help="Page candidates passed to LateOn; expanded to page/table records only with --lateon-expand-page-records.",
    )
    parser.add_argument(
        "--lateon-rerank-k",
        type=int,
        default=DEFAULT_PAGE_TABLE_LATEON_RERANK_K,
        help="LateOn rerank depth.",
    )
    parser.add_argument(
        "--lateon-expand-page-records",
        action="store_true",
        help="Expand each candidate page to all page/table LateOn records before reranking (default on).",
    )
    parser.add_argument(
        "--no-lateon-expand-page-records",
        action="store_true",
        help="Disable LateOn page-record expansion.",
    )
    parser.add_argument(
        "--lateon-include-row-records",
        action="store_true",
        help="Pass row-window records to LateOn (default on).",
    )
    parser.add_argument(
        "--no-lateon-include-row-records",
        action="store_true",
        help="Disable LateOn row records.",
    )
    parser.add_argument("--openrouter-model", help="LLM model (OpenRouter id or Gemini model name).")
    parser.add_argument("--openrouter-timeout", type=float, default=25.0, help="LLM timeout in seconds.")
    parser.add_argument(
        "--llm-provider",
        choices=("openrouter", "gemini"),
        help="LLM backend for query planning (default: openrouter).",
    )
    parser.add_argument("--no-llm-planner", action="store_true")
    parser.add_argument("--no-decompose", action="store_true")
    parser.add_argument("--no-retry", action="store_true")
    parser.add_argument("--no-lateon", action="store_true")


def _parse_lateon_device(value):
    if not value:
        return None
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        return None
    return devices[0] if len(devices) == 1 else devices


def _lateon_settings(args) -> dict:
    return {
        "device": _parse_lateon_device(args.lateon_device),
        "search_batch_size": args.lateon_search_batch_size,
        "n_full_scores": args.lateon_n_full_scores,
        "n_ivf_probe": args.lateon_n_ivf_probe,
    }


def _timed(value, start: float):
    value["elapsed_seconds"] = round(time.perf_counter() - start, 6)
    return value


def _sources_label(record_source: str = "corpus", include_rows: bool = False) -> str:
    if record_source == "page-table":
        sources = ["page", "table"]
        if include_rows:
            sources.append("row")
        return "page-table:" + ",".join(sources)
    return ",".join(DEFAULT_RECORD_SOURCES)


def _lateon_build_records(args):
    if args.record_source == "page-table":
        return build_page_table_lateon_records(
            args.data_dir,
            include_rows=args.include_row_records,
        )
    return build_records(args.data_dir)


def _default_lateon_records_file(index_folder: str) -> str:
    return str(Path(index_folder).expanduser() / "{}_records.jsonl".format(DEFAULT_LATEON_INDEX_NAME))


def _page_table_retriever_from_args(args):
    config = _config_from_args(args)
    config.page_table_index = args.index_file
    if getattr(args, "lateon_folder", None):
        config.lateon_dir = args.lateon_folder
    return build_retriever(config, args)


def _resolve_data_dir(args) -> str:
    if getattr(args, "data_dir", None):
        return args.data_dir
    if os.environ.get("SKUNK_DATA_DIR"):
        return os.environ["SKUNK_DATA_DIR"]
    index_file = getattr(args, "index_file", None)
    if index_file:
        return str(Path(index_file).expanduser().parent)
    benchmark_csv = getattr(args, "benchmark_csv", None)
    if benchmark_csv:
        return str(Path(benchmark_csv).expanduser().parent)
    return "/tmp/officeqa"


def _config_from_args(args) -> PipelineConfig:
    data_dir = _resolve_data_dir(args)
    config = default_pipeline_config(data_dir)
    if getattr(args, "llm_provider", None):
        config.llm_provider = args.llm_provider
    if getattr(args, "openrouter_model", None):
        config.llm_model = args.openrouter_model
    if getattr(args, "openrouter_timeout", None):
        config.llm_timeout = args.openrouter_timeout
    if getattr(args, "no_llm_planner", False):
        config.use_llm_planner = False
    elif getattr(args, "openrouter_model", None):
        config.use_llm_planner = True
        config.llm_model = args.openrouter_model
    else:
        config.use_llm_planner = bool(os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    if getattr(args, "no_decompose", False):
        config.decompose_hard = False
    if getattr(args, "no_retry", False):
        config.retry_hard = False
    if getattr(args, "no_lateon", False):
        config.use_lateon = False
    if getattr(args, "row_k", None) is not None:
        config.row_k = args.row_k
    if getattr(args, "page_k", None) is not None:
        config.page_k = args.page_k
    if getattr(args, "table_k", None) is not None:
        config.table_k = args.table_k
    if getattr(args, "file_k", None) is not None:
        config.file_k = args.file_k
    if getattr(args, "lateon_candidate_k", None) is not None:
        config.lateon_candidate_k = args.lateon_candidate_k
    if getattr(args, "lateon_rerank_k", None) is not None:
        config.lateon_rerank_k = args.lateon_rerank_k
    if getattr(args, "no_lateon_expand_page_records", False):
        config.lateon_expand_page_records = False
    elif getattr(args, "lateon_expand_page_records", False):
        config.lateon_expand_page_records = True
    if getattr(args, "no_lateon_include_row_records", False):
        config.lateon_include_row_records = False
    elif getattr(args, "lateon_include_row_records", False):
        config.lateon_include_row_records = True
    if getattr(args, "k", None):
        config.eval_k = args.k
    return config


def _add_llm_args(parser) -> None:
    parser.add_argument("--llm-provider", choices=("openrouter", "gemini"), default="openrouter")
    parser.add_argument("--llm-model", default="google/gemini-2.5-flash")
    parser.add_argument("--format-judges", type=int, default=3)


def _run_preprocess(args, display, start: float) -> dict:
    config = _config_from_args(args)
    config.data_dir = args.data_dir
    if not args.skip_format_choice:
        if args.non_interactive:
            recommendation = recommend_canonical_format(args.data_dir, config)
            discover_layout(args.data_dir).save_canonical(recommendation.recommended)
            config.canonical_source = recommendation.recommended
        else:
            config.canonical_source = run_format_selection(args.data_dir, config, interactive=True)

    layout = discover_layout(args.data_dir, canonical_source=config.canonical_source)
    config.canonical_source = layout.canonical_source

    display.build_start("Page/table index", args.data_dir, layout.canonical_source)
    index_payload = build_page_table_index(
        args.data_dir,
        str(config.index_path()),
        canonical_source=layout.canonical_source,
    )

    lateon_payload = None
    if config.use_lateon:
        records = build_page_table_lateon_records(
            args.data_dir,
            include_rows=config.lateon_include_row_records,
            canonical_source=layout.canonical_source,
        )
        _require_records(records, args.data_dir)
        records_jsonl = str(config.lateon_path() / "{}_records.jsonl".format(DEFAULT_LATEON_INDEX_NAME))
        LateOnIndex.build(
            records=records,
            index_folder=str(config.lateon_path()),
            index_name=DEFAULT_LATEON_INDEX_NAME,
            records_file=records_jsonl,
            batch_size=DEFAULT_LATEON_BATCH_SIZE,
            device=_parse_lateon_device(getattr(args, "lateon_device", None)),
        )
        lateon_payload = {"records": len(records), "index_folder": str(config.lateon_path())}

    config.save()
    return _timed(
        {
            "data_dir": args.data_dir,
            "canonical_source": layout.canonical_source,
            "page_table_index": index_payload,
            "lateon": lateon_payload,
            "config": str(config.meta_dir() / "pipeline_config.json"),
        },
        start,
    )


def _page_table_settings(args) -> dict:
    config = _config_from_args(args)
    settings = {
        "page_k": args.page_k,
        "table_k": args.table_k,
        "row_k": args.row_k,
        "file_k": args.file_k,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
        "use_llm_planner": config.use_llm_planner,
        "decompose_hard": config.decompose_hard,
        "retry_hard": config.retry_hard,
        "openrouter_model": args.openrouter_model,
    }
    if getattr(args, "lateon_folder", None):
        settings["lateon_folder"] = args.lateon_folder
        settings["lateon_candidate_k"] = args.lateon_candidate_k
        settings["lateon_rerank_k"] = args.lateon_rerank_k
        settings["lateon_expand_page_records"] = args.lateon_expand_page_records
        settings["lateon_include_row_records"] = args.lateon_include_row_records
        settings["lateon"] = _lateon_settings(args)
    return settings


def _run_page_table_search(args, display, start: float) -> dict:
    display.search_start(args.query, args.k)
    load_start = time.perf_counter()
    retriever = _page_table_retriever_from_args(args)
    load_seconds = time.perf_counter() - load_start
    plan = retriever.plan(args.query)
    retrieval_start = time.perf_counter()
    results = retriever.search(args.query, k=args.k)
    retrieval_seconds = time.perf_counter() - retrieval_start
    result_dicts = [result.to_dict() for result in results]
    display.search_done(result_dicts, retrieval_seconds)
    retriever.close()
    return _timed(
        {
            "query": args.query,
            "k": args.k,
            "years_found": plan.years,
            "months_found": plan.months,
            "terms": plan.terms,
            "phrases": plan.phrases,
            "index_load_seconds": round(load_seconds, 6),
            "retrieval_seconds": round(retrieval_seconds, 6),
            "result_count": len(result_dicts),
            "settings": _page_table_settings(args),
            "results": result_dicts,
        },
        start,
    )


def _run_page_table_eval(args, display, start: float) -> dict:
    rows = _load_eval_input_rows(args)
    if args.limit is not None:
        rows = rows[: args.limit]

    existing_by_uid = {}
    rows_path = Path(args.rows_jsonl) if args.rows_jsonl else None
    if rows_path:
        _probe_output_dir(rows_path.parent)
    if rows_path and not args.no_resume:
        existing_by_uid = _read_jsonl_rows_by_uid(rows_path)
    existing_count = sum(1 for row in rows if row.uid in existing_by_uid)

    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if row.uid not in existing_by_uid
    ]

    load_seconds = 0.0
    if pending:
        load_start = time.perf_counter()
        retriever = _page_table_retriever_from_args(args)
        load_seconds = time.perf_counter() - load_start
        try:
            writer = _StreamingJsonlWriter(rows_path, append=not args.no_resume) if rows_path else None
            try:
                with display.eval(len(pending)) as progress:
                    for index, row in pending:
                        row_start = time.perf_counter()
                        eval_row = evaluate_row(retriever, row, k=args.k)
                        eval_row.elapsed_seconds = round(time.perf_counter() - row_start, 6)
                        row_payload = eval_row.to_dict()
                        existing_by_uid[row.uid] = row_payload
                        if writer:
                            writer.write(row_payload)
                        progress.eval_row(index, row, eval_row, eval_row.elapsed_seconds)
            finally:
                if writer:
                    writer.close()
        finally:
            retriever.close()
    else:
        with display.eval(0):
            pass

    row_payloads = [
        existing_by_uid[row.uid]
        for row in rows
        if row.uid in existing_by_uid
    ]
    summary = add_retrieval_metrics(_summarize_eval_payloads(row_payloads))
    if not args.rows_jsonl:
        summary["rows"] = row_payloads
    else:
        summary.pop("rows", None)
    payload = _timed(summary, start)
    payload["k"] = args.k
    payload["settings"] = _page_table_settings(args)
    payload["index_load_seconds"] = round(load_seconds, 6)
    payload["rows_jsonl"] = str(rows_path) if rows_path else None
    payload["resume"] = {
        "enabled": bool(rows_path and not args.no_resume),
        "existing_rows": existing_count,
        "evaluated_rows": len(pending),
        "completed_rows": len(row_payloads),
        "target_rows": len(rows),
    }
    display.eval_done(payload, time.perf_counter() - start)
    return payload


def _write_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


class _StreamingJsonlWriter:
    def __init__(self, path: Path, append: bool = True):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, row: dict) -> None:
        self.handle.write(json.dumps(row, sort_keys=True))
        self.handle.write("\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def _load_eval_input_rows(args):
    if args.benchmark_csv:
        return load_officeqa_rows(csv_path=args.benchmark_csv)
    return load_officeqa_rows(data_file=args.hf_data_file)


def _read_jsonl_rows_by_uid(path: Path) -> dict:
    if not path.is_file():
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("uid")
            if uid:
                rows[str(uid)] = row
    return rows


def _probe_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".officeqa_write_test_{}".format(os.getpid())
    try:
        with open(probe, "w", encoding="utf-8") as handle:
            handle.write("ok\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            probe.unlink()
        except FileNotFoundError:
            pass


def _summarize_eval_payloads(rows: list) -> dict:
    total = len(rows)
    page_rows = [row for row in rows if row.get("page_hit") is not None]
    file_hits = sum(1 for row in rows if row.get("file_hit"))
    page_hits = sum(1 for row in page_rows if row.get("page_hit"))
    return {
        "total": total,
        "file_recall": file_hits / total if total else 0.0,
        "page_total": len(page_rows),
        "page_recall": page_hits / len(page_rows) if page_rows else None,
        "rows": rows,
    }


def _write_json(payload, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require_records(records, data_dir: str) -> None:
    if not records:
        raise SystemExit(
            "No corpus records found. Check --data-dir={} and expected OfficeQA corpus paths.".format(
                data_dir,
            )
        )
