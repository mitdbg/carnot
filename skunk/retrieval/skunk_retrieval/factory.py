from typing import Optional

from .config import PipelineConfig
from .dataset import discover_layout
from .lateon import LateOnIndex
from .page_table import PageTableLateOnRetriever, PageTableRetriever, load_query_expansions
from .pipeline import StrongRetriever

DEFAULT_LATEON_INDEX_NAME = "officeqa_lateon"


def build_retriever(config: PipelineConfig, args=None):
    layout = discover_layout(config.data_dir, canonical_source=config.canonical_source)
    expansions = layout.load_query_expansions() or load_query_expansions(config.data_dir)

    llm_model = config.llm_model
    if args is not None and getattr(args, "openrouter_model", None):
        llm_model = args.openrouter_model

    page_table = PageTableRetriever(
        str(config.index_path()),
        page_k=_get(args, "page_k", config.page_k),
        table_k=_get(args, "table_k", config.table_k),
        row_k=_get(args, "row_k", config.row_k),
        file_k=_get(args, "file_k", config.file_k),
        llm_config=config if config.use_llm_planner else None,
        openrouter_model=llm_model if config.llm_provider == "openrouter" and config.use_llm_planner else None,
        openrouter_timeout=config.llm_timeout,
        query_expansions=expansions,
    )

    base = page_table
    if config.use_lateon and _get(args, "lateon_folder", None) is not None:
        lateon_folder = str(_get(args, "lateon_folder"))
        records_file = "{}/{}_records.jsonl".format(lateon_folder, DEFAULT_LATEON_INDEX_NAME)
        lateon = LateOnIndex(
            index_folder=lateon_folder,
            index_name=DEFAULT_LATEON_INDEX_NAME,
            records_file=records_file,
            search_batch_size=_get(args, "lateon_search_batch_size", 2048),
            n_full_scores=_get(args, "lateon_n_full_scores", 1024),
            n_ivf_probe=_get(args, "lateon_n_ivf_probe", 8),
            device=_parse_lateon_device(_get(args, "lateon_device", None)),
        )
        base = PageTableLateOnRetriever(
            page_table=page_table,
            lateon=lateon,
            candidate_k=_get(args, "lateon_candidate_k", config.lateon_candidate_k),
            rerank_k=_get(args, "lateon_rerank_k", config.lateon_rerank_k),
            lateon_weight=config.lateon_weight,
            expand_page_records=_get(args, "lateon_expand_page_records", config.lateon_expand_page_records),
            include_row_records=_get(args, "lateon_include_row_records", config.lateon_include_row_records),
        )
    elif config.use_lateon:
        lateon_folder = str(config.lateon_path())
        records_file = "{}/{}_records.jsonl".format(lateon_folder, DEFAULT_LATEON_INDEX_NAME)
        lateon = LateOnIndex(
            index_folder=lateon_folder,
            index_name=DEFAULT_LATEON_INDEX_NAME,
            records_file=records_file,
            search_batch_size=_get(args, "lateon_search_batch_size", 2048) if args else 2048,
            n_full_scores=_get(args, "lateon_n_full_scores", 1024) if args else 1024,
            n_ivf_probe=_get(args, "lateon_n_ivf_probe", 8) if args else 8,
            device=_parse_lateon_device(_get(args, "lateon_device", None) if args else None),
        )
        base = PageTableLateOnRetriever(
            page_table=page_table,
            lateon=lateon,
            candidate_k=config.lateon_candidate_k,
            rerank_k=config.lateon_rerank_k,
            lateon_weight=config.lateon_weight,
            expand_page_records=config.lateon_expand_page_records,
            include_row_records=config.lateon_include_row_records,
        )

    if config.decompose_hard or config.retry_hard:
        return StrongRetriever(base, config)
    return base


def _get(args, name, default):
    if args is None:
        return default
    value = getattr(args, name, None)
    return default if value is None else value


def _parse_lateon_device(value):
    if not value:
        return None
    devices = [part.strip() for part in str(value).split(",") if part.strip()]
    if not devices:
        return None
    return devices[0] if len(devices) == 1 else devices
