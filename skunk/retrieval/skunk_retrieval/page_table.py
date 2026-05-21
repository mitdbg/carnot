import json
import math
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ._old.bm25 import tokenize
from .config import PipelineConfig
from .corpus import CorpusRecord, SearchResult, compact_whitespace, html_to_text, parse_year_month
from .dataset import discover_layout
from .llm import chat_json
from .officeqa import MONTHS


YEAR_RE = re.compile(r"\b(18\d{2}|19\d{2}|20\d{2})\b")
MONTH_RE = re.compile(r"\b({})\b".format("|".join(MONTHS.keys())), re.IGNORECASE)
SOURCE_FILE_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")

STOPWORDS = {
    "a", "about", "according", "actual", "all", "amount", "amounts", "an", "and",
    "answer", "are", "as", "at", "be", "between", "both", "by", "calculate",
    "calendar", "compute", "determine", "did", "do", "does", "dollar", "dollars",
    "during", "each", "federal", "fiscal", "for", "from", "had", "has", "have",
    "how", "in", "inclusive", "into", "is", "it", "just", "million", "millions",
    "nominal", "of", "on", "only", "or", "reported", "return", "rounded", "same",
    "specifically", "states", "sum", "the", "these", "this", "to", "total",
    "treasury", "treat", "u", "united", "us", "using", "value", "values", "was",
    "were", "what", "when", "which", "with", "year", "years",
}

_PROFILE_EXPANSIONS = Path(__file__).parent / "profiles" / "expansions.json"


def load_query_expansions(data_dir: Optional[str] = None) -> Dict[str, List[str]]:
    if data_dir:
        from .dataset import discover_layout

        loaded = discover_layout(data_dir).load_query_expansions()
        if loaded:
            return loaded
    if _PROFILE_EXPANSIONS.is_file():
        with open(_PROFILE_EXPANSIONS, encoding="utf-8") as handle:
            raw = json.load(handle)
        return {str(k): list(v) for k, v in raw.items() if isinstance(v, list)}
    return {}


GENERIC_HEADING_RE = re.compile(
    r"^(?:"
    r"page|treasury bulletin|treasury department|office of the secretary|"
    r"\(?\[?in (?:millions|billions|thousands)|"
    r"\[?source:|source:|note:|see footnotes|footnotes follow|"
    r"\(?percent per annum|payable in u\.?\s*s\.?\s*dollars|"
    r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}"
    r")",
    re.IGNORECASE,
)


@dataclass
class PageTableBuildSummary:
    data_dir: str
    index_file: str
    files: int = 0
    pages: int = 0
    tables: int = 0
    row_windows: int = 0
    records: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "data_dir": self.data_dir,
            "index_file": self.index_file,
            "files": self.files,
            "pages": self.pages,
            "tables": self.tables,
            "row_windows": self.row_windows,
            "records": self.records,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


@dataclass
class PageTableQueryPlan:
    question: str
    years: List[int]
    months: List[int]
    issue_dates: List[Tuple[int, int]]
    page_numbers: List[int]
    pdf_pages: List[int]
    report_pages: List[int]
    terms: List[str]
    phrases: List[str]
    match_expr: str


@dataclass
class _PageState:
    source_file: str
    page_id: int
    score: float = 0.0
    evidence_score: float = -1.0
    evidence: Optional[CorpusRecord] = None
    breakdown: Dict[str, float] = field(default_factory=dict)
    channels: Dict[str, int] = field(default_factory=dict)


class PageTableRetriever:
    def __init__(
        self,
        index_file: str,
        page_k: int = 800,
        table_k: int = 800,
        row_k: int = 200,
        file_k: int = 80,
        rrf_k: int = 60,
        openrouter_model: Optional[str] = None,
        openrouter_timeout: float = 20.0,
        llm_config: Optional[PipelineConfig] = None,
        query_expansions: Optional[Dict[str, List[str]]] = None,
    ):
        self.index_file = index_file
        self.page_k = page_k
        self.table_k = table_k
        self.row_k = row_k
        self.file_k = file_k
        self.rrf_k = rrf_k
        self.llm_config = llm_config
        self.query_expansions = query_expansions or {}
        self.openrouter_model = openrouter_model
        self.openrouter_timeout = openrouter_timeout
        self._plan_cache: Dict[str, PageTableQueryPlan] = {}
        self._lateon_records_cache: Dict[Tuple[str, int, bool], List[str]] = {}
        self.con = sqlite3.connect(index_file)
        self.con.row_factory = sqlite3.Row

    def plan(self, question: str) -> PageTableQueryPlan:
        if question in self._plan_cache:
            return self._plan_cache[question]
        plan = build_page_table_query_plan(question, query_expansions=self.query_expansions)
        use_llm = self.llm_config and self.llm_config.use_llm_planner
        if use_llm:
            plan = augment_plan_with_llm(plan, self.llm_config)
        elif self.openrouter_model:
            plan = augment_plan_with_llm(
                plan,
                PipelineConfig(
                    llm_provider="openrouter",
                    llm_model=self.openrouter_model,
                    llm_timeout=self.openrouter_timeout,
                    use_llm_planner=True,
                ),
            )
        self._plan_cache[question] = plan
        return plan

    def search(self, question: str, k: int = 50) -> List[SearchResult]:
        plan = self.plan(question)
        if not plan.match_expr:
            return []

        states: Dict[Tuple[str, int], _PageState] = {}
        file_scores = defaultdict(float)

        self._add_record_channel(states, plan, "page_fts", "page", self.page_k, 1.00)
        self._add_record_channel(states, plan, "table_fts", "table", self.table_k, 1.35)
        if self.row_k > 0:
            self._add_record_channel(states, plan, "row_fts", "row", self.row_k, 1.20)

        for rank, record, _score in self._fts_search("file_fts", plan.match_expr, self.file_k):
            file_scores[record.source_file] += 0.75 / (self.rrf_k + rank)

        self._add_direct_page_candidates(states, plan)

        for state in states.values():
            if state.source_file in file_scores:
                self._add_score(state, "file", file_scores[state.source_file])
            date_boost = _date_boost(plan, state.source_file)
            if date_boost:
                self._add_score(state, "date", date_boost)
            penalty = _page_penalty(plan, state.evidence)
            if penalty:
                self._add_score(state, "penalty", -penalty)

        ranked = sorted(states.values(), key=lambda state: state.score, reverse=True)[:k]
        results = []
        for rank, state in enumerate(ranked, start=1):
            evidence = state.evidence
            if evidence is None:
                continue
            title = evidence.title or _page_title(evidence.text)
            record = CorpusRecord(
                record_id="{}:page:{}:page_table".format(state.source_file, state.page_id),
                record_type="page",
                text=evidence.text,
                source_file=state.source_file,
                page_id=state.page_id,
                year=evidence.year,
                month=evidence.month,
                title=title,
                section_path=evidence.section_path,
                metadata={
                    "evidence_record_id": evidence.record_id,
                    "evidence_record_type": evidence.record_type,
                    "channels": state.channels,
                },
            )
            results.append(
                SearchResult(
                    record=record,
                    score=state.score,
                    rank=rank,
                    channel="page_table",
                    score_breakdown=dict(sorted(state.breakdown.items())),
                )
            )
        return results

    def close(self) -> None:
        self.con.close()

    def _add_record_channel(
        self,
        states: Dict[Tuple[str, int], _PageState],
        plan: PageTableQueryPlan,
        table: str,
        channel: str,
        limit: int,
        weight: float,
    ) -> None:
        for rank, record, score in self._fts_search(table, plan.match_expr, limit):
            if record.page_id is None:
                continue
            key = (record.source_file, record.page_id)
            state = states.setdefault(key, _PageState(source_file=record.source_file, page_id=record.page_id))
            contribution = weight / (self.rrf_k + rank)
            self._add_score(state, channel, contribution)
            state.channels[channel] = min(rank, state.channels.get(channel, rank))
            evidence_score = contribution + min(score, 50.0) * 0.0001
            if evidence_score > state.evidence_score:
                state.evidence = record
                state.evidence_score = evidence_score

    def _fts_search(self, table: str, match_expr: str, limit: int) -> List[Tuple[int, CorpusRecord, float]]:
        weights = {
            "page_fts": "0.35, 4.5, 1.0",
            "table_fts": "0.35, 6.0, 1.25",
            "row_fts": "0.35, 5.5, 1.4",
            "file_fts": "0.8, 4.0, 1.0",
        }[table]
        sql = """
            SELECT records.*, -bm25({table}, {weights}) AS lexical_score
            FROM {table}
            JOIN records ON records.rowid = {table}.rowid
            WHERE {table} MATCH ?
            ORDER BY bm25({table}, {weights})
            LIMIT ?
        """.format(table=table, weights=weights)
        try:
            rows = self.con.execute(sql, (match_expr, limit)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [
            (rank, _record_from_row(row), float(row["lexical_score"] or 0.0))
            for rank, row in enumerate(rows, start=1)
        ]

    def _add_score(self, state: _PageState, channel: str, contribution: float) -> None:
        state.score += contribution
        state.breakdown[channel] = state.breakdown.get(channel, 0.0) + contribution

    def _add_direct_page_candidates(
        self,
        states: Dict[Tuple[str, int], _PageState],
        plan: PageTableQueryPlan,
    ) -> None:
        if not plan.issue_dates or not (plan.page_numbers or plan.pdf_pages or plan.report_pages):
            return
        for year, month in plan.issue_dates:
            source_file = "treasury_bulletin_{:04d}_{:02d}.txt".format(year, month)
            candidates: Dict[int, float] = {}
            for page in plan.pdf_pages:
                candidates[page] = max(candidates.get(page, 0.0), 0.090)
            for page in plan.report_pages:
                candidates[page] = max(candidates.get(page, 0.0), 0.040)
                for offset in range(12, 21):
                    candidates[page + offset] = max(candidates.get(page + offset, 0.0), 0.070)
            for page in plan.page_numbers:
                candidates[page] = max(candidates.get(page, 0.0), 0.040)
                for offset in range(12, 21):
                    candidates[page + offset] = max(candidates.get(page + offset, 0.0), 0.055)
            for page_id, contribution in candidates.items():
                record = self._page_record(source_file, page_id)
                if record is None:
                    continue
                key = (record.source_file, record.page_id)
                state = states.setdefault(key, _PageState(source_file=record.source_file, page_id=record.page_id))
                self._add_score(state, "direct_page", contribution)
                state.channels["direct_page"] = 1
                if contribution > state.evidence_score:
                    state.evidence = record
                    state.evidence_score = contribution

    def _page_record(self, source_file: str, page_id: int) -> Optional[CorpusRecord]:
        row = self.con.execute(
            """
            SELECT * FROM records
            WHERE record_type = 'page' AND source_file = ? AND page_id = ?
            LIMIT 1
            """,
            (source_file, page_id),
        ).fetchone()
        return _record_from_row(row) if row else None

    def lateon_record_ids_for_page(
        self,
        source_file: str,
        page_id: int,
        include_rows: bool = False,
    ) -> List[str]:
        key = (source_file, page_id, include_rows)
        cached = self._lateon_records_cache.get(key)
        if cached is not None:
            return cached

        record_types = ("page", "table", "row") if include_rows else ("page", "table")
        placeholders = ",".join("?" for _ in record_types)
        sql = """
            SELECT record_id
            FROM records
            WHERE source_file = ? AND page_id = ? AND record_type IN ({})
            ORDER BY
                CASE record_type
                    WHEN 'page' THEN 0
                    WHEN 'table' THEN 1
                    ELSE 2
                END,
                COALESCE(table_index, 0),
                rowid
        """.format(placeholders)
        rows = self.con.execute(sql, (source_file, page_id, *record_types)).fetchall()
        ids = [str(row["record_id"]) for row in rows]
        self._lateon_records_cache[key] = ids
        return ids


class PageTableLateOnRetriever:
    def __init__(
        self,
        page_table: PageTableRetriever,
        lateon,
        candidate_k: int = 800,
        rerank_k: int = 1000,
        lateon_weight: float = 2.0,
        rrf_k: int = 60,
        expand_page_records: bool = False,
        include_row_records: bool = False,
    ):
        self.page_table = page_table
        self.lateon = lateon
        self.candidate_k = candidate_k
        self.rerank_k = rerank_k
        self.lateon_weight = lateon_weight
        self.rrf_k = rrf_k
        self.expand_page_records = expand_page_records
        self.include_row_records = include_row_records

    def plan(self, question: str) -> PageTableQueryPlan:
        return self.page_table.plan(question)

    def search(self, question: str, k: int = 50) -> List[SearchResult]:
        base = self.page_table.search(question, k=max(k, self.candidate_k))
        states: Dict[Tuple[str, int], SearchResult] = {}
        scores: Dict[Tuple[str, int], float] = defaultdict(float)

        for rank, result in enumerate(base, start=1):
            key = (result.record.source_file, result.record.page_id)
            if key[1] is None:
                continue
            states[key] = result
            scores[key] += result.score
            scores[key] += 1.0 / (self.rrf_k + rank)

        subset = _lateon_subset_ids(
            base,
            lateon_record_ids_for_page=(
                self.page_table.lateon_record_ids_for_page if self.expand_page_records else None
            ),
            include_row_records=self.include_row_records,
        )
        if subset:
            lateon_results = self.lateon.search(
                question,
                k=min(self.rerank_k, len(subset)),
                subset=subset,
            )
            if self.expand_page_records:
                best_by_page: Dict[Tuple[str, int], Tuple[int, SearchResult]] = {}
                for rank, result in enumerate(lateon_results, start=1):
                    key = (result.record.source_file, result.record.page_id)
                    if key[1] is None:
                        continue
                    if key not in best_by_page:
                        best_by_page[key] = (rank, result)
                for key, (rank, result) in best_by_page.items():
                    if key not in states:
                        states[key] = result
                    contribution = self.lateon_weight / (self.rrf_k + rank)
                    scores[key] += contribution
                    states[key].score_breakdown["lateon_rerank"] = contribution
                    states[key].score_breakdown["lateon_best_rank"] = float(rank)
                    states[key].score_breakdown["lateon_best_score"] = result.score
            else:
                for rank, result in enumerate(lateon_results, start=1):
                    key = (result.record.source_file, result.record.page_id)
                    if key[1] is None:
                        continue
                    if key not in states:
                        states[key] = result
                    contribution = self.lateon_weight / (self.rrf_k + rank)
                    scores[key] += contribution
                    states[key].score_breakdown["lateon_rerank"] = (
                        states[key].score_breakdown.get("lateon_rerank", 0.0)
                        + contribution
                    )

        ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)[:k]
        out = []
        for rank, key in enumerate(ranked_keys, start=1):
            result = states[key]
            result.rank = rank
            result.score = scores[key]
            result.channel = "page_table_lateon"
            out.append(result)
        return out

    def close(self) -> None:
        self.page_table.close()


def build_page_table_index(data_dir: str, index_file: str, canonical_source: Optional[str] = None) -> Dict:
    start = time.perf_counter()
    layout = discover_layout(data_dir, canonical_source=canonical_source)
    source = layout.canonical_source
    if source != "json":
        raise NotImplementedError(
            "Page/table index build currently requires canonical_source=json; got {}. "
            "Re-run choose-format or set SKUNK_CANONICAL_SOURCE=json.".format(source)
        )
    json_dir = layout.paths.json_dir
    if not json_dir or not json_dir.is_dir():
        raise FileNotFoundError("Missing JSON corpus directory under {}".format(data_dir))
    data_path = layout.data_dir

    index_path = Path(index_file).expanduser()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    con = sqlite3.connect(str(tmp_path))
    try:
        _create_schema(con)
        summary = PageTableBuildSummary(data_dir=str(data_path), index_file=str(index_path))
        rowid = 0
        for file_index, json_file in enumerate(sorted(json_dir.glob("treasury_bulletin_*.json")), start=1):
            file_records = list(iter_page_table_records(json_file))
            summary.files += 1
            for record in file_records:
                rowid += 1
                _insert_record(con, rowid, record)
                summary.records += 1
                if record.record_type == "page":
                    summary.pages += 1
                elif record.record_type == "table":
                    summary.tables += 1
                elif record.record_type == "row":
                    summary.row_windows += 1
            if file_index % 25 == 0:
                con.commit()
        con.commit()
        _optimize_fts(con)
        con.commit()
    finally:
        con.close()

    tmp_path.replace(index_path)
    summary.elapsed_seconds = time.perf_counter() - start
    return summary.to_dict()


def build_page_table_lateon_records(
    data_dir: str,
    include_rows: bool = False,
    canonical_source: Optional[str] = None,
) -> List[CorpusRecord]:
    layout = discover_layout(data_dir, canonical_source=canonical_source)
    json_dir = layout.paths.json_dir
    if not json_dir or not json_dir.is_dir():
        raise FileNotFoundError("Missing JSON corpus directory under {}".format(data_dir))

    record_types = {"page", "table"}
    if include_rows:
        record_types.add("row")

    records = []
    for json_file in sorted(json_dir.glob("treasury_bulletin_*.json")):
        for record in iter_page_table_records(json_file, include_rows=include_rows):
            if record.record_type in record_types:
                records.append(record)
    return records


def iter_page_table_records(json_file: Path, include_rows: bool = True) -> Iterator[CorpusRecord]:
    source_file = json_file.with_suffix(".txt").name
    year, month = parse_year_month(source_file)
    obj = json.loads(json_file.read_text(encoding="utf-8"))
    elements = _extract_json_elements(obj)
    issue_title = _issue_title(year, month)

    headings_for_file = []
    pages: Dict[int, List[_Element]] = defaultdict(list)
    for element in elements:
        pages[element.page_id].append(element)
        if element.kind in {"title", "section_header", "caption"} and _meaningful_heading(element.text):
            headings_for_file.append(element.text)

    file_text = "\n".join(_dedupe_keep_order(headings_for_file)[:120])
    yield CorpusRecord(
        record_id="{}:file".format(source_file),
        record_type="file",
        text="\n".join(part for part in [issue_title, file_text] if part),
        source_file=source_file,
        page_id=None,
        year=year,
        month=month,
        title=issue_title,
    )

    table_index = 0
    for page_id in sorted(pages):
        page_elements = pages[page_id]
        page_context = _page_context(page_elements)
        page_text = _page_record_text(issue_title, source_file, page_id, page_context, page_elements)
        yield CorpusRecord(
            record_id="{}:page:{}".format(source_file, page_id),
            record_type="page",
            text=page_text,
            source_file=source_file,
            page_id=page_id,
            year=year,
            month=month,
            title=page_context[-1] if page_context else issue_title,
            section_path=page_context,
        )

        for index, element in enumerate(page_elements):
            if not element.is_table:
                continue
            table_index += 1
            context = _table_context(page_elements, index)
            title = context[-1] if context else "Table on page {}".format(page_id)
            table_text = _table_record_text(issue_title, source_file, page_id, context, element.text)
            table_id = "{}:page:{}:table:{}".format(source_file, page_id, table_index)
            yield CorpusRecord(
                record_id=table_id,
                record_type="table",
                text=table_text,
                source_file=source_file,
                page_id=page_id,
                year=year,
                month=month,
                title=title,
                section_path=context,
                table_index=table_index,
                metadata={"element_index": element.index},
            )
            if include_rows:
                for row_record in _iter_row_records(
                    source_file=source_file,
                    year=year,
                    month=month,
                    page_id=page_id,
                    table_index=table_index,
                    table_id=table_id,
                    title=title,
                    context=context,
                    html=element.raw,
                ):
                    yield row_record


def build_page_table_query_plan(
    question: str,
    query_expansions: Optional[Dict[str, List[str]]] = None,
) -> PageTableQueryPlan:
    normalized = _normalize_query(question)
    years = sorted({int(value) for value in YEAR_RE.findall(normalized)})
    months = sorted({MONTHS[m.group(1).lower()] for m in MONTH_RE.finditer(normalized)})
    issue_dates = _issue_dates(normalized)
    page_numbers = _page_numbers(normalized, r"\bpage\s+(\d{1,3})\b")
    pdf_pages = _page_numbers(normalized, r"\bpdf\s+page\s+(\d{1,3})\b")
    report_pages = _page_numbers(normalized, r"\breport\s+page\s+(\d{1,3})\b")
    phrases = _query_phrases(normalized, query_expansions or {})
    terms = _query_terms(normalized, phrases)
    match_expr = _match_expr(phrases, terms)
    return PageTableQueryPlan(
        question=question,
        years=years,
        months=months,
        issue_dates=issue_dates,
        page_numbers=page_numbers,
        pdf_pages=pdf_pages,
        report_pages=report_pages,
        terms=terms,
        phrases=phrases,
        match_expr=match_expr,
    )


def augment_plan_with_llm(plan: PageTableQueryPlan, config: PipelineConfig) -> PageTableQueryPlan:
    domain = config.planner_domain_hint or config.dataset_domain
    hints = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "You are a retrieval query planner for {}. "
                    "Return compact JSON only. Do not answer the question. Do not invent source files."
                ).format(domain),
            },
            {
                "role": "user",
                "content": (
                    "Extract retrieval hints. Return JSON with keys: "
                    "phrases, terms, issue_dates (objects with year and month 1-12), "
                    "page_numbers, pdf_pages, report_pages. Question: "
                    + json.dumps(plan.question)
                ),
            },
        ],
        provider=config.llm_provider,
        model=config.llm_model,
        timeout=config.llm_timeout,
        max_tokens=600,
    )
    return PageTableQueryPlan(
        question=plan.question,
        years=plan.years,
        months=plan.months,
        issue_dates=_merge_issue_dates(plan.issue_dates, hints.get("issue_dates")),
        page_numbers=_merge_ints(plan.page_numbers, hints.get("page_numbers")),
        pdf_pages=_merge_ints(plan.pdf_pages, hints.get("pdf_pages")),
        report_pages=_merge_ints(plan.report_pages, hints.get("report_pages")),
        terms=_dedupe_keep_order([*plan.terms, *[str(term) for term in hints.get("terms", [])]])[:48],
        phrases=_dedupe_keep_order([*plan.phrases, *[str(phrase) for phrase in hints.get("phrases", [])]])[:18],
        match_expr=_match_expr(
            _dedupe_keep_order([*plan.phrases, *[str(phrase) for phrase in hints.get("phrases", [])]])[:18],
            _dedupe_keep_order([*plan.terms, *[str(term) for term in hints.get("terms", [])]])[:48],
        ),
    )


def add_retrieval_metrics(summary: Dict) -> Dict:
    rows = summary.get("rows") or []
    file_total = 0
    file_hits = 0
    page_total = 0
    page_hits = 0
    all_files_hit = 0
    all_pages_hit = 0
    predicted_page_total = 0
    predicted_file_total = 0
    reciprocal_file = 0.0
    reciprocal_page = 0.0

    for row in rows:
        gold_files = set(row.get("gold_files") or [])
        predicted_files = set(row.get("predicted_files") or [])
        gold_pages = {
            (source_file, page)
            for source_file, pages in (row.get("gold_pages") or {}).items()
            for page in pages
        }
        predicted_pages = {
            (source_file, page)
            for source_file, pages in (row.get("predicted_pages") or {}).items()
            for page in pages
        }
        file_total += len(gold_files)
        file_hits += len(gold_files & predicted_files)
        page_total += len(gold_pages)
        page_hits += len(gold_pages & predicted_pages)
        predicted_file_total += len(predicted_files)
        predicted_page_total += len(predicted_pages)
        if gold_files and gold_files <= predicted_files:
            all_files_hit += 1
        if gold_pages and gold_pages <= predicted_pages:
            all_pages_hit += 1
        if row.get("best_file_rank"):
            reciprocal_file += 1.0 / row["best_file_rank"]
        if row.get("best_page_rank"):
            reciprocal_page += 1.0 / row["best_page_rank"]

    total = len(rows)
    file_rows = [row for row in rows if row.get("gold_files")]
    page_rows = [row for row in rows if row.get("gold_pages")]
    summary["file_instance_recall"] = file_hits / file_total if file_total else 0.0
    summary["page_instance_recall"] = page_hits / page_total if page_total else 0.0
    summary["all_gold_files_recall"] = all_files_hit / len(file_rows) if file_rows else None
    summary["all_gold_pages_recall"] = all_pages_hit / len(page_rows) if page_rows else None
    summary["file_precision"] = file_hits / predicted_file_total if predicted_file_total else 0.0
    summary["page_precision"] = page_hits / predicted_page_total if predicted_page_total else 0.0
    summary["file_mrr"] = reciprocal_file / total if total else 0.0
    summary["page_mrr"] = reciprocal_page / len(page_rows) if page_rows else 0.0
    summary["gold_file_instances"] = file_total
    summary["gold_page_instances"] = page_total
    summary["hit_file_instances"] = file_hits
    summary["hit_page_instances"] = page_hits
    return summary


@dataclass
class _Element:
    index: int
    page_id: int
    kind: str
    text: str
    raw: str
    is_table: bool


class _TableHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: List[List[str]] = []
        self._row: Optional[List[str]] = None
        self._cell_parts: Optional[List[str]] = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            self._cell_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append(" ")

    def handle_data(self, data):
        if self._cell_parts is not None and data:
            self._cell_parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_parts is not None:
            if self._row is not None:
                self._row.append(compact_whitespace(" ".join(self._cell_parts)))
            self._cell_parts = None
        elif tag == "tr" and self._row is not None:
            row = [cell for cell in self._row if cell]
            if row:
                self.rows.append(row)
            self._row = None


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        PRAGMA cache_size = -200000;

        CREATE TABLE records (
            rowid INTEGER PRIMARY KEY,
            record_id TEXT NOT NULL UNIQUE,
            record_type TEXT NOT NULL,
            source_file TEXT NOT NULL,
            page_id INTEGER,
            year INTEGER,
            month INTEGER,
            title TEXT,
            text TEXT NOT NULL,
            section_path TEXT NOT NULL,
            table_index INTEGER,
            metadata TEXT NOT NULL
        );
        CREATE INDEX records_source_page_idx ON records(source_file, page_id);
        CREATE INDEX records_type_idx ON records(record_type);

        CREATE VIRTUAL TABLE page_fts USING fts5(
            source_file,
            title,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE table_fts USING fts5(
            source_file,
            title,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE row_fts USING fts5(
            source_file,
            title,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        CREATE VIRTUAL TABLE file_fts USING fts5(
            source_file,
            title,
            text,
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )


def _insert_record(con: sqlite3.Connection, rowid: int, record: CorpusRecord) -> None:
    con.execute(
        """
        INSERT INTO records (
            rowid, record_id, record_type, source_file, page_id, year, month, title, text,
            section_path, table_index, metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rowid,
            record.record_id,
            record.record_type,
            record.source_file,
            record.page_id,
            record.year,
            record.month,
            record.title,
            record.text,
            json.dumps(record.section_path),
            record.table_index,
            json.dumps(record.metadata, sort_keys=True),
        ),
    )
    table = {
        "file": "file_fts",
        "page": "page_fts",
        "table": "table_fts",
        "row": "row_fts",
    }.get(record.record_type)
    if table:
        con.execute(
            "INSERT INTO {}(rowid, source_file, title, text) VALUES (?, ?, ?, ?)".format(table),
            (rowid, record.source_file, record.title or "", record.text),
        )


def _optimize_fts(con: sqlite3.Connection) -> None:
    for table in ("page_fts", "table_fts", "row_fts", "file_fts"):
        con.execute("INSERT INTO {}({}) VALUES ('optimize')".format(table, table))


def _extract_json_elements(obj: Dict) -> List[_Element]:
    out = []
    for index, element in enumerate(((obj.get("document") or {}).get("elements") or [])):
        content = element.get("content") if isinstance(element, dict) else None
        if not isinstance(content, str) or not content.strip():
            continue
        page_id = _get_page_id(element)
        if page_id is None:
            continue
        is_table = "<table" in content.lower()
        text = html_to_text(content) if is_table else compact_whitespace(content)
        if not text:
            continue
        out.append(
            _Element(
                index=index,
                page_id=page_id,
                kind=str(element.get("type") or ""),
                text=text,
                raw=content,
                is_table=is_table,
            )
        )
    return out


def _get_page_id(element: Dict) -> Optional[int]:
    bbox = element.get("bbox")
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], dict):
        page_id = bbox[0].get("page_id")
        if isinstance(page_id, int):
            return page_id
    return None


def _page_context(elements: Sequence[_Element]) -> List[str]:
    candidates = []
    for element in elements:
        if element.is_table:
            continue
        if element.kind in {"title", "section_header", "caption", "page_header"}:
            if _meaningful_heading(element.text):
                candidates.append(element.text)
    if not candidates:
        for element in elements[:12]:
            if not element.is_table and _meaningful_heading(element.text):
                candidates.append(element.text)
    return _dedupe_keep_order(candidates)[-5:]


def _table_context(elements: Sequence[_Element], table_pos: int) -> List[str]:
    candidates = []
    for element in elements[max(0, table_pos - 18) : table_pos]:
        if element.is_table:
            continue
        if _meaningful_heading(element.text):
            candidates.append(element.text)
    return _dedupe_keep_order(candidates)[-5:]


def _meaningful_heading(text: str) -> bool:
    text = compact_whitespace(text)
    if not text or len(text) < 4 or len(text) > 240:
        return False
    if not re.search(r"[A-Za-z]", text):
        return False
    if GENERIC_HEADING_RE.search(text.strip(" .:-")):
        return False
    if len(tokenize(text)) <= 2 and not re.search(
        r"\b(?:table|operations|debt|receipts|expenditures|outlays)\b",
        text,
        re.IGNORECASE,
    ):
        return False
    return True


def _page_record_text(
    issue_title: str,
    source_file: str,
    page_id: int,
    context: Sequence[str],
    elements: Sequence[_Element],
) -> str:
    parts = [
        "Issue: {}".format(issue_title),
        "Source file: {}".format(source_file),
        "Page: {}".format(page_id),
    ]
    if context:
        parts.append("Page context: {}".format(" / ".join(context)))
    parts.extend(element.text for element in elements if element.text)
    return "\n".join(parts)


def _table_record_text(
    issue_title: str,
    source_file: str,
    page_id: int,
    context: Sequence[str],
    table_text: str,
) -> str:
    parts = [
        "Issue: {}".format(issue_title),
        "Source file: {}".format(source_file),
        "Page: {}".format(page_id),
    ]
    if context:
        parts.append("Table context: {}".format(" / ".join(context)))
        parts.append("Table title: {}".format(context[-1]))
    parts.append(table_text)
    return "\n".join(parts)


def _iter_row_records(
    source_file: str,
    year: Optional[int],
    month: Optional[int],
    page_id: int,
    table_index: int,
    table_id: str,
    title: str,
    context: Sequence[str],
    html: str,
) -> Iterator[CorpusRecord]:
    rows = _parse_table_rows(html)
    if len(rows) < 2:
        return
    header_count = min(3, max(1, len(rows) // 8))
    headers = rows[:header_count]
    data_start = header_count
    window_size = 6
    max_windows = 60
    data_rows = max(0, len(rows) - data_start)
    step = max(3, math.ceil(data_rows / max_windows)) if data_rows else 3

    for start in range(data_start, len(rows), step):
        end = min(len(rows), start + window_size)
        window_rows = rows[start:end]
        if not window_rows:
            continue
        text = _row_window_text(source_file, page_id, title, context, headers, window_rows)
        yield CorpusRecord(
            record_id="{}:rows:{}-{}".format(table_id, start + 1, end),
            record_type="row",
            text=text,
            source_file=source_file,
            page_id=page_id,
            year=year,
            month=month,
            title=title,
            section_path=list(context),
            table_index=table_index,
            metadata={"row_start": start + 1, "row_end": end},
        )


def _parse_table_rows(html: str) -> List[List[str]]:
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.rows


def _row_window_text(
    source_file: str,
    page_id: int,
    title: str,
    context: Sequence[str],
    headers: Sequence[Sequence[str]],
    rows: Sequence[Sequence[str]],
) -> str:
    parts = [
        "Source file: {}".format(source_file),
        "Page: {}".format(page_id),
        "Table title: {}".format(title),
    ]
    if context:
        parts.append("Context: {}".format(" / ".join(context)))
    if headers:
        parts.append("Headers:\n{}".format("\n".join(_format_row(row) for row in headers)))
    parts.append("Rows:\n{}".format("\n".join(_format_row(row) for row in rows)))
    return "\n".join(parts)


def _format_row(row: Sequence[str]) -> str:
    return " | ".join(cell for cell in row if cell)


def _record_from_row(row: sqlite3.Row) -> CorpusRecord:
    return CorpusRecord(
        record_id=row["record_id"],
        record_type=row["record_type"],
        text=row["text"],
        source_file=row["source_file"],
        page_id=row["page_id"],
        year=row["year"],
        month=row["month"],
        title=row["title"],
        section_path=json.loads(row["section_path"] or "[]"),
        table_index=row["table_index"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


def _normalize_query(question: str) -> str:
    text = question or ""
    text = text.replace("U.S.", "US").replace("U. S.", "US")
    text = re.sub(r"\bFY\b", "fiscal year", text)
    return compact_whitespace(text)


def _query_phrases(normalized: str, query_expansions: Dict[str, List[str]]) -> List[str]:
    lower = normalized.lower()
    phrases = []
    for trigger, expansions in query_expansions.items():
        if trigger in lower:
            phrases.extend(expansions)
    tokens = [token for token in tokenize(lower) if token not in STOPWORDS and len(token) > 2]
    for width in (3, 2):
        for i in range(0, max(0, len(tokens) - width + 1)):
            phrase = " ".join(tokens[i : i + width])
            if len(phrase) >= 12:
                phrases.append(phrase)
    return _dedupe_keep_order(phrases)[:12]


def _query_terms(normalized: str, phrases: Sequence[str]) -> List[str]:
    terms = []
    for token in tokenize(normalized):
        if token in STOPWORDS and not token.isdigit():
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        terms.append(token)
    for phrase in phrases:
        for token in tokenize(phrase):
            if token not in STOPWORDS or token.isdigit():
                terms.append(token)
    for month in MONTHS:
        if re.search(r"\b{}\b".format(month), normalized, flags=re.IGNORECASE):
            terms.append(month)
    return _dedupe_keep_order(terms)[:36]


def _match_expr(phrases: Sequence[str], terms: Sequence[str]) -> str:
    parts = []
    for phrase in phrases:
        phrase_terms = tokenize(phrase)
        if len(phrase_terms) > 1:
            parts.append('"{}"'.format(" ".join(_escape_match_term(term) for term in phrase_terms)))
    for term in terms:
        parts.append('"{}"'.format(_escape_match_term(term)))
    return " OR ".join(_dedupe_keep_order(parts))


def _escape_match_term(term: str) -> str:
    return term.replace('"', '""')


def _lateon_subset_ids(
    results: Sequence[SearchResult],
    lateon_record_ids_for_page: Optional[Callable[[str, int, bool], Sequence[str]]] = None,
    include_row_records: bool = False,
) -> List[str]:
    out = []
    seen = set()
    for result in results:
        if lateon_record_ids_for_page and result.record.page_id is not None:
            candidates = lateon_record_ids_for_page(
                result.record.source_file,
                result.record.page_id,
                include_row_records,
            )
        else:
            candidates = [
                result.record.metadata.get("evidence_record_id") if result.record.metadata else None,
                "{}:page:{}".format(result.record.source_file, result.record.page_id),
            ]
        for record_id in candidates:
            if not record_id or (not include_row_records and ":rows:" in record_id) or record_id in seen:
                continue
            seen.add(record_id)
            out.append(record_id)
    return out


def _extract_json_object(content: str) -> Dict:
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


def _merge_ints(base: Sequence[int], extra) -> List[int]:
    values = list(base)
    if isinstance(extra, list):
        for item in extra:
            try:
                values.append(int(item))
            except (TypeError, ValueError):
                pass
    return sorted(set(values))


def _merge_issue_dates(base: Sequence[Tuple[int, int]], extra) -> List[Tuple[int, int]]:
    values = list(base)
    if isinstance(extra, list):
        for item in extra:
            if not isinstance(item, dict):
                continue
            try:
                year = int(item.get("year"))
                month = int(item.get("month"))
            except (TypeError, ValueError):
                continue
            if 1 <= month <= 12:
                values.append((year, month))
    return sorted(set(values))


def _date_boost(plan: PageTableQueryPlan, source_file: str) -> float:
    match = SOURCE_FILE_RE.search(source_file or "")
    if not match:
        return 0.0
    file_year = int(match.group(1))
    file_month = int(match.group(2))
    boost = 0.0
    if (file_year, file_month) in set(plan.issue_dates):
        boost += 0.060
    if plan.years:
        deltas = [file_year - year for year in plan.years]
        if 1 in deltas:
            boost += 0.012
        elif 2 in deltas:
            boost += 0.008
        elif 0 in deltas:
            boost += 0.006
        elif -1 in deltas:
            boost += 0.003
    if plan.months and file_month in plan.months:
        boost += 0.004
    return boost


def _issue_dates(text: str) -> List[Tuple[int, int]]:
    dates = []
    lower = text.lower()
    for match in re.finditer(
        r"\b({})\s+((?:18|19|20)\d{{2}})\b".format("|".join(MONTHS.keys())),
        lower,
        flags=re.IGNORECASE,
    ):
        start, end = match.span()
        window = lower[max(0, start - 50) : min(len(lower), end + 50)]
        if re.search(r"\b(?:bulletin|bulletins|edition|monthly bulletin|treasury monthly)\b", window):
            dates.append((int(match.group(2)), MONTHS[match.group(1).lower()]))
    return sorted(set(dates))


def _page_numbers(text: str, pattern: str) -> List[int]:
    values = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        try:
            values.append(int(match.group(1)))
        except ValueError:
            pass
    return sorted(set(values))


def _page_penalty(plan: PageTableQueryPlan, record: Optional[CorpusRecord]) -> float:
    if record is None:
        return 0.0
    text = "{}\n{}".format(record.title or "", record.text[:500] if record.text else "").lower()
    question = plan.question.lower()
    if "table of contents" in text and "contents" not in question:
        return 0.018
    if re.search(r"\b(subscription|order form|subscribers|superintendent of documents)\b", text):
        return 0.020
    if "description of statistics" in text and "description" not in question:
        return 0.014
    return 0.0


def _page_title(text: str) -> Optional[str]:
    for line in (text or "").splitlines():
        line = compact_whitespace(line)
        if _meaningful_heading(line):
            return line
    return None


def _issue_title(year: Optional[int], month: Optional[int]) -> str:
    if year and month:
        month_name = next((name.title() for name, value in MONTHS.items() if value == month), str(month))
        return "Treasury Bulletin {} {}".format(month_name, year)
    if year:
        return "Treasury Bulletin {}".format(year)
    return "Treasury Bulletin"


def _dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out = []
    seen = set()
    for value in values:
        value = compact_whitespace(value)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out
