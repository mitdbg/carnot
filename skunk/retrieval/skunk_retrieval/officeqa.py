import ast
import csv
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import parse_qs, urlparse

DEFAULT_REPO_ID = "databricks/officeqa"
DEFAULT_SPLIT = "train"
DEFAULT_PRO_FILE = "officeqa_pro.csv"
DEFAULT_FULL_FILE = "officeqa_full.csv"
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

URL_RE = re.compile(r"https?://[^\s,\]\)'\"<>]+")
MONTH_YEAR_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)-(\d{4})",
    re.IGNORECASE,
)
SOURCE_FILE_RE = re.compile(r"treasury_bulletin_\d{4}_\d{2}\.txt")


@dataclass(frozen=True)
class SourceRef:
    url: str
    source_file: Optional[str]
    page: Optional[int]
    year: Optional[int]
    month: Optional[int]


@dataclass(frozen=True)
class OfficeQARow:
    uid: str
    question: str
    answer: str
    source_docs: object
    source_files: object
    difficulty: str
    raw: Dict


@dataclass(frozen=True)
class OfficeQAOracle:
    source_files: List[str]
    source_refs: List[SourceRef]
    pages_by_source_file: Dict[str, List[int]]


@dataclass
class EvalRow:
    uid: str
    file_hit: bool
    page_hit: Optional[bool]
    best_file_rank: Optional[int]
    best_page_rank: Optional[int]
    gold_files: List[str]
    gold_pages: Dict[str, List[int]]
    predicted_files: List[str]
    predicted_pages: Dict[str, List[int]]
    top_results: List[Dict]
    elapsed_seconds: Optional[float] = None

    def to_dict(self):
        return {
            "uid": self.uid,
            "file_hit": self.file_hit,
            "page_hit": self.page_hit,
            "best_file_rank": self.best_file_rank,
            "best_page_rank": self.best_page_rank,
            "gold_files": self.gold_files,
            "gold_pages": self.gold_pages,
            "predicted_files": self.predicted_files,
            "predicted_pages": self.predicted_pages,
            "top_results": self.top_results,
            "elapsed_seconds": self.elapsed_seconds,
        }


def load_officeqa_rows(
    csv_path: Optional[str] = None,
    data_file: str = DEFAULT_PRO_FILE,
    repo_id: str = DEFAULT_REPO_ID,
    split: str = DEFAULT_SPLIT,
) -> List[OfficeQARow]:
    if csv_path:
        return list(read_officeqa_csv(csv_path))
    return list(load_officeqa_hf(data_file=data_file, repo_id=repo_id, split=split))


def read_officeqa_csv(path: str) -> Iterable[OfficeQARow]:
    with open(path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            yield normalize_officeqa_row(row)


def load_officeqa_hf(
    data_file: str = DEFAULT_PRO_FILE,
    repo_id: str = DEFAULT_REPO_ID,
    split: str = DEFAULT_SPLIT,
) -> Iterable[OfficeQARow]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install `datasets` to load OfficeQA from Hugging Face.") from exc

    dataset = load_dataset(repo_id, data_files=data_file, split=split)
    for row in dataset:
        yield normalize_officeqa_row(dict(row))


def normalize_officeqa_row(row: Dict) -> OfficeQARow:
    return OfficeQARow(
        uid=_first(row, "uid", "UID"),
        question=_first(row, "question", "Question"),
        answer=_first(row, "answer", "Answer"),
        source_docs=row.get("source_docs", row.get("source_doc", "")),
        source_files=row.get("source_files", row.get("source_file", "")),
        difficulty=_first(row, "difficulty", "Difficulty"),
        raw=row,
    )


def extract_oracle(row: OfficeQARow) -> OfficeQAOracle:
    source_refs = parse_source_docs(row.source_docs)
    source_files = set(parse_source_files(row.source_files))
    pages_by_source_file: Dict[str, List[int]] = {}

    for ref in source_refs:
        if ref.source_file:
            source_files.add(ref.source_file)
        if ref.source_file and ref.page is not None:
            pages_by_source_file.setdefault(ref.source_file, []).append(ref.page)

    return OfficeQAOracle(
        source_files=sorted(source_files),
        source_refs=source_refs,
        pages_by_source_file={
            source_file: sorted(set(pages))
            for source_file, pages in pages_by_source_file.items()
        },
    )


def evaluate_csv(
    retriever,
    benchmark_csv: str,
    k: int = 100,
    limit: Optional[int] = None,
    on_row=None,
) -> Dict:
    return evaluate_officeqa_rows(
        retriever,
        load_officeqa_rows(csv_path=benchmark_csv),
        k=k,
        limit=limit,
        on_row=on_row,
    )


def evaluate_hf(
    retriever,
    data_file: str,
    repo_id: str = DEFAULT_REPO_ID,
    split: str = DEFAULT_SPLIT,
    k: int = 100,
    limit: Optional[int] = None,
    on_row=None,
) -> Dict:
    return evaluate_officeqa_rows(
        retriever,
        load_officeqa_rows(data_file=data_file, repo_id=repo_id, split=split),
        k=k,
        limit=limit,
        on_row=on_row,
    )


def evaluate_officeqa_rows(
    retriever,
    rows: Iterable[OfficeQARow],
    k: int = 100,
    limit: Optional[int] = None,
    on_row=None,
) -> Dict:
    eval_rows = []
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        start = time.perf_counter()
        eval_rows.append(evaluate_row(retriever, row, k=k))
        eval_rows[-1].elapsed_seconds = round(time.perf_counter() - start, 6)
        if on_row:
            on_row(index, row, eval_rows[-1], eval_rows[-1].elapsed_seconds)
    return summarize_eval(eval_rows)


def evaluate_row(retriever, row: OfficeQARow, k: int = 100) -> EvalRow:
    oracle = extract_oracle(row)
    results = retriever.search(row.question, k=k)
    best_file_rank = _best_file_rank(results, set(oracle.source_files))
    best_page_rank = _best_page_rank(results, oracle.pages_by_source_file)
    page_hit = None if not oracle.pages_by_source_file else best_page_rank is not None

    return EvalRow(
        uid=row.uid,
        file_hit=best_file_rank is not None,
        page_hit=page_hit,
        best_file_rank=best_file_rank,
        best_page_rank=best_page_rank,
        gold_files=oracle.source_files,
        gold_pages=oracle.pages_by_source_file,
        predicted_files=_predicted_files(results),
        predicted_pages=_predicted_pages(results),
        top_results=[_result_report(result) for result in results[:20]],
    )


def summarize_eval(rows: Iterable[EvalRow]) -> Dict:
    rows = list(rows)
    total = len(rows)
    page_rows = [row for row in rows if row.page_hit is not None]
    file_hits = sum(1 for row in rows if row.file_hit)
    page_hits = sum(1 for row in page_rows if row.page_hit)
    return {
        "total": total,
        "file_recall": file_hits / total if total else 0.0,
        "page_total": len(page_rows),
        "page_recall": page_hits / len(page_rows) if page_rows else None,
        "rows": [row.to_dict() for row in rows],
    }


def _first(row: Dict, *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return str(value)
    return ""


def parse_source_docs(value) -> List[SourceRef]:
    return [_parse_url(url) for url in _extract_urls(value)]


def parse_source_files(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]

    text = str(value)
    parsed = _literal_or_none(text)
    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed if item]

    matches = SOURCE_FILE_RE.findall(text)
    if matches:
        return matches

    return [part.strip() for part in re.split(r"[,;]", text) if part.strip()]


def _extract_urls(value):
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        urls = []
        for item in value:
            urls.extend(_extract_urls(item))
        return urls

    text = str(value)
    parsed = _literal_or_none(text)
    if isinstance(parsed, (list, tuple)):
        return _extract_urls(parsed)
    return URL_RE.findall(text)


def _literal_or_none(text: str):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None


def _parse_url(url: str) -> SourceRef:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    page = _parse_int((qs.get("page") or [None])[0])

    match = MONTH_YEAR_RE.search(parsed.path.lower())
    if not match:
        return SourceRef(url=url, source_file=None, page=page, year=None, month=None)

    month = MONTHS[match.group(1).lower()]
    year = int(match.group(2))
    source_file = "treasury_bulletin_{:04d}_{:02d}.txt".format(year, month)
    return SourceRef(url=url, source_file=source_file, page=page, year=year, month=month)


def _parse_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(str(value))
    except ValueError:
        return None


def _best_file_rank(results, gold_files) -> Optional[int]:
    for result in results:
        if result.record.source_file in gold_files:
            return result.rank
    return None


def _best_page_rank(results, gold_pages: Dict[str, List[int]]) -> Optional[int]:
    for result in results:
        pages = gold_pages.get(result.record.source_file)
        if pages and result.record.page_id in pages:
            return result.rank
    return None


def _predicted_files(results) -> List[str]:
    files = []
    seen = set()
    for result in results:
        source_file = result.record.source_file
        if source_file and source_file not in seen:
            seen.add(source_file)
            files.append(source_file)
    return files


def _predicted_pages(results) -> Dict[str, List[int]]:
    pages_by_file: Dict[str, List[int]] = {}
    for result in results:
        source_file = result.record.source_file
        page_id = result.record.page_id
        if source_file and page_id is not None:
            pages_by_file.setdefault(source_file, []).append(page_id)
    return {
        source_file: sorted(set(pages))
        for source_file, pages in pages_by_file.items()
    }


def _result_report(result) -> Dict:
    record = result.record
    return {
        "rank": result.rank,
        "score": result.score,
        "channel": result.channel,
        "score_breakdown": result.score_breakdown,
        "record_id": record.record_id,
        "record_type": record.record_type,
        "source_file": record.source_file,
        "page_id": record.page_id,
        "year": record.year,
        "month": record.month,
        "title": record.title,
    }
