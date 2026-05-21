import json
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


@dataclass
class CorpusRecord:
    record_id: str
    record_type: str
    text: str
    source_file: str
    page_id: Optional[int] = None
    year: Optional[int] = None
    month: Optional[int] = None
    title: Optional[str] = None
    section_path: List[str] = field(default_factory=list)
    table_index: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [
            self.source_file,
            self.record_type,
            str(self.year or ""),
            str(self.month or ""),
            self.title or "",
            " ".join(self.section_path),
            self.text,
        ]
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusRecord":
        return cls(**data)


@dataclass
class SearchResult:
    record: CorpusRecord
    score: float
    rank: int
    channel: str
    score_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "channel": self.channel,
            "score_breakdown": self.score_breakdown,
            "record": self.record.to_dict(),
        }

DEFAULT_RECORD_SOURCES = ("json", "page_txt", "txt")
PAGE_MARKER_RE = re.compile(r"^--- PAGE (\d+) ---\s*$")
SPLIT_PAGE_FILE_RE = re.compile(r"^(treasury_bulletin_\d{4}_\d{2})_(\d+)\.txt$")
SOURCE_FILE_DATE_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


def build_records(data_dir: str, sources: Sequence[str] = DEFAULT_RECORD_SOURCES) -> List[CorpusRecord]:
    data_path = Path(data_dir).expanduser()
    records = []

    if "json" in sources:
        records.extend(iter_json_records(data_path / "treasury_bulletins_parsed" / "jsons"))
    if "page_txt" in sources:
        records.extend(iter_page_text_records(data_path / "treasury_bulletins_parsed" / "transformed_page_level"))
    if "txt" in sources:
        records.extend(iter_transformed_text_records(data_path / "treasury_bulletins_parsed" / "transformed"))

    deduped: Dict[str, CorpusRecord] = {}
    for record in records:
        deduped.setdefault(record.record_id, record)
    return list(deduped.values())


def iter_json_records(json_dir: Path) -> Iterator[CorpusRecord]:
    if not json_dir.is_dir():
        return

    for path in sorted(json_dir.glob("treasury_bulletin_*.json")):
        source_file = path.with_suffix(".txt").name
        year, month = parse_year_month(source_file)
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)

        pages: Dict[int, List[str]] = {}
        recent_text: Dict[int, List[str]] = {}
        table_count = 0

        for element_index, element in enumerate(_extract_elements(obj)):
            content = element.get("content") if isinstance(element, dict) else None
            if not isinstance(content, str) or not content.strip():
                continue

            page_id = _get_page_id(element) or 0
            is_table = "<table" in content.lower()
            text = html_to_text(content) if is_table else content.strip()
            pages.setdefault(page_id, []).append(text)

            if is_table:
                table_count += 1
                title = _infer_title(recent_text.get(page_id, []))
                yield CorpusRecord(
                    record_id="{}:page:{}:table:{}".format(source_file, page_id, table_count),
                    record_type="table",
                    text=_table_record_text(title, recent_text.get(page_id, []), text),
                    source_file=source_file,
                    page_id=page_id,
                    year=year,
                    month=month,
                    title=title,
                    table_index=table_count,
                    metadata={"element_index": element_index},
                )
            else:
                cleaned = compact_whitespace(content)
                if cleaned:
                    recent_text.setdefault(page_id, []).append(cleaned)
                    recent_text[page_id] = recent_text[page_id][-8:]

        for page_id, page_parts in sorted(pages.items()):
            text = "\n".join(part for part in page_parts if part).strip()
            if not text:
                continue
            yield CorpusRecord(
                record_id="{}:page:{}".format(source_file, page_id),
                record_type="page",
                text=text,
                source_file=source_file,
                page_id=page_id,
                year=year,
                month=month,
            )


def iter_page_text_records(page_dir: Path) -> Iterator[CorpusRecord]:
    if not page_dir.is_dir():
        return

    for path in sorted(page_dir.glob("treasury_bulletin_*.txt")):
        split_match = SPLIT_PAGE_FILE_RE.match(path.name)
        if split_match:
            source_file = split_match.group(1) + ".txt"
            page_id = int(split_match.group(2))
            text = path.read_text(encoding="utf-8")
            yield _page_record(source_file, page_id, text, "page_txt")
            continue

        source_file = path.name
        text = path.read_text(encoding="utf-8")
        for page_id, page_text in split_page_markers(text):
            yield _page_record(source_file, page_id, page_text, "page_txt")


def iter_transformed_text_records(text_dir: Path, max_chars: int = 6000) -> Iterator[CorpusRecord]:
    if not text_dir.is_dir():
        return

    for path in sorted(text_dir.glob("treasury_bulletin_*.txt")):
        source_file = path.name
        year, month = parse_year_month(source_file)
        lines = path.read_text(encoding="utf-8").splitlines()
        for chunk_index, chunk in enumerate(chunk_lines(lines, max_chars=max_chars), start=1):
            yield CorpusRecord(
                record_id="{}:chunk:{}".format(source_file, chunk_index),
                record_type="chunk",
                text=chunk,
                source_file=source_file,
                page_id=None,
                year=year,
                month=month,
                metadata={"chunk_index": chunk_index},
            )


def split_page_markers(text: str) -> Iterator[Tuple[int, str]]:
    current_page: Optional[int] = None
    current_lines: List[str] = []

    for line in text.splitlines():
        match = PAGE_MARKER_RE.match(line.strip())
        if match:
            if current_page is not None and current_lines:
                yield current_page, "\n".join(current_lines).strip()
            current_page = int(match.group(1))
            current_lines = []
        elif current_page is not None:
            current_lines.append(line)

    if current_page is not None and current_lines:
        yield current_page, "\n".join(current_lines).strip()


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.text()


def write_records_jsonl(records: Iterable[CorpusRecord], path: str) -> int:
    count = 0
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_records_jsonl(path: str) -> List[CorpusRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(CorpusRecord.from_dict(json.loads(line)))
    return records


def parse_year_month(source_file: str) -> Tuple[Optional[int], Optional[int]]:
    match = SOURCE_FILE_DATE_RE.search(source_file or "")
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_lines(lines: Iterable[str], max_chars: int = 6000) -> List[str]:
    chunks = []
    current = []
    current_len = 0

    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        extra = len(line) + 1
        if current and current_len + extra > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += extra

    if current:
        chunks.append("\n".join(current))
    return chunks


def _page_record(source_file: str, page_id: int, text: str, source: str) -> CorpusRecord:
    year, month = parse_year_month(source_file)
    return CorpusRecord(
        record_id="{}:page:{}".format(source_file, page_id),
        record_type="page",
        text=text.strip(),
        source_file=source_file,
        page_id=page_id,
        year=year,
        month=month,
        metadata={"source": source},
    )


def _extract_elements(obj: dict) -> Iterable[dict]:
    document = obj.get("document") or {}
    elements = document.get("elements")
    return elements if isinstance(elements, list) else []


def _get_page_id(element: dict) -> Optional[int]:
    bbox = element.get("bbox")
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], dict):
        page_id = bbox[0].get("page_id")
        if isinstance(page_id, int):
            return page_id
    return None


def _infer_title(recent_text: List[str]) -> Optional[str]:
    for text in reversed(recent_text):
        if len(text) <= 220:
            return text
    return recent_text[-1] if recent_text else None


def _table_record_text(title: Optional[str], preamble: List[str], table_text: str) -> str:
    parts = []
    if title:
        parts.append("Title: {}".format(title))
    if preamble:
        parts.append("Preamble:\n{}".format("\n".join(preamble[-4:])))
    parts.append("Table:\n{}".format(table_text))
    return "\n\n".join(parts)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.parts: List[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("tr", "p", "br"):
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_data(self, data):
        if data and data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        lines = [compact_whitespace(line) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)
