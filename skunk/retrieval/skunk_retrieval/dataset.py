import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import CANONICAL_SOURCE_FILE, QUERY_EXPANSIONS_FILE, SKUNK_META_DIR


@dataclass
class CorpusPaths:
    json_dir: Optional[Path] = None
    page_txt_dir: Optional[Path] = None
    txt_dir: Optional[Path] = None
    pdf_dir: Optional[Path] = None

    def available_formats(self) -> List[str]:
        out = []
        if self.json_dir and self.json_dir.is_dir():
            out.append("json")
        if self.page_txt_dir and self.page_txt_dir.is_dir():
            out.append("page_txt")
        if self.txt_dir and self.txt_dir.is_dir():
            out.append("txt")
        if self.pdf_dir and self.pdf_dir.is_dir():
            out.append("pdf")
        return out


@dataclass
class CorpusLayout:
    data_dir: Path
    paths: CorpusPaths
    canonical_source: str = "json"
    file_glob: str = "*"
    source_file_pattern: str = r"(.+)\.(txt|json|pdf)$"

    def meta_dir(self) -> Path:
        return self.data_dir / SKUNK_META_DIR

    def canonical_path(self) -> Path:
        return self.meta_dir() / CANONICAL_SOURCE_FILE

    def save_canonical(self, source: str, rationale: str = "") -> Path:
        self.meta_dir().mkdir(parents=True, exist_ok=True)
        payload = {"canonical_source": source, "rationale": rationale}
        with open(self.canonical_path(), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return self.canonical_path()

    def load_query_expansions(self) -> Dict[str, List[str]]:
        path = self.meta_dir() / QUERY_EXPANSIONS_FILE
        if not path.is_file():
            return {}
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            return {}
        out = {}
        for key, value in raw.items():
            if isinstance(value, list):
                out[str(key)] = [str(item) for item in value]
        return out


def discover_layout(data_dir: str, canonical_source: Optional[str] = None) -> CorpusLayout:
    root = Path(data_dir).expanduser()
    paths = CorpusPaths(
        json_dir=_first_existing(
            root / "treasury_bulletins_parsed" / "jsons",
            root / "parsed" / "jsons",
            root / "jsons",
        ),
        page_txt_dir=_first_existing(
            root / "treasury_bulletins_parsed" / "transformed_page_level",
            root / "parsed" / "transformed_page_level",
            root / "page_txt",
        ),
        txt_dir=_first_existing(
            root / "treasury_bulletins_parsed" / "transformed",
            root / "parsed" / "transformed",
            root / "txt",
        ),
        pdf_dir=_first_existing(
            root / "treasury_bulletin_pdfs",
            root / "pdfs",
            root / "pdf",
        ),
    )
    source = canonical_source or _load_saved_canonical(root) or _default_canonical(paths)
    return CorpusLayout(data_dir=root, paths=paths, canonical_source=source)


def _first_existing(*candidates: Path) -> Optional[Path]:
    for path in candidates:
        if path.is_dir() and any(path.iterdir()):
            return path
    return None


def _load_saved_canonical(root: Path) -> Optional[str]:
    path = root / SKUNK_META_DIR / CANONICAL_SOURCE_FILE
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    return raw.get("canonical_source")


def _default_canonical(paths: CorpusPaths) -> str:
    available = paths.available_formats()
    if "json" in available:
        return "json"
    if "page_txt" in available:
        return "page_txt"
    if "txt" in available:
        return "txt"
    return "json"
