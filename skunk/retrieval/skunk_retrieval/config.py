import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


SKUNK_META_DIR = ".skunk"
CANONICAL_SOURCE_FILE = "canonical_source.json"
PIPELINE_CONFIG_FILE = "pipeline_config.json"
QUERY_EXPANSIONS_FILE = "query_expansions.json"
DATASET_PROFILE_FILE = "dataset_profile.json"


@dataclass
class PipelineConfig:
    data_dir: str = "/tmp/officeqa"
    results_dir: Optional[str] = None

    page_table_index: Optional[str] = None
    lateon_dir: Optional[str] = None

    page_k: int = 800
    table_k: int = 800
    row_k: int = 200
    file_k: int = 80

    use_lateon: bool = True
    lateon_candidate_k: int = 500
    lateon_rerank_k: int = 1000
    lateon_expand_page_records: bool = True
    lateon_include_row_records: bool = True
    lateon_weight: float = 2.0

    eval_k: int = 500

    llm_provider: str = "openrouter"
    llm_model: str = "google/gemini-2.5-flash"
    llm_timeout: float = 25.0
    parallel_llm: int = 4

    use_llm_planner: bool = True
    decompose_hard: bool = True
    retry_hard: bool = True
    max_sub_queries: int = 12

    format_judges: int = 3
    canonical_source: Optional[str] = None

    dataset_domain: str = "financial and statistical government documents"
    planner_domain_hint: str = ""

    extra: Dict[str, Any] = field(default_factory=dict)

    def meta_dir(self) -> Path:
        return Path(self.data_dir).expanduser() / SKUNK_META_DIR

    def results_path(self) -> Path:
        if self.results_dir:
            return Path(self.results_dir).expanduser()
        return Path(self.data_dir).expanduser() / "results"

    def index_path(self) -> Path:
        if self.page_table_index:
            return Path(self.page_table_index).expanduser()
        return Path(self.data_dir).expanduser() / "page_table.sqlite"

    def lateon_path(self) -> Path:
        if self.lateon_dir:
            return Path(self.lateon_dir).expanduser()
        return Path(self.data_dir).expanduser() / "lateon"

    def save(self) -> Path:
        path = self.meta_dir() / PIPELINE_CONFIG_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    @classmethod
    def load(cls, data_dir: str) -> "PipelineConfig":
        path = Path(data_dir).expanduser() / SKUNK_META_DIR / PIPELINE_CONFIG_FILE
        if not path.is_file():
            cfg = cls(data_dir=data_dir)
            cfg._apply_env()
            return cfg
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {key: raw[key] for key in raw if key in known}
        extra = {key: raw[key] for key in raw if key not in known}
        cfg = cls(**kwargs, extra=extra)
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        if os.environ.get("SKUNK_DATA_DIR"):
            self.data_dir = os.environ["SKUNK_DATA_DIR"]
        if os.environ.get("SKUNK_LLM_PROVIDER"):
            self.llm_provider = os.environ["SKUNK_LLM_PROVIDER"]
        if os.environ.get("SKUNK_LLM_MODEL"):
            self.llm_model = os.environ["SKUNK_LLM_MODEL"]
        if os.environ.get("OPENROUTER_API_KEY") and self.llm_provider == "openrouter":
            pass
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            pass
        if os.environ.get("SKUNK_CANONICAL_SOURCE"):
            self.canonical_source = os.environ["SKUNK_CANONICAL_SOURCE"]
        if os.environ.get("EVAL_K"):
            self.eval_k = int(os.environ["EVAL_K"])
        disable = os.environ.get("SKUNK_DISABLE", "")
        if "lateon" in disable:
            self.use_lateon = False
        if "llm" in disable:
            self.use_llm_planner = False
        if "decompose" in disable:
            self.decompose_hard = False
        if "retry" in disable:
            self.retry_hard = False
        if "rows" in disable:
            self.row_k = 0


def default_pipeline_config(data_dir: str) -> PipelineConfig:
    return PipelineConfig.load(data_dir)
