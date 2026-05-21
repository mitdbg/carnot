import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import PipelineConfig
from .dataset import CorpusLayout, discover_layout
from .llm import parallel_json


@dataclass
class FormatStats:
    format_id: str
    path: str
    file_count: int
    sample_chars: int
    has_page_markers: bool = False
    has_tables: bool = False
    notes: str = ""


@dataclass
class FormatVote:
    judge_id: str
    ranking: List[str]
    rationale: str


@dataclass
class FormatRecommendation:
    formats: List[FormatStats]
    votes: List[FormatVote] = field(default_factory=list)
    aggregate_scores: Dict[str, float] = field(default_factory=dict)
    recommended: str = "json"

    def to_dict(self) -> Dict:
        return {
            "formats": [f.__dict__ for f in self.formats],
            "votes": [v.__dict__ for v in self.votes],
            "aggregate_scores": self.aggregate_scores,
            "recommended": self.recommended,
        }


def collect_format_stats(layout: CorpusLayout, sample_files: int = 3) -> List[FormatStats]:
    stats = []
    mapping = {
        "json": layout.paths.json_dir,
        "page_txt": layout.paths.page_txt_dir,
        "txt": layout.paths.txt_dir,
        "pdf": layout.paths.pdf_dir,
    }
    for format_id, directory in mapping.items():
        if not directory or not directory.is_dir():
            continue
        files = sorted(directory.glob("*"))[: max(sample_files, 1)]
        if not files:
            continue
        sample_text = ""
        has_page = False
        has_tables = False
        for path in files:
            if path.suffix.lower() == ".json":
                text = path.read_text(encoding="utf-8", errors="replace")[:8000]
                has_tables = "<table" in text.lower()
            elif path.suffix.lower() == ".pdf":
                text = "[binary pdf — not sampled]"
            else:
                text = path.read_text(encoding="utf-8", errors="replace")[:8000]
                has_page = "--- PAGE " in text or bool(path.name.rsplit("_", 1)[-1].replace(".txt", "").isdigit())
            sample_text += text[:2500] + "\n"
        stats.append(
            FormatStats(
                format_id=format_id,
                path=str(directory),
                file_count=len(list(directory.glob("*"))),
                sample_chars=len(sample_text),
                has_page_markers=has_page,
                has_tables=has_tables,
            )
        )
    return stats


def recommend_canonical_format(
    data_dir: str,
    config: PipelineConfig,
    *,
    num_judges: Optional[int] = None,
) -> FormatRecommendation:
    layout = discover_layout(data_dir)
    formats = collect_format_stats(layout)
    if not formats:
        raise FileNotFoundError("No corpus formats found under {}".format(data_dir))

    if len(formats) == 1:
        return FormatRecommendation(
            formats=formats,
            aggregate_scores={formats[0].format_id: 1.0},
            recommended=formats[0].format_id,
        )

    judges = num_judges or config.format_judges
    tasks = []
    for judge_id in range(judges):
        tasks.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You judge which document corpus format is best for page-level retrieval "
                            "in grounded reasoning QA. Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _judge_prompt(formats, config.dataset_domain),
                    },
                ],
            }
        )

    raw_votes = parallel_json(
        tasks,
        provider=config.llm_provider,
        model=config.llm_model,
        timeout=config.llm_timeout,
        max_workers=config.parallel_llm,
    )

    votes: List[FormatVote] = []
    scores: Dict[str, float] = {f.format_id: 0.0 for f in formats}
    for index, raw in enumerate(raw_votes):
        ranking = raw.get("ranking") or raw.get("ranked_formats") or []
        if not isinstance(ranking, list):
            ranking = []
        ranking = [str(item) for item in ranking if str(item) in scores]
        rationale = str(raw.get("rationale") or raw.get("reason") or "")
        votes.append(FormatVote(judge_id="judge_{}".format(index + 1), ranking=ranking, rationale=rationale))
        for rank, format_id in enumerate(ranking):
            scores[format_id] += 1.0 / (rank + 1)

    recommended = max(scores, key=lambda key: scores[key]) if scores else formats[0].format_id
    return FormatRecommendation(
        formats=formats,
        votes=votes,
        aggregate_scores=scores,
        recommended=recommended,
    )


def interactive_choose_format(
    recommendation: FormatRecommendation,
    *,
    input_stream=None,
    output_stream=None,
) -> str:
    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stderr
    options = [f.format_id for f in recommendation.formats]
    print("\n=== Canonical corpus format ===", file=output_stream)
    for index, stats in enumerate(recommendation.formats, start=1):
        score = recommendation.aggregate_scores.get(stats.format_id, 0.0)
        mark = " *" if stats.format_id == recommendation.recommended else ""
        print(
            "  [{}] {}  files={}  path={}  judge_score={:.2f}{}".format(
                index, stats.format_id, stats.file_count, stats.path, score, mark
            ),
            file=output_stream,
        )
    for vote in recommendation.votes:
        print("  {} ranked: {} — {}".format(vote.judge_id, vote.ranking, vote.rationale[:120]), file=output_stream)

    default = recommendation.recommended
    prompt = "Choose canonical format [1-{}] (Enter={}): ".format(len(options), default)
    print(prompt, end="", file=output_stream)
    output_stream.flush()
    choice = (input_stream.readline() or "").strip()
    if not choice:
        return default
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return options[idx]
    if choice in options:
        return choice
    return default


def run_format_selection(
    data_dir: str,
    config: PipelineConfig,
    *,
    interactive: bool = True,
    save_report: bool = True,
) -> str:
    recommendation = recommend_canonical_format(data_dir, config)
    chosen = recommendation.recommended
    if interactive and len(recommendation.formats) > 1:
        chosen = interactive_choose_format(recommendation)

    layout = discover_layout(data_dir)
    rationale = "LLM judges recommended {}; user chose {}.".format(
        recommendation.recommended,
        chosen,
    )
    layout.save_canonical(chosen, rationale=rationale)

    if save_report:
        report_path = layout.meta_dir() / "format_selection_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = recommendation.to_dict()
        payload["chosen"] = chosen
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return chosen


def _judge_prompt(formats: Sequence[FormatStats], domain: str) -> str:
    lines = [
        "Domain: {}.".format(domain),
        "Rank formats for page-level retrieval (best first). Prefer structured page boundaries and tables.",
        "Return JSON: {\"ranking\": [\"json\", ...], \"rationale\": \"...\"}.",
        "Formats:",
    ]
    for stats in formats:
        lines.append(
            "- {}: files={}, page_markers={}, tables={}, sample_path={}".format(
                stats.format_id,
                stats.file_count,
                stats.has_page_markers,
                stats.has_tables,
                stats.path,
            )
        )
    return "\n".join(lines)
