import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .config import PipelineConfig
from .llm import chat_json
from .page_table import PageTableQueryPlan, build_page_table_query_plan


YEAR_RANGE_RE = re.compile(
    r"\b(?:from|between)\s+((?:18|19|20)\d{2})\s+(?:through|to|and|-)\s+((?:18|19|20)\d{2})\b",
    re.I,
)
YEAR_LIST_RE = re.compile(r"\b((?:18|19|20)\d{2})\b")


@dataclass
class SubQuery:
    query: str
    reason: str = ""


def is_hard_question(question: str, plan: Optional[PageTableQueryPlan] = None) -> bool:
    plan = plan or build_page_table_query_plan(question)
    years = YEAR_LIST_RE.findall(question or "")
    unique_years = sorted(set(int(y) for y in years))
    if len(unique_years) >= 3:
        return True
    if len(plan.issue_dates) >= 2:
        return True
    if YEAR_RANGE_RE.search(question or ""):
        return True
    if re.search(r"\b(?:each|every|across|respectively|time series|calendar years|fiscal years)\b", question, re.I):
        if len(unique_years) >= 2:
            return True
    if re.search(r"\b(?:sum|mean|average|correlation|compare|change)\b", question, re.I) and len(unique_years) >= 2:
        return True
    return False


def decompose_question(question: str, config: PipelineConfig) -> List[SubQuery]:
    if not config.decompose_hard:
        return [SubQuery(query=question, reason="decompose_disabled")]

    heuristic_subs = _heuristic_decompose(question)
    if not config.use_llm_planner:
        return heuristic_subs or [SubQuery(query=question)]

    payload = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "You split document-retrieval questions into focused sub-queries. "
                    "Return JSON only. Do not answer the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Split this question into sub-queries that each target specific document issues "
                    "(year/month), tables, or pages needed for grounded reasoning. "
                    "Domain: {}. Max {} sub-queries. "
                    'Return {"sub_queries": [{"query": "...", "reason": "..."}]}. Question: '
                    + repr(question)
                ).format(
                    config.planner_domain_hint or config.dataset_domain,
                    config.max_sub_queries,
                ),
            },
        ],
        provider=config.llm_provider,
        model=config.llm_model,
        timeout=config.llm_timeout,
    )
    subs = []
    for item in payload.get("sub_queries") or []:
        if isinstance(item, dict) and item.get("query"):
            subs.append(SubQuery(query=str(item["query"]), reason=str(item.get("reason") or "")))
    if not subs and heuristic_subs:
        return heuristic_subs[: config.max_sub_queries]
    if not subs:
        return [SubQuery(query=question)]
    return subs[: config.max_sub_queries]


def retry_hints(question: str, first_query: str, config: PipelineConfig) -> Optional[str]:
    if not config.retry_hard or not config.use_llm_planner:
        return None
    payload = chat_json(
        [
            {
                "role": "system",
                "content": (
                    "You identify missing retrieval targets after an initial search. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    'Return {"retry_query": "short focused retrieval query", "phrases": [], "terms": []}. '
                    "The retry_query should target document issues/pages likely still missing. "
                    "Original: {} Initial sub-query: {}".format(repr(question), repr(first_query))
                ),
            },
        ],
        provider=config.llm_provider,
        model=config.llm_model,
        timeout=config.llm_timeout,
    )
    retry = payload.get("retry_query")
    return str(retry).strip() if retry else None


def _heuristic_decompose(question: str) -> List[SubQuery]:
    text = question or ""
    years = sorted(set(int(y) for y in YEAR_LIST_RE.findall(text)))
    subs: List[SubQuery] = []

    range_match = YEAR_RANGE_RE.search(text)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if end >= start and (end - start) <= 40:
            years = list(range(start, end + 1))

    if len(years) >= 2 and re.search(r"\b(?:june|march|september|december|january|february|april|may|july|august|october|november)\b", text, re.I):
        month_match = re.search(
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
            text,
            re.I,
        )
        month = month_match.group(1) if month_match else ""
        for year in years:
            q = "{} {}".format(month, year).strip()
            if month:
                subs.append(SubQuery(query="{} — {}".format(q, text[:200]), reason="year_month_slice"))
            else:
                subs.append(SubQuery(query="{} {}".format(year, text[:160]), reason="year_slice"))

    if len(years) >= 3 and not subs:
        for year in years:
            subs.append(SubQuery(query="{} {}".format(year, text[:180]), reason="multi_year"))

    return subs
