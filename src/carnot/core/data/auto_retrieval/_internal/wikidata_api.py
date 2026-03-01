from __future__ import annotations

import json as _json
import logging
import re
import threading
import time
from typing import Dict, List, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_VALID_QID = re.compile(r"^Q[1-9]\d*$")


def _sanitize_qids(qids: List[str]) -> List[str]:
    """Filter to valid QIDs, dedupe, preserve order."""
    seen: Set[str] = set()
    out: List[str] = []
    for q in qids:
        if not q or not isinstance(q, str):
            continue
        q = str(q).strip()
        if _VALID_QID.match(q) and q not in seen:
            seen.add(q)
            out.append(q)
    return out

SEARCH_URL = "https://www.wikidata.org/w/api.php"
SPARQL_URL = "https://query.wikidata.org/sparql"

DEFAULT_SEARCH_TIMEOUT = 30
DEFAULT_SPARQL_TIMEOUT = 60
DEFAULT_MAX_RPS = 3.0
DEFAULT_MAX_RETRIES = 3
SPARQL_BATCH_SIZE = 30
LABELS_BATCH_SIZE = 50
SPARQL_COOLDOWN = 0.5
USER_AGENT = "Carnot/1.0 (Research; https://github.com/carnot; Wikidata entity resolution)"


class WikidataAPI:
    _sparql_lock = threading.Lock()
    _last_sparql_time = 0.0

    def __init__(
        self,
        max_rps: float = DEFAULT_MAX_RPS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self._min_interval = 1.0 / max_rps
        self._last_request_time = 0.0
        self._max_retries = max_retries

    def _rate_limit(self) -> None:
        now = time.monotonic()
        gap = self._min_interval - (now - self._last_request_time)
        if gap > 0:
            time.sleep(gap)
        self._last_request_time = time.monotonic()

    def _sparql_throttle(self) -> None:
        with WikidataAPI._sparql_lock:
            now = time.monotonic()
            elapsed = now - WikidataAPI._last_sparql_time
            if elapsed < SPARQL_COOLDOWN:
                time.sleep(SPARQL_COOLDOWN - elapsed)
            WikidataAPI._last_sparql_time = time.monotonic()

    def _get_json(self, url: str, *, timeout: int) -> dict:
        """Fetch JSON with retries, rate-limiting, and capped exponential backoff.

        Backoff schedule: 1s, 2s, 4s (capped at 4s to avoid long hangs when
        WDQS is flaky).  Total worst-case wait = 7s before final failure.
        """
        max_backoff = 4
        for attempt in range(self._max_retries):
            self._rate_limit()
            req = Request(url)
            req.add_header("User-Agent", USER_AGENT)
            req.add_header("Accept", "application/json")
            try:
                with urlopen(req, timeout=timeout) as resp:
                    return _json.loads(resp.read().decode())
            except HTTPError as exc:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait = min(int(retry_after) if retry_after else 2 ** attempt, max_backoff)
                    logger.warning("429 on attempt %d/%d, waiting %ds",
                                   attempt + 1, self._max_retries, wait)
                    time.sleep(wait)
                    continue
                if exc.code >= 500:
                    wait = min(2 ** attempt, max_backoff)
                    logger.warning("HTTP %d on attempt %d/%d, backing off %ds — %s",
                                   exc.code, attempt + 1, self._max_retries,
                                   wait, url[:100])
                    time.sleep(wait)
                    continue
                raise
            except (URLError, TimeoutError, OSError) as exc:
                wait = min(2 ** attempt, max_backoff)
                logger.warning(
                    "%s on attempt %d/%d (timeout=%ds), backing off %ds — %s",
                    type(exc).__name__, attempt + 1, self._max_retries,
                    timeout, wait, url[:100],
                )
                time.sleep(wait)
        raise RuntimeError(f"Failed after {self._max_retries} retries: {url[:200]}")

    def _sparql_post(self, query: str, *, timeout: int) -> dict:
        """POST SPARQL query to avoid URL length limits."""
        data = urlencode({"query": query, "format": "json"}).encode()
        req = Request(SPARQL_URL, data=data, method="POST")
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        req.add_header("Accept", "application/json")
        max_backoff = 4
        for attempt in range(self._max_retries):
            self._rate_limit()
            try:
                with urlopen(req, timeout=timeout) as resp:
                    return _json.loads(resp.read().decode())
            except HTTPError as exc:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                    wait = min(int(retry_after) if retry_after else 2 ** attempt, max_backoff)
                    logger.warning("429 on attempt %d/%d, waiting %ds", attempt + 1, self._max_retries, wait)
                    time.sleep(wait)
                    continue
                if exc.code >= 500:
                    wait = min(2 ** attempt, max_backoff)
                    logger.warning("HTTP %d on attempt %d/%d, backing off %ds", exc.code, attempt + 1, self._max_retries, wait)
                    time.sleep(wait)
                    continue
                if exc.code == 400:
                    body = exc.fp.read().decode(errors="replace") if exc.fp else ""
                    logger.error("SPARQL 400 response body: %s", body[:1000])
                    logger.error("SPARQL query (first 1500 chars): %s", query[:1500])
                raise
            except (URLError, TimeoutError, OSError) as exc:
                wait = min(2 ** attempt, max_backoff)
                logger.warning("%s on attempt %d/%d (timeout=%ds), backing off %ds", type(exc).__name__, attempt + 1, self._max_retries, timeout, wait)
                time.sleep(wait)
        raise RuntimeError(f"SPARQL POST failed after {self._max_retries} retries")

    def search_entities(
        self, query: str, language: str = "en", limit: int = 10,
    ) -> List[Dict[str, str]]:
        params = urlencode({
            "action": "wbsearchentities", "format": "json",
            "language": language, "uselang": language,
            "type": "item", "limit": limit, "search": query,
        })
        data = self._get_json(f"{SEARCH_URL}?{params}", timeout=DEFAULT_SEARCH_TIMEOUT)
        return [
            {
                "qid": item.get("id", ""),
                "label": item.get("label", ""),
                "description": item.get("description", ""),
            }
            for item in data.get("search", [])
        ]

    def fetch_labels_batch(
        self, qids: List[str], language: str = "en",
    ) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for start in range(0, len(qids), LABELS_BATCH_SIZE):
            chunk = qids[start : start + LABELS_BATCH_SIZE]
            params = urlencode({
                "action": "wbgetentities", "format": "json",
                "ids": "|".join(chunk),
                "props": "labels", "languages": language,
            })
            try:
                data = self._get_json(
                    f"{SEARCH_URL}?{params}", timeout=DEFAULT_SEARCH_TIMEOUT,
                )
            except Exception:
                logger.exception("fetch_labels_batch failed for chunk at %d", start)
                continue
            for qid, entity in data.get("entities", {}).items():
                labels = entity.get("labels", {})
                lbl = labels.get(language, {}).get("value")
                if lbl:
                    out[qid] = lbl
        return out

    def validate_types_batch(
        self,
        qids: List[str],
        type_qids: List[str],
    ) -> Set[str]:
        type_qids = _sanitize_qids(type_qids)
        if not qids or not type_qids:
            return set()

        valid: Set[str] = set()
        for start in range(0, len(qids), SPARQL_BATCH_SIZE):
            self._sparql_throttle()
            chunk = _sanitize_qids(qids[start : start + SPARQL_BATCH_SIZE])
            if not chunk:
                continue
            items = " ".join(f"wd:{q}" for q in chunk)
            types = " ".join(f"wd:{t}" for t in type_qids)
            query = (
                "SELECT DISTINCT ?item WHERE {\n"
                f"  VALUES ?item {{ {items} }}\n"
                f"  VALUES ?type {{ {types} }}\n"
                "  ?item wdt:P31/wdt:P279* ?type .\n"
                "}"
            )
            try:
                data = self._sparql_post(query, timeout=DEFAULT_SPARQL_TIMEOUT)
            except Exception:
                logger.exception("validate_types_batch SPARQL failed (offset %d)", start)
                continue
            for b in data.get("results", {}).get("bindings", []):
                uri = b.get("item", {}).get("value", "")
                if uri:
                    valid.add(uri.rsplit("/", 1)[-1])
        return valid

    # ------------------------------------------------------------------
    # Audit helpers — type introspection for ontology gap analysis
    # ------------------------------------------------------------------

    def fetch_p31_types_batch(
        self, qids: List[str], language: str = "en",
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Fetch direct P31 (instance of) types for a batch of QIDs.

        Returns ``{qid: [(type_qid, type_label), ...]}``.
        Uses a single VALUES clause per chunk to keep WDQS load low.
        """
        all_results: Dict[str, List[Tuple[str, str]]] = {q: [] for q in qids}
        for start in range(0, len(qids), SPARQL_BATCH_SIZE):
            self._sparql_throttle()
            chunk = _sanitize_qids(qids[start : start + SPARQL_BATCH_SIZE])
            if not chunk:
                continue
            items = " ".join(f"wd:{q}" for q in chunk)
            query = (
                "SELECT ?item ?type ?typeLabel WHERE {\n"
                f"  VALUES ?item {{ {items} }}\n"
                "  ?item wdt:P31 ?type .\n"
                f'  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language},en" . }}\n'
                "}"
            )
            try:
                data = self._sparql_post(query, timeout=DEFAULT_SPARQL_TIMEOUT)
            except Exception:
                logger.exception("fetch_p31_types_batch SPARQL failed (offset %d)", start)
                continue
            for b in data.get("results", {}).get("bindings", []):
                item_uri = b.get("item", {}).get("value", "")
                type_uri = b.get("type", {}).get("value", "")
                type_label = b.get("typeLabel", {}).get("value", "")
                item_qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
                type_qid = type_uri.rsplit("/", 1)[-1] if type_uri else ""
                if item_qid in all_results and type_qid:
                    all_results[item_qid].append((type_qid, type_label))
        return all_results

    def fetch_class_instance_signals_batch(
        self, qids: List[str],
    ) -> Dict[str, Dict[str, int]]:
        """Return lightweight class/instance structure counts per QID.

        Output shape:
          {qid: {"p279_count": int, "p31_count": int}}
        """
        out: Dict[str, Dict[str, int]] = {
            q: {"p279_count": 0, "p31_count": 0}
            for q in qids
        }
        for start in range(0, len(qids), SPARQL_BATCH_SIZE):
            self._sparql_throttle()
            chunk = _sanitize_qids(qids[start : start + SPARQL_BATCH_SIZE])
            if not chunk:
                continue
            items = " ".join(f"wd:{q}" for q in chunk)
            query = (
                "SELECT ?item "
                "(COUNT(DISTINCT ?superClass) AS ?p279Count) "
                "(COUNT(DISTINCT ?instanceType) AS ?p31Count) WHERE {\n"
                f"  VALUES ?item {{ {items} }}\n"
                "  OPTIONAL { ?item wdt:P279 ?superClass . }\n"
                "  OPTIONAL { ?item wdt:P31 ?instanceType . }\n"
                "} GROUP BY ?item"
            )
            try:
                data = self._sparql_post(query, timeout=DEFAULT_SPARQL_TIMEOUT)
            except Exception:
                logger.exception("fetch_class_instance_signals_batch SPARQL failed (offset %d)", start)
                continue
            for b in data.get("results", {}).get("bindings", []):
                item_uri = b.get("item", {}).get("value", "")
                item_qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
                if not item_qid:
                    continue
                p279_raw = b.get("p279Count", {}).get("value", "0")
                p31_raw = b.get("p31Count", {}).get("value", "0")
                try:
                    p279_count = int(p279_raw)
                except Exception:
                    p279_count = 0
                try:
                    p31_count = int(p31_raw)
                except Exception:
                    p31_count = 0
                out[item_qid] = {
                    "p279_count": max(0, p279_count),
                    "p31_count": max(0, p31_count),
                }
        return out

    def match_types_batch(
        self,
        qids: List[str],
        type_qids: List[str],
    ) -> Dict[str, Set[str]]:
        """Check which *type_qids* each item reaches via ``P31/P279*``.

        Like :meth:`validate_types_batch` but returns the matched types,
        not just a boolean.  Returns ``{item_qid: {matched_type_qids}}``.
        """
        all_results: Dict[str, Set[str]] = {q: set() for q in qids}
        if not qids or not type_qids:
            return all_results
        types_str = " ".join(f"wd:{t}" for t in _sanitize_qids(type_qids))
        if not types_str:
            return all_results
        for start in range(0, len(qids), SPARQL_BATCH_SIZE):
            self._sparql_throttle()
            chunk = _sanitize_qids(qids[start : start + SPARQL_BATCH_SIZE])
            if not chunk:
                continue
            items = " ".join(f"wd:{q}" for q in chunk)
            query = (
                "SELECT DISTINCT ?item ?type WHERE {\n"
                f"  VALUES ?item {{ {items} }}\n"
                f"  VALUES ?type {{ {types_str} }}\n"
                "  ?item wdt:P31/wdt:P279* ?type .\n"
                "}"
            )
            try:
                data = self._sparql_post(query, timeout=DEFAULT_SPARQL_TIMEOUT)
            except Exception:
                logger.exception("match_types_batch SPARQL failed (offset %d)", start)
                continue
            for b in data.get("results", {}).get("bindings", []):
                item_uri = b.get("item", {}).get("value", "")
                type_uri = b.get("type", {}).get("value", "")
                item_qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
                type_qid = type_uri.rsplit("/", 1)[-1] if type_uri else ""
                if item_qid in all_results and type_qid:
                    all_results[item_qid].add(type_qid)
        return all_results

    # ------------------------------------------------------------------
    # Parent hierarchy
    # ------------------------------------------------------------------

    @staticmethod
    def _build_parents_query(
        qids: List[str], property_id: str, max_depth: int, language: str = "en",
    ) -> str:
        values = " ".join(f"wd:{q}" for q in qids)
        prop = f"wdt:{property_id}"
        patterns: List[str] = []
        for d in range(1, max_depth + 1):
            parts: List[str] = []
            for step in range(d):
                src = "?item" if step == 0 else f"?_s{step}"
                dst = f"?_s{step + 1}" if step < d - 1 else "?ancestor"
                parts.append(f"{src} {prop} {dst}")
            chain = " . ".join(parts)
            patterns.append(f"{{ {chain} . BIND({d} AS ?depth) }}")
        union = "\n    UNION\n    ".join(patterns)
        return (
            f"SELECT ?item ?ancestor ?ancestorLabel ?depth WHERE {{\n"
            f"  VALUES ?item {{ {values} }}\n"
            f"  {{\n    {union}\n  }}\n"
            f'  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{language},en" . }}\n'
            f"}}"
        )

    def sparql_parents_batch(
        self,
        qids: List[str],
        property_id: str,
        max_depth: int = 3,
        language: str = "en",
    ) -> Dict[str, List[Tuple[str, str]]]:
        all_results: Dict[str, List[Tuple[str, str]]] = {q: [] for q in qids}
        seen: Dict[str, set] = {q: set() for q in qids}

        for start in range(0, len(qids), SPARQL_BATCH_SIZE):
            self._sparql_throttle()
            chunk = _sanitize_qids(qids[start : start + SPARQL_BATCH_SIZE])
            if not chunk:
                continue
            query = self._build_parents_query(chunk, property_id, max_depth, language)
            try:
                data = self._sparql_post(query, timeout=DEFAULT_SPARQL_TIMEOUT)
            except Exception:
                logger.exception("sparql_parents_batch failed (offset %d)", start)
                continue
            for b in data.get("results", {}).get("bindings", []):
                item_uri = b.get("item", {}).get("value", "")
                anc_uri = b.get("ancestor", {}).get("value", "")
                anc_label = b.get("ancestorLabel", {}).get("value", "")
                item_qid = item_uri.rsplit("/", 1)[-1] if item_uri else ""
                anc_qid = anc_uri.rsplit("/", 1)[-1] if anc_uri else ""
                if item_qid in all_results and anc_qid and anc_qid not in seen[item_qid]:
                    seen[item_qid].add(anc_qid)
                    all_results[item_qid].append((anc_qid, anc_label))
        return all_results
