"""lookup_external operator — two-mode external data lookup.

Mode A (code): the model emits a fenced Python block that calls
`fetch_fred(series_id, date)` / `fetch_bls(series_id, date)`; we exec
in-process with those helpers in scope and read back `result`.

Mode B (search): the model emits a bare JSON `{value, unit, source}`
object, grounded against Google Search. `source` is parsed-and-discarded;
Gemini's `grounding_titles` already constrain which domain the answer
came from, so we don't re-validate the token.

On any failure in mode A (network, no data, unknown series, code-parse
error) the operator catches the exception and re-issues the call once
with the prior failure appended to the user message, forcing mode B.
This makes the API tier best-effort and search the safety net.
"""

from __future__ import annotations

import json
import os
import re
import ssl
import urllib.parse
import urllib.request
from datetime import date as _date

from pydantic import BaseModel, ConfigDict, ValidationError

from skunk.common import strip_code_fence
from skunk.errors import MissingData, StepFailed
from skunk.prompted_call import PromptedCall
from skunk.models import AnnotatedValue, HarnessContext
from skunk.plan import LookupBranch
from skunk.pyexec import exec_python_capture_stdout


# Shared SSL context for HTTPS calls to FRED/BLS. Uses the system trust
# store; certifi is not strictly required but the deleted code used it
# and most production Python envs ship with it.
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


# Accepted date forms for fetch_fred / fetch_bls. Explicit parsing — no
# `len(date_str)` heuristic — so a malformed date raises ValueError
# instead of silently dispatching to the wrong branch.
_DATE_YEAR_RE = re.compile(r"^\d{4}$")
_DATE_YEAR_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def _parse_date(date_str: str) -> tuple[str, int, int | None, int | None]:
    """Return (kind, year, month, day). kind ∈ {"Y","YM","YMD"}.

    Raises ValueError on any other form. `month` / `day` are None when
    unused at the given granularity."""
    if _DATE_YEAR_RE.match(date_str):
        return ("Y", int(date_str), None, None)
    if _DATE_YEAR_MONTH_RE.match(date_str):
        return ("YM", int(date_str[:4]), int(date_str[5:7]), None)
    # Strict ISO date: this raises ValueError on bad input.
    d = _date.fromisoformat(date_str)
    return ("YMD", d.year, d.month, d.day)


def fetch_fred(series_id: str, date_str: str) -> float:
    """FRED REST API. `date_str` is "YYYY", "YYYY-MM", or "YYYY-MM-DD".

    For YYYY: returns the arithmetic mean of all observations within
    that calendar year (works for daily, weekly, monthly, etc.).
    For YYYY-MM: returns the mean of all observations within that
    month (one observation for monthly series, ~20 for daily series).
    For YYYY-MM-DD: returns the observation on that exact date, or
    the nearest observation within 5 days (handles weekends/holidays
    for daily series).
    Raises RuntimeError on no data / network failure / missing key."""
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set")

    kind, year, month, day = _parse_date(date_str)
    if kind == "Y":
        start, end = f"{year:04d}-01-01", f"{year:04d}-12-31"
    elif kind == "YM":
        # Span the full month so daily series produce 20+ observations,
        # not zero. Use the 28th as a lower bound for end-of-month —
        # avoids leap-year and short-month edge cases without a date
        # library round-trip; FRED tolerates an end > actual EOM.
        start = f"{year:04d}-{month:02d}-01"
        end = f"{year:04d}-{month:02d}-{_last_day_of_month(year, month):02d}"
    else:
        # Single-day request: pad by ±5 days so a weekend/holiday lookup
        # still finds the surrounding trading-day observation. Series
        # that report monthly will return at most one obs in the window.
        d = _date(year, month, day)
        from datetime import timedelta
        start = (d - timedelta(days=5)).isoformat()
        end = (d + timedelta(days=5)).isoformat()

    url = (
        "https://api.stlouisfed.org/fred/series/observations?"
        + urllib.parse.urlencode({
            "series_id": series_id,
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
            "api_key": api_key,
            "file_type": "json",
        })
    )
    with urllib.request.urlopen(url, timeout=10, context=_SSL_CTX) as r:
        observations = json.loads(r.read()).get("observations", [])
    values = [(o["date"], float(o["value"])) for o in observations
              if o.get("value", ".") != "."]
    if not values:
        raise RuntimeError(f"FRED returned no observations for {series_id} {date_str}")
    if kind == "YMD":
        # Closest-by-date observation to the requested day.
        target = _date(year, month, day)
        best = min(values, key=lambda dv: abs((_date.fromisoformat(dv[0]) - target).days))
        return best[1]
    # YYYY or YYYY-MM: mean of in-window observations.
    return sum(v for _, v in values) / len(values)


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    from datetime import timedelta
    next_month_first = _date(year, month + 1, 1)
    return (next_month_first - timedelta(days=1)).day


def fetch_bls(series_id: str, date_str: str) -> float:
    """BLS public REST API. `date_str` is "YYYY", "YYYY-MM", or
    "YYYY-MM-DD" (day component ignored — BLS series are monthly or
    annual).

    For YYYY: returns M13 (annual aggregate) if the series reports it;
    else the arithmetic mean of M01..M12. Raises if the returned
    periods are not monthly (e.g. quarterly Q01..Q04).
    For YYYY-MM(-DD): returns that month's value, or raises if the
    series does not have a period code for that month.
    Raises RuntimeError on any of: bad input, no data, non-monthly
    series, network failure."""
    kind, year, month, _ = _parse_date(date_str)

    params = {"startyear": str(year), "endyear": str(year)}
    api_key = os.environ.get("BLS_API_KEY", "")
    if api_key:
        params["registrationkey"] = api_key
    url = (
        f"https://api.bls.gov/publicAPI/v2/timeseries/data/{series_id}?"
        + urllib.parse.urlencode(params)
    )
    with urllib.request.urlopen(url, timeout=15, context=_SSL_CTX) as r:
        payload = json.loads(r.read())
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS error: {payload.get('status')} / {payload.get('message')}")
    series = payload.get("Results", {}).get("series", [])
    if not series or not series[0].get("data"):
        raise RuntimeError(f"BLS returned no data for {series_id} {date_str}")
    data = series[0]["data"]
    periods = {d["period"] for d in data}

    if kind == "Y":
        # Prefer M13 (annual aggregate) if the series provides one;
        # else mean of M01..M12. Refuse to reduce if the series is not
        # monthly — anything outside M01..M13 means the assumption
        # behind a mean is wrong, and we'd rather fall back to search
        # than silently average the wrong thing.
        non_monthly = periods - {f"M{i:02d}" for i in range(1, 14)}
        if non_monthly:
            raise RuntimeError(
                f"BLS series {series_id} is not monthly "
                f"(periods include {sorted(non_monthly)}); cannot reduce to annual mean"
            )
        for d in data:
            if d["period"] == "M13":
                return float(d["value"])
        monthly = [float(d["value"]) for d in data if d["period"] != "M13"]
        if not monthly:
            raise RuntimeError(f"BLS no monthly data for {series_id} {year}")
        return sum(monthly) / len(monthly)

    # Monthly lookup: only valid if the series actually carries that
    # month's period code. Quarterly / semi-annual / annual-only series
    # surface a clear error rather than a missing-data raise.
    target = f"M{month:02d}"
    if target not in periods:
        raise RuntimeError(
            f"BLS series {series_id} has no period {target} for {year} "
            f"(returned periods: {sorted(periods)})"
        )
    for d in data:
        if d["period"] == target:
            return float(d["value"])
    raise RuntimeError(f"BLS no observation for {series_id} {year}-{target}")


class LookupResult(BaseModel):
    """One Gemini lookup reply (mode B). `value` is the publisher's
    printed figure (number, entity string, or list for multi-value
    requests), or `None` when the model could not find a value;
    `unit` is a short label; `source` is the publisher token
    (parsed-and-discarded at the call site)."""

    model_config = ConfigDict(frozen=True)
    value: float | int | str | list[float | int | str] | None
    unit: str = ""
    source: str = "none"


# TODO: need to move dataset/setup specific calls into the prompt override system
class LookupExternalPromptedCall(PromptedCall):
    name: str = "lookup_external"
    system_prompt: str = """\
You find data points from a reliable external source. You must
ground every answer in real data — never invent values.

## Input

The user message is a single bare JSON object:

  {"target": "<natural-language request for a single value>",
   "src":    "<natural-language description of required source | null>"}

`target` is a natural-language request for a single value (or a small
list of values in a fixed order). `src` is an optional hint about the
required publisher.

## Output modes

Two modes are available. Pick exactly one per response.

### Mode A — direct API fetch

Use mode A only if BOTH of the following hold:
  (i) the requested value is available from one of the configured
      data sources (currently FRED and BLS), AND
  (ii) `src` does NOT name a different specific source.

For (ii): `src` values that name the Federal Reserve, a regional
Federal Reserve Bank (e.g. "St. Louis Fed", "FRBSF", "Minneapolis
Fed"), or FRED/FRASER ARE FRED-compatible — prefer mode A with FRED
for those. They are the same publisher, not a substitute aggregator.

If either condition fails, use mode B.

Emit a single fenced ```python``` block AND NOTHING ELSE — no JSON,
no prose.

**The following are already in scope. Do NOT write any `import` or
`from … import …` statements — they will fail with `ModuleNotFoundError`
or `ImportError`. Just use the names directly.**

  fetch_fred(series_id: str, date: str) -> float
      FRED REST API. `date` is "YYYY" (annual mean of observations),
      "YYYY-MM" (that month's value), or "YYYY-MM-DD" (that day's
      value). Raises on no-data or network failure.

  fetch_bls(series_id: str, date: str) -> float
      BLS public REST API. Same date format. Returns the M13 annual
      aggregate if the series reports one, else the mean of M01..M12
      for a YYYY date. Raises if the series is not monthly.

Also already in scope (no imports needed):
  math, statistics, datetime, numpy (as np), pandas (as pd).

**Every numeric value used in your code must come from a
`fetch_fred(...)` or `fetch_bls(...)` call. Never hardcode numbers
from training memory, training-data estimates, or "well-known"
constants — if you cannot satisfy this rule, use mode B instead.**

Use `print()` to emit the value(s) you fetched. After your code
runs, the harness will exec it, capture stdout, then show you the
program text and stdout and ask you to emit the final JSON in a
follow-up turn.

### Mode B — search

Emit a single bare JSON object. No markdown fences. No prose.

  {"value":  <number | string | array of those>,
   "unit":   "<string>",
   "source": "<string>"}

Field semantics:
  value     publisher's full printed precision. JSON number for
            numeric answers, JSON string for entity names, JSON array
            for multiple values in request order. Return null if you
            are unable to find such a value.
  unit      natural-language scale/base label, e.g. "millions of
            dollars", "percent", "year". Empty string ("") for
            non-measurement answers.
  source    short token naming the publisher domain, e.g. "bls",
            "bankofengland".

If `src` is present, search that site (use a `site:` filter or
include the publisher name). Do not substitute a different aggregator
even if you know it carries the same series. Issue the search via
your Google Search tool — answering from training memory alone is not
allowed. Read the snippets and extract the value at the publisher's
full printed precision. Never round, format, or simplify.
{{ default_tail }}"""

    def run(self, ctx: HarnessContext, branch: LookupBranch) -> list[AnnotatedValue]:
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True
        )

        ctx.emit(self.name, "calling gemini", target=branch.target, src=branch.src)
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx),
            user_msg,
            effort="off",
            use_google_search=True,
            ctx=ctx,
        )
        ctx.emit(
            self.name, "gemini response",
            raw=resp.text, grounding_titles=resp.grounding_titles,
        )

        # Mode dispatch: detect Python code mode from the raw response
        # (before strip_code_fence eats the language tag). Any fenced
        # python block, or any text mentioning the helper functions,
        # routes to code mode — let exec + the second-call JSON
        # synthesizer handle whatever the model actually wrote.
        raw_stripped = resp.text.strip()
        is_code = (
            raw_stripped.startswith("```python")
            or "fetch_fred(" in raw_stripped
            or "fetch_bls(" in raw_stripped
        )

        if is_code:
            try:
                return self._run_code_mode(ctx, branch, raw_stripped)
            except Exception as e:
                ctx.emit(self.name, "code mode failed; retrying",
                         error=f"{type(e).__name__}: {e}")
                return self._run_retry(
                    ctx, branch,
                    prior_response=raw_stripped, prior_error=str(e),
                )

        return self._parse_search_response(ctx, branch, resp, strip_code_fence(resp.text))

    def _run_code_mode(
        self, ctx: HarnessContext, branch: LookupBranch, raw_code_response: str,
    ) -> list[AnnotatedValue]:
        # Strip fences once so we can both exec the body and show the
        # same stripped body back to the model in the synth message.
        from skunk.pyexec import strip_code_fences
        code = strip_code_fences(raw_code_response)

        # Exec with stdout capture. Helpers live in the env namespace;
        # no `result` variable required — the model is told to print().
        # pyexec pre-injects the standard sandbox env (math, np, pd,
        # statistics, datetime, ...); we add fetch_fred / fetch_bls.
        try:
            _env, stdout = exec_python_capture_stdout(
                code,
                local_vars={"fetch_fred": fetch_fred, "fetch_bls": fetch_bls},
            )
        except Exception as e:
            raise RuntimeError(
                f"code exec failed: {type(e).__name__}: {e}"
            ) from e
        ctx.emit(self.name, "code mode exec done", stdout=stdout)

        # Second call: feed the program + its stdout back to the model
        # and ask for the final JSON. Reuse the same system prompt
        # (which documents the mode B JSON shape) and effort=off — this
        # is pure transcription, not reasoning.
        synth_user_msg = (
            "You previously emitted the following Python code in mode A "
            "for this lookup:\n\n"
            "```python\n"
            f"{code}\n"
            "```\n\n"
            "It was executed in a sandbox:"
            "Captured stdout:\n\n"
            "```\n"
            f"{stdout}\n"
            "```\n\n"
            "Original request:\n"
            f"{branch.model_dump_json(include={'target', 'src'}, indent=2, exclude_none=True)}\n\n"
            "Now emit the final JSON object — the same `{value, unit, source}` "
            "schema documented in mode B. `value` is the fetched value (a number "
            "or a list of numbers in the requested order). `source` is the "
            "publisher domain you fetched from (e.g. \"fred\", \"bls\"). "
            "Bare JSON. No markdown fences. No prose."
        )
        synth_resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx),
            synth_user_msg,
            effort="off",
            ctx=ctx,
        )
        ctx.emit(self.name, "code mode JSON synthesis", raw=synth_resp.text)
        cleaned = strip_code_fence(synth_resp.text)
        try:
            result = LookupResult.model_validate_json(cleaned)
        except ValidationError as e:
            raise RuntimeError(
                f"code-mode JSON synthesis failed: {e}\nRaw: {synth_resp.text}"
            ) from e

        v = result.value
        if v is None or (isinstance(v, (str, list)) and len(v) == 0):
            raise MissingData(
                f"lookup_external (code mode) returned no value for target: {branch.target}",
                missing=[branch.target],
            )
        ctx.emit(self.name, "code mode parsed", value=v, unit=result.unit)
        return [AnnotatedValue(description=branch.target, value=v, unit=result.unit)]

    def _run_retry(
        self, ctx: HarnessContext, branch: LookupBranch,
        prior_response: str, prior_error: str,
    ) -> list[AnnotatedValue]:
        # Re-issue the lookup with the model's prior response and the
        # error it produced. The model may fix the Python (mode A again)
        # or switch to mode B — its choice. Single retry only — no loop.
        # If the retry also fails, surface as StepFailed.
        user_msg = branch.model_dump_json(
            include={"target", "src"}, indent=2, exclude_none=True,
        )
        user_msg += (
            "\n\nYour prior response failed.\n\n"
            "--- prior response ---\n"
            f"{prior_response}\n"
            "--- error ---\n"
            f"{prior_error}\n"
            "--- end ---\n\n"
            "Either: (a) fix the Python and resubmit in mode A, or "
            "(b) switch to mode B (search + bare JSON) if the failure leads you to believe your initial pick was wrong."
        )
        ctx.emit(self.name, "calling gemini (retry)",
                 target=branch.target, src=branch.src,
                 prior_error=prior_error)
        resp = ctx.llm_client.call(
            self.assemble_system_prompt(ctx),
            user_msg,
            effort="off",
            use_google_search=True,
            ctx=ctx,
        )
        ctx.emit(
            self.name, "gemini response (retry)",
            raw=resp.text, grounding_titles=resp.grounding_titles,
        )

        # Same mode-dispatch as run(), but no further retry on failure.
        raw_stripped = resp.text.strip()
        is_code = (
            raw_stripped.startswith("```python")
            or "fetch_fred(" in raw_stripped
            or "fetch_bls(" in raw_stripped
        )
        if is_code:
            try:
                return self._run_code_mode(ctx, branch, raw_stripped)
            except Exception as e:
                raise StepFailed(
                    self.name,
                    f"retry code-mode also failed: {type(e).__name__}: {e}",
                ) from e
        return self._parse_search_response(ctx, branch, resp, strip_code_fence(resp.text))

    def _parse_search_response(
        self, ctx: HarnessContext, branch: LookupBranch, resp, cleaned: str,
    ) -> list[AnnotatedValue]:
        try:
            result = LookupResult.model_validate_json(cleaned)
        except ValidationError as e:
            raise StepFailed(
                self.name,
                f"Cannot parse response: {e}\nRaw: {resp.text}",
            ) from e

        # System prompt requires search-mode answers to be grounded. Accept
        # any of: chunk titles, chunk URIs, or web_search_queries. Vertex AI
        # often returns only `web_search_queries` even when search clearly
        # ran (AI Studio populated `grounding_chunks` more reliably).
        if not (resp.grounding_titles or resp.grounding_urls or resp.search_queries):
            raise StepFailed(
                self.name,
                f"Ungrounded reply (no grounding metadata) for target: {branch.target}",
            )

        v = result.value
        if v is None or (isinstance(v, (str, list)) and len(v) == 0):
            raise MissingData(
                f"lookup_external found no value for target: {branch.target}",
                missing=[branch.target],
            )

        ctx.emit(self.name, "parsed", value=result.value, unit=result.unit)
        return [
            AnnotatedValue(
                description=branch.target, value=result.value, unit=result.unit
            )
        ]
