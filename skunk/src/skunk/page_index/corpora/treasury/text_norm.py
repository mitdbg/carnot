"""Deterministic text normalization for Treasury Bulletin section labels.

OCR-substitution table, year-suffix / roman-numeral / continuation
stripping, and the final whitespace-collapsing `normalize_label`
entrypoint. Pure deterministic, no LLM. Used by the Treasury placer and
merger for cleaning labels before matching / clustering.

The substitution table grew empirically from OCR errors observed across
the 1939-2025 corpus. Keep entries focused on whole-word patterns where
the replacement is unambiguous.
"""

from __future__ import annotations

import re


# Letter-substitution + missing-whitespace + truncation OCR errors seen in
# the parsed-JSON output on scanned mid-century bulletins.
_OCR_SUBS: list[tuple[re.Pattern, str]] = [
    # Letter substitutions that turn DEBT/DATA/etc. into garbage.
    (re.compile(r"\bOEBT\b", re.IGNORECASE), "DEBT"),
    (re.compile(r"\bOATA\b", re.IGNORECASE), "DATA"),
    (re.compile(r"\bYIELOS\b", re.IGNORECASE), "YIELDS"),
    (re.compile(r"\bBONOS\b", re.IGNORECASE), "BONDS"),
    (re.compile(r"\bPISCAL\b", re.IGNORECASE), "FISCAL"),
    (re.compile(r"\bFEERAL\b", re.IGNORECASE), "FEDERAL"),
    (re.compile(r"\bACOUNT\b", re.IGNORECASE), "ACCOUNT"),
    (re.compile(r"\bFUNOS\b", re.IGNORECASE), "FUNDS"),
    (re.compile(r"\bFUNO\b", re.IGNORECASE), "FUND"),
    (re.compile(r"\bUNITEO\b", re.IGNORECASE), "UNITED"),
    (re.compile(r"\bSAV INGS\b", re.IGNORECASE), "SAVINGS"),
    (re.compile(r"\bRE PORT\b", re.IGNORECASE), "REPORT"),
    (re.compile(r"\bFIS CAL\b", re.IGNORECASE), "FISCAL"),
    # Missing-whitespace concatenations.
    (re.compile(r"\bTRUSTFUNDS\b", re.IGNORECASE), "TRUST FUNDS"),
    (re.compile(r"\bFOREIGNCURRENCYPOSITIONS\b", re.IGNORECASE),
     "FOREIGN CURRENCY POSITIONS"),
    (re.compile(r"\bPUBLICDEBTOPERATIONS\b", re.IGNORECASE),
     "PUBLIC DEBT OPERATIONS"),
    (re.compile(r"\bEXCHANGESTABILIZATION\b", re.IGNORECASE),
     "EXCHANGE STABILIZATION"),
    (re.compile(r"U\.S\.TREASURY", re.IGNORECASE), "U.S. TREASURY"),
    (re.compile(r"\bD\.S\.", re.IGNORECASE), "U.S."),
    # Truncation artifacts.
    (re.compile(r"\bCAPAL\b", re.IGNORECASE), "CAPITAL"),
    (re.compile(r"\bCAPITAL MOVEM\b", re.IGNORECASE), "CAPITAL MOVEMENTS"),
    (re.compile(r"\bMOVEM\b", re.IGNORECASE), "MOVEMENTS"),
]

# Year-marker suffixes ("FISCAL YEAR 1988 (PROTOTYPE)", "AS OF SEPT. 30, 1984.")
_YEAR_SUFFIX_RE = re.compile(
    r",?\s*FISCAL(?:\s+YEAR)?\s+\d{4}\s*\(?(?:PROTOTYPE|EXCERPT|EXTRACT|EXCERPTED)?\)?\.?$",
    re.IGNORECASE,
)
_AS_OF_DATE_RE = re.compile(
    r"\s+AS\s+OF\s+\w+\.?\s+\d{1,2},?\s+\d{4}\.?$",
    re.IGNORECASE,
)
# Roman-numeral / leading-decimal prefixes: "V. ", "VI. ", "II. ", "1. ".
_PREFIX_NUMERAL_RE = re.compile(r"^\s*(?:[IVX]+|\d+)\.\s+")
# Continuation suffix ", con", "(continued)", etc.
_CONT_SUFFIX_RE = re.compile(
    r"[,\s\-–—]+(?:con|cont|continued)\s*\.?\s*\)?\s*$", re.IGNORECASE,
)
# Trailing punctuation / whitespace.
_TRAILING_PUNCT_RE = re.compile(r"[\s\.,;:\-–—]+$")


def normalize_label(raw: str) -> str:
    """Deterministic normalization for a raw section label.

    Strips OCR errors, year suffixes, roman-numeral prefixes, continuation
    suffixes, and trailing punctuation. Collapses whitespace. Iterates to
    a fixed point (up to 4 passes) because labels often stack multiple
    issues. Returns the cleaned string in its original case so downstream
    callers can preserve display form via their own lowercased keying.
    """
    s = raw.strip()
    for _ in range(4):
        prev = s
        for pattern, repl in _OCR_SUBS:
            s = pattern.sub(repl, s)
        s = _YEAR_SUFFIX_RE.sub("", s).strip()
        s = _AS_OF_DATE_RE.sub("", s).strip()
        s = _PREFIX_NUMERAL_RE.sub("", s).strip()
        s = _CONT_SUFFIX_RE.sub("", s).strip()
        s = _TRAILING_PUNCT_RE.sub("", s).strip()
        if s == prev:
            break
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_key(s: str) -> str:
    """Lowercased grouping key for case-insensitive matching."""
    return s.lower().strip()
