"""Lightweight query analysis: which fiscal years / doc type does the question refer to?

NVIDIA's fiscal year ends in late January, so "fiscal 2024" = year ended Jan 28, 2024. Users often
write a bare "2024"; we treat that as a fiscal year *and* pull a few passages from the next year's
report (which covers calendar 2024 in its comparatives).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FY_RE = re.compile(r"\b(?:fiscal(?:\s+year)?|fy)\s*'?\s*((?:19|20)?\d{2})\b", re.I)
RANGE_RE = re.compile(r"(?<![\d,$.])((?:19|20)\d{2})\s*(?:-|\u2013|to|through)\s*((?:19|20)\d{2})(?!\d)", re.I)
YEAR_RE = re.compile(r"(?<![\d,$.])(19[89]\d|20[0-3]\d)(?![\d%]|,\d)")
PROXY_HINT = re.compile(
    r"\b(proxy|say-on-pay|executive compensation|compensation committee|director nominees?|nominees?|"
    r"annual meeting|stockholder proposals?|pay ratio|beneficial ownership|independent directors?|"
    r"board of directors|summary compensation)\b",
    re.I,
)
MAX_YEARS = 8


@dataclass
class QueryInfo:
    years: list[int] = field(default_factory=list)
    plain: set[int] = field(default_factory=set)  # years written without "fiscal"/"FY"
    doc_type: str | None = None


def _norm_year(s: str) -> int:
    if len(s) == 4:
        return int(s)
    n = int(s)
    return 1900 + n if n >= 90 else 2000 + n


def analyze_query(q: str) -> QueryInfo:
    years: set[int] = set()
    plain: set[int] = set()
    rest = q

    for m in RANGE_RE.finditer(q):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b and b - a < MAX_YEARS:
            years.update(range(a, b + 1))
    rest = RANGE_RE.sub(" ", rest)

    for m in FY_RE.finditer(rest):
        years.add(_norm_year(m.group(1)))
    rest = FY_RE.sub(" ", rest)

    for m in YEAR_RE.finditer(rest):
        y = int(m.group(1))
        if y not in years:
            plain.add(y)
        years.add(y)

    ys = sorted(years)[:MAX_YEARS]
    return QueryInfo(
        years=ys,
        plain={y for y in plain if y in ys},
        doc_type="proxy" if PROXY_HINT.search(q) else None,
    )
