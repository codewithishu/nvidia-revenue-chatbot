"""Infer document type + year for each PDF.

Priority: manifest.csv override > filename > text on the first pages.

Year semantics (important for temporal accuracy):
  * annual_report -> NVIDIA *fiscal* year (FY ends late January; FY2024 ended Jan 28, 2024).
  * proxy         -> year of the annual meeting / filing (e.g. the 2024 proxy covers FY2024
                     results, but the meeting is held in fiscal 2025).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
FY_END_RE = re.compile(rf"fiscal\s+year\s+ended\s+(?:{MONTHS})\s+\d{{1,2}},?\s+((?:19|20)\d{{2}})", re.I)
MEETING_RE = re.compile(rf"annual\s+meeting[^.]{{0,250}}?(?:{MONTHS})\s+\d{{1,2}},?\s+((?:19|20)\d{{2}})", re.I | re.S)
FILE_YEAR_RE = re.compile(r"(?<!\d)(19[89]\d|20[0-3]\d)(?!\d)")


@dataclass
class DocMeta:
    doc_id: str
    file: str
    doc_type: str  # annual_report | proxy | unknown
    year: int | None
    year_source: str  # manifest | filename | text | unknown


def load_manifest(path: Path) -> dict[str, dict]:
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["file"].strip(): row for row in csv.DictReader(f)}


def _type_from_name(name: str) -> str | None:
    n = name.lower()
    if any(k in n for k in ("proxy", "def14a", "def-14a", "def_14a", "notice")):
        return "proxy"
    if any(k in n for k in ("10-k", "10k", "annual", "report", "review")):
        return "annual_report"
    return None


def _type_from_text(text: str) -> str | None:
    head = text[:6000].lower()
    if "proxy statement" in head or "notice of annual meeting" in head:
        return "proxy"
    if "form 10-k" in head or "annual report" in head:
        return "annual_report"
    return None


def infer_meta(path: Path, first_pages_text: str, manifest: dict[str, dict] | None = None) -> DocMeta:
    manifest = manifest or {}
    row = manifest.get(path.name)
    doc_id = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")

    doc_type = (row or {}).get("doc_type") or _type_from_name(path.name) or _type_from_text(first_pages_text) or "unknown"

    year, source = None, "unknown"
    if row and str(row.get("year", "")).strip().isdigit():
        year, source = int(row["year"]), "manifest"
    else:
        years = set(FILE_YEAR_RE.findall(path.stem))
        if len(years) == 1:
            year, source = int(years.pop()), "filename"
        else:
            rx = MEETING_RE if doc_type == "proxy" else FY_END_RE
            m = rx.search(first_pages_text)
            if m:
                year, source = int(m.group(1)), "text"
    return DocMeta(doc_id=doc_id, file=path.name, doc_type=doc_type, year=year, year_source=source)
