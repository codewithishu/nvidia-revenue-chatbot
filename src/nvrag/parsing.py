"""PDF -> per-page prose blocks + tables (as markdown), using PyMuPDF.

Tables matter for financial filings, so we:
  * detect tables with PyMuPDF's find_tables(),
  * normalise split cells ("$" / ")" / "%" columns) into clean markdown rows,
  * remove text blocks that fall inside a table bbox (avoid duplicate / garbled text),
  * capture the caption/heading just above the table and any "(in millions...)" unit hint.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .textutil import clean

UNITS_RE = re.compile(
    r"\(?\s*(?:in|amounts in|dollars in|\$ in|U\.S\. dollars in)\s+(millions|thousands|billions)"
    r"(?:\s*,?\s*except[^)\n]{0,80})?\s*\)?",
    re.I,
)


@dataclass
class Table:
    bbox: tuple
    markdown: str
    caption: str = ""


@dataclass
class PageContent:
    page: int  # 1-based physical PDF page index
    blocks: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    units: str | None = None
    ocr: bool = False
    printed_page: str | None = None


def detect_units(text: str) -> str | None:
    m = UNITS_RE.search(text)
    return f"in {m.group(1).lower()}" if m else None


def _clean_cell(c) -> str:
    return re.sub(r"\s+", " ", (c or "")).strip()


def rows_to_markdown(rows: list[list]) -> str | None:
    """Turn raw extracted rows into a compact markdown table (or None if not table-like)."""
    grid = [[_clean_cell(c) for c in r] for r in rows if r]
    if len(grid) < 2:
        return None
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]

    # Fold ")" and "%" only-columns into the previous column; drop "$"-only / empty columns.
    j = width - 1
    while j > 0:
        vals = {r[j] for r in grid if r[j]}
        if vals and vals <= {")", "%", ")%"}:
            for r in grid:
                if r[j]:
                    r[j - 1] = (r[j - 1] + r[j]).strip()
            for r in grid:
                del r[j]
        j -= 1
    keep = [j for j in range(len(grid[0])) if {r[j] for r in grid if r[j]} - {"$", ""}]
    grid = [[r[j] for j in keep] for r in grid]
    grid = [r for r in grid if any(r)]
    if len(grid) < 2 or not grid[0] or len(grid[0]) < 2:
        return None

    esc = lambda s: s.replace("|", "/")
    lines = ["| " + " | ".join(esc(c) for c in grid[0]) + " |", "|" + "---|" * len(grid[0])]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in grid[1:]]
    return "\n".join(lines)


def _caption_above(page: "fitz.Page", bbox: tuple) -> str:
    clip = fitz.Rect(page.rect.x0, max(page.rect.y0, bbox[1] - 55), page.rect.x1, bbox[1])
    txt = clean(page.get_text("text", clip=clip))
    return re.sub(r"\s+", " ", txt)[-220:]


def _printed_page_number(page: "fitz.Page") -> str | None:
    """Best-effort: a bare number in the bottom 8% of the page (the page number printed on it)."""
    h = page.rect.height
    clip = fitz.Rect(page.rect.x0, h * 0.92, page.rect.x1, h)
    for tok in reversed(page.get_text("text", clip=clip).split()):
        if re.fullmatch(r"\d{1,3}", tok):
            return tok
    return None


def parse_pdf(path: Path, ocr: bool = False) -> list[PageContent]:
    doc = fitz.open(path)
    pages: list[PageContent] = []
    for i, page in enumerate(doc, start=1):
        tables: list[Table] = []
        boxes: list["fitz.Rect"] = []
        try:
            for t in page.find_tables().tables:
                md = rows_to_markdown(t.extract())
                if md:
                    bbox = tuple(t.bbox)
                    tables.append(Table(bbox, md, _caption_above(page, bbox)))
                    boxes.append(fitz.Rect(bbox))
        except Exception:
            pass  # table detection is best-effort; prose extraction below still works

        blocks: list[str] = []
        for b in page.get_text("blocks", sort=True):
            if b[6] != 0:  # image block
                continue
            r = fitz.Rect(b[:4])
            centre = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            if any(centre in bb for bb in boxes):
                continue
            txt = clean(b[4])
            if txt:
                blocks.append(txt)

        used_ocr = False
        if ocr and sum(map(len, blocks)) + sum(len(t.markdown) for t in tables) < 40:
            try:  # scanned page: needs Tesseract installed
                tp = page.get_textpage_ocr(dpi=200, full=True)
                text = page.get_text("text", textpage=tp, sort=True)
                blocks = [clean(p) for p in re.split(r"\n\s*\n", text) if clean(p)]
                used_ocr = True
            except Exception:
                pass

        probe = " ".join(blocks + [t.caption for t in tables])
        pages.append(
            PageContent(
                page=i,
                blocks=blocks,
                tables=tables,
                units=detect_units(probe),
                ocr=used_ocr,
                printed_page=_printed_page_number(page),
            )
        )
    doc.close()
    return pages
