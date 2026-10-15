"""Structure-aware chunking.

* Chunks never cross page boundaries -> every citation maps to exactly one PDF page.
* Prose is packed paragraph-by-paragraph up to ~target chars; headings start a new chunk.
* Each table becomes its own chunk (markdown + caption + units); very large tables are split
  by rows with the header row repeated, so numbers always sit next to their column labels.
* Every chunk carries a context header (doc type, period, page, section) that is also embedded,
  so "FY2023 data center revenue" style queries can match on period.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

from .config import Config
from .docmeta import DocMeta
from .parsing import PageContent

HEAD_RE = re.compile(r"^((?i:item\s+\d+[a-c]?\.?.{0,80})|[A-Z][A-Z0-9 ,&'\-:()/]{4,80})$")
DOC_LABEL = {"annual_report": "NVIDIA Annual Report/10-K", "proxy": "NVIDIA Proxy Statement"}


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file: str
    doc_type: str
    year: int | None
    page: int
    kind: str  # text | table
    section: str
    units: str | None
    text: str
    printed_page: str | None = None

    def period(self) -> str:
        if self.year is None:
            return "unknown period"
        return f"FY{self.year}" if self.doc_type == "annual_report" else f"{self.year} proxy"

    def header(self) -> str:
        label = DOC_LABEL.get(self.doc_type, "NVIDIA document")
        parts = [label, self.period(), f"page {self.page}", self.kind]
        if self.units:
            parts.append(f"units: {self.units}")
        return " | ".join(parts)

    def embed_text(self) -> str:
        sec = f"\nSection: {self.section}" if self.section else ""
        return f"{self.header()}{sec}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def detect_heading(block: str) -> str | None:
    line = block.strip()
    if "\n" in line or len(line) > 90 or len(line) < 4 or line.isdigit():
        return None
    return line if HEAD_RE.match(line) else None


def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def _mk(meta: DocMeta, pc: PageContent, kind: str, section: str, text: str, n: int) -> Chunk:
    return Chunk(
        chunk_id=f"{meta.doc_id}:p{pc.page}:{kind[0]}{n}",
        doc_id=meta.doc_id,
        file=meta.file,
        doc_type=meta.doc_type,
        year=meta.year,
        page=pc.page,
        kind=kind,
        section=section,
        units=pc.units,
        text=text,
        printed_page=pc.printed_page,
    )


def _split_table(md: str, caption: str, max_chars: int) -> list[str]:
    prefix = f"{caption}\n" if caption else ""
    if len(prefix) + len(md) <= max_chars:
        return [prefix + md]
    lines = md.split("\n")
    head, body = lines[:2], lines[2:]
    out, cur = [], []
    for ln in body:
        if cur and len(prefix) + sum(map(len, head + cur)) + len(ln) > max_chars:
            out.append(prefix + "\n".join(head + cur))
            cur = []
        cur.append(ln)
    if cur:
        out.append(prefix + "\n".join(head + cur))
    return out


def chunk_page(meta: DocMeta, pc: PageContent, section: str, cfg: Config) -> tuple[list[Chunk], str]:
    out: list[Chunk] = []
    buf: list[str] = []
    size = 0
    n = 0

    def flush():
        nonlocal buf, size, n
        text = "\n".join(buf).strip()
        if text:
            out.append(_mk(meta, pc, "text", section, text, n))
            n += 1
        buf, size = [], 0

    for blk in pc.blocks:
        h = detect_heading(blk)
        if h:
            flush()
            section = h
        for piece in split_long(blk, cfg.max_chunk_chars):
            if size + len(piece) > cfg.target_chunk_chars and buf:
                flush()
            buf.append(piece)
            size += len(piece) + 1
    flush()

    # merge a tiny trailing text chunk into its predecessor
    if len(out) >= 2 and len(out[-1].text) < 200:
        out[-2] = replace(out[-2], text=out[-2].text + "\n" + out[-1].text)
        out.pop()

    for t_i, t in enumerate(pc.tables):
        for part_i, part in enumerate(_split_table(t.markdown, t.caption, cfg.table_max_chars)):
            out.append(_mk(meta, pc, "table", section, part, t_i * 100 + part_i))
    return out, section


def make_chunks(meta: DocMeta, pages: list[PageContent], cfg: Config) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    for pc in pages:
        page_chunks, section = chunk_page(meta, pc, section, cfg)
        chunks.extend(page_chunks)
    return chunks
