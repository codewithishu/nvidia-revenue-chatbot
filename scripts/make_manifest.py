"""Create/extend data/manifest.csv for PDFs whose type/year can't be read from the file name."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fitz  # noqa: E402

from nvrag.docmeta import FY_END_RE, MEETING_RE, infer_meta  # noqa: E402

RAW = Path("data/raw")
MANIFEST = Path("data/manifest.csv")
ACCESSION_RE = re.compile(r"-(\d{6,7})-(\d{2})-(\d+)\.pdf$", re.I)


def accession_year(name: str) -> int | None:
    m = ACCESSION_RE.search(name)
    if not m:
        return None
    yy = int(m.group(2))
    return 1900 + yy if yy >= 90 else 2000 + yy


def guess_type(text: str) -> str:
    low = " ".join(text.split()).lower()
    head = low[:3000]
    if "form 10-k" in head or "annual report pursuant to section 13" in head or "form 10-k405" in head:
        return "annual_report"
    if "schedule 14a" in head or "proxy statement" in head or "notice of annual meeting" in head:
        return "proxy"
    if "form 10-q" in head or "quarterly report pursuant" in head or "form 8-k" in head or "current report pursuant" in head:
        return "skip"
    if "form 10-k" in low[:12000]:
        return "annual_report"
    if "proxy statement" in low[:12000]:
        return "proxy"
    return "unknown"


def main() -> None:
    existing: dict[str, dict] = {}
    if MANIFEST.exists():
        with open(MANIFEST, newline="", encoding="utf-8") as f:
            existing = {r["file"]: r for r in csv.DictReader(f)}

    rows = dict(existing)
    report = []
    for p in sorted(RAW.glob("*.pdf")):
        if p.name in existing:
            continue
        meta = infer_meta(p, "")  # filename only
        if meta.doc_type != "unknown" and meta.year:
            continue  # name is already good enough
        try:
            doc = fitz.open(p)
            text = " ".join(doc[i].get_text("text") for i in range(min(3, len(doc))))
            doc.close()
        except Exception as e:
            report.append((p.name, "ERROR", "", str(e)))
            continue
        flat = " ".join(text.split())
        typ = guess_type(text)
        year, how = None, ""
        if typ in ("annual_report", "proxy"):
            m = (FY_END_RE if typ == "annual_report" else MEETING_RE).search(flat)
            if m:
                year, how = int(m.group(1)), "text"
        acc = accession_year(p.name)
        if year is None and acc:
            year, how = acc, "filename-accession"
        note = how
        if year and acc and year != acc:
            note += f" (filing-year in name says {acc} - check)"
        if not text.strip():
            note = "NO TEXT - scanned? needs OCR; set type/year by hand"
        rows[p.name] = {"file": p.name, "doc_type": typ, "year": year or ""}
        report.append((p.name, typ, year or "", note))

    MANIFEST.parent.mkdir(exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "doc_type", "year"])
        w.writeheader()
        for r in rows.values():
            w.writerow({k: r.get(k, "") for k in ("file", "doc_type", "year")})

    print(f"{'file':<45} {'type':<14} {'year':<6} note")
    for name, typ, yr, note in report:
        print(f"{name:<45} {typ:<14} {str(yr):<6} {note}")
    todo = [r[0] for r in report if r[1] in ("unknown", "ERROR") or not r[2] and r[1] != "skip"]
    print(f"\nWrote {MANIFEST} ({len(rows)} rows). Files needing manual attention: {len(todo)}")
    for t in todo:
        print("  -", t)
    print("Rows with type 'skip' (10-Q, 8-K, ...) will be left out of the index.")


if __name__ == "__main__":
    main()