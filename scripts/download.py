"""Download PDFs listed in data/sources.csv (columns: url,doc_type,year) into data/raw.

NVIDIA's Investor Relations archive (https://investor.nvidia.com -> Financial Info -> Annual Reports and
Proxies) lists every file. Right-click each PDF link, copy the URL, and add a row to data/sources.csv:

    url,doc_type,year
    https://.../NVIDIA-2024-Annual-Report.pdf,annual_report,2024
    https://.../NVIDIA-2024-Proxy-Statement.pdf,proxy,2024

Files are saved as nvidia_<doc_type>_<year>.pdf, which scripts/ingest.py understands. Alternatively
just download manually into data/raw/ (name them like above, or list them in data/manifest.csv).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import requests

SRC = Path("data/sources.csv")
OUT = Path("data/raw")


def main() -> None:
    if not SRC.exists():
        sys.exit(f"{SRC} not found - see the docstring of this script.")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(SRC, newline="") as f:
        for row in csv.DictReader(f):
            dest = OUT / f"nvidia_{row['doc_type'].strip()}_{row['year'].strip()}.pdf"
            if dest.exists():
                print(f"skip {dest.name} (exists)")
                continue
            try:
                r = requests.get(row["url"].strip(), timeout=60, headers={"User-Agent": "Mozilla/5.0 (research)"})
                r.raise_for_status()
                if not r.content.startswith(b"%PDF"):
                    print(f"!! {row['url']} is not a PDF (got {r.headers.get('content-type')})")
                    continue
                dest.write_bytes(r.content)
                print(f"saved {dest.name} ({len(r.content) // 1024} KB)")
            except Exception as e:
                print(f"!! {row['url']}: {e}")


if __name__ == "__main__":
    main()
