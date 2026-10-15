"""Command-line chatbot:  python cli.py "question"   |   python cli.py   (interactive)"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nvrag.pipeline import RagPipeline  # noqa: E402


def show(a) -> None:
    print("\n" + a.answer)
    for w in a.warnings:
        print(f"  [warning] {w}")
    if a.cited_ids:
        print("\nSources:")
        for n in a.cited_ids:
            s = a.sources[n - 1]
            print(f"  [S{n}] {s['file']} - PDF page {s['page']} ({s['period']}, {s['kind']})")


def main() -> None:
    pipe = RagPipeline()
    if len(sys.argv) > 1:
        show(pipe.ask(" ".join(sys.argv[1:])))
        return
    history: list[dict] = []
    while True:
        try:
            q = input("\nask> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        a = pipe.ask(q, history)
        show(a)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": a.answer}]


if __name__ == "__main__":
    main()
