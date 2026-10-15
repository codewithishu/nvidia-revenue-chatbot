"""Evaluate retrieval + answers on a JSONL question set.

Usage:
  python scripts/evaluate.py --retrieval-only          # no LLM key needed
  python scripts/evaluate.py                           # full: retrieval + answer + abstention + citations
  python scripts/evaluate.py --file my_questions.jsonl

Question fields (JSONL):
  id, question, abstain (bool, default false),
  expect_years [int]     - every listed year must appear among retrieved chunks
  expect_doc_type        - annual_report | proxy
  expect_pages [int]     - optional: retrieved chunk must be on one of these PDF pages
  retrieval_keywords [s] - at least one must appear in a retrieved chunk (case-insensitive)
  answer_regex [s]       - every regex must match the answer (case-insensitive)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nvrag.config import load_config  # noqa: E402
from nvrag.pipeline import RagPipeline  # noqa: E402


def retrieval_rank(q: dict, hits) -> int | None:
    """Rank (1-based) at which all expectations are met; None if not met. Multi-year: worst year."""
    years = q.get("expect_years") or [None]
    kws = [k.lower() for k in q.get("retrieval_keywords", [])]
    pages = q.get("expect_pages")
    dt = q.get("expect_doc_type")
    ranks = []
    for y in years:
        r = None
        for i, h in enumerate(hits, 1):
            c = h.chunk
            if y is not None and c.year != y:
                continue
            if dt and c.doc_type != dt:
                continue
            if pages and c.page not in pages:
                continue
            if kws and not any(k in c.text.lower() for k in kws):
                continue
            r = i
            break
        if r is None:
            return None
        ranks.append(r)
    return max(ranks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/questions.jsonl")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()

    qs = [json.loads(l) for l in Path(args.file).read_text().splitlines() if l.strip() and not l.startswith("#")]
    pipe = RagPipeline(load_config())
    rows = []
    for q in qs:
        sr = pipe.retrieve(q["question"], top_k=args.top_k)
        row = {"id": q["id"], "abstain_expected": q.get("abstain", False)}
        if not row["abstain_expected"]:
            rank = retrieval_rank(q, sr.hits)
            row.update(retrieval_rank=rank, retrieval_hit=rank is not None)
        if not args.retrieval_only:
            a = pipe.generate(q["question"], q["question"], sr)
            row.update(abstained=a.abstained, answer=a.answer, warnings=a.warnings, cited=a.cited_ids)
            if row["abstain_expected"]:
                row["correct"] = a.abstained
            else:
                ok = (not a.abstained) and all(re.search(rx, a.answer, re.I) for rx in q.get("answer_regex", []))
                row["correct"] = ok
                row["citations_valid"] = bool(a.cited_ids) and all(1 <= n <= len(a.sources) for n in a.cited_ids)
                row["numbers_unverified"] = any("not stated verbatim" in w for w in a.warnings)
        rows.append(row)
        flag = "" if args.retrieval_only else ("PASS" if row.get("correct") else "FAIL")
        print(f"{q['id']:<22} retrieval_rank={row.get('retrieval_rank')} {flag}")

    ans = [r for r in rows if not r["abstain_expected"]]
    aba = [r for r in rows if r["abstain_expected"]]
    pct = lambda n, d: f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"
    print("\n=== Summary ===")
    if ans:
        hit = [r for r in ans if r["retrieval_hit"]]
        print("Retrieval hit@k :", pct(len(hit), len(ans)))
        print("Retrieval MRR   :", f"{sum(1 / r['retrieval_rank'] for r in hit) / len(ans):.3f}")
    if not args.retrieval_only:
        if ans:
            print("Answer correct  :", pct(sum(r["correct"] for r in ans), len(ans)))
            print("Citations valid :", pct(sum(r["citations_valid"] for r in ans), len(ans)))
            print("Numbers flagged :", pct(sum(r["numbers_unverified"] for r in ans), len(ans)), "(lower is better)")
        if aba:
            print("Abstention ok   :", pct(sum(r["correct"] for r in aba), len(aba)))
    Path("eval/results.json").write_text(json.dumps(rows, indent=2))
    print("Details -> eval/results.json")


if __name__ == "__main__":
    main()
