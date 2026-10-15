# NVIDIA Filings RAG Chatbot

Live Demo video = https://nvidia-revenue-chatbot-8nl8e5n9cfzssnecgebca6.streamlit.app/

A retrieval-augmented chatbot that answers questions about NVIDIA annual reports and proxy statements
(1998–2026) with **page-level citations**, careful handling of **tables / units / fiscal periods**, and
**explicit abstention** when the filings don't contain the answer.

* **Stack:** Python · PyMuPDF (parsing + tables) · `BAAI/bge-small-en-v1.5` embeddings (local) · BM25 ·
  reciprocal-rank fusion · optional cross-encoder reranker · Claude or OpenAI-compatible LLM · Streamlit UI.
* **No vector DB service:** exact cosine search over a numpy matrix (fast at this scale, trivial to deploy).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # on CPU-only hosts: pip install torch --index-url https://download.pytorch.org/whl/cpu first
cp .env.example .env                       # add ANTHROPIC_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai)

# 1. Get the data (see below), then build the index
python scripts/ingest.py                   # writes index/ and docs/COVERAGE.md

# 2. Chat
streamlit run app.py                       # web UI
python cli.py "What was NVIDIA's revenue in fiscal 2024?"   # or interactive: python cli.py

# 3. Test / evaluate
pytest -q                                  # unit + pipeline tests (no keys/models needed)
python scripts/evaluate.py --retrieval-only
python scripts/evaluate.py
```

### Getting the data
Download the PDFs from NVIDIA's Investor Relations archive (*Financial Info → Annual Reports and Proxies*)
into `data/raw/`. Either:
* name files like `nvidia_annual_report_2024.pdf` / `nvidia_proxy_2024.pdf` (any name containing
  "proxy" → proxy, "10-K"/"annual"/"report"/"review" → annual report, and one 4-digit year), **or**
* list them in `data/manifest.csv` (`file,doc_type,year`) to override inference, **or**
* put URLs in `data/sources.csv` (`url,doc_type,year`) and run `python scripts/download.py`.

Document types/years are inferred in this order: manifest → filename → text on the first pages
(`fiscal year ended January 28, 2024`, or the annual-meeting date for proxies). **`docs/COVERAGE.md` is
generated at ingest time and lists exactly which files were ingested, page/chunk counts, missing
years (1998–2026), likely-scanned PDFs and files whose type/year couldn't be inferred.**
Year meaning: for annual reports it is the NVIDIA *fiscal* year (FY2024 = year ended Jan 28, 2024);
for proxies it is the meeting/filing year.

## How it works

```
PDFs ──► parse (PyMuPDF) ──► chunk (per page, tables separate) ──► embed + BM25 ──► index/
                                                                                      │
question ─► [condense follow-up] ─► analyze years/doc type ─► hybrid retrieve (per-year pools)
        ─► [rerank] ─► grounded prompt ─► LLM ─► verify citations & numbers ─► answer + sources
```

**Parsing (`parsing.py`).** Text blocks come from PyMuPDF in reading order. Tables are detected with
`find_tables()`, cleaned (the `$`, `)` and `%` cells that SEC-style layouts put in their own columns are folded
or dropped), and rendered as markdown. Text blocks inside a table's bbox are removed to avoid duplicated,
scrambled numbers. For each table we capture the caption above it, and each page's unit hint
(`in millions`, `in thousands`, `except per share data`). Optional OCR (`OCR=1`, needs Tesseract) handles
scanned pages.

**Chunking (`chunking.py`).** Chunks never cross page boundaries, so each citation is one PDF page.
Prose is packed by paragraph to ≈1,400 characters; headings (`Item 7…`, ALL-CAPS lines) start new chunks and
are carried forward as `section`. Each table is its own chunk (caption + markdown + units); large tables are
split by rows with the header row repeated. Every chunk is embedded with a header such as
`NVIDIA Annual Report/10-K | FY2024 | page 41 | table | units: in millions`, so period and units are part of
what is searched.

**Retrieval (`retriever.py`).** Dense cosine + BM25 (numbers like `60,922` are kept as single tokens), fused
with reciprocal-rank fusion. Because filings are near-duplicates year to year, the retriever builds a
**separate pool per year named in the question** (supports `fiscal 2024`, `FY24`, `2019-2023`, bare `2024`) plus
a small global pool, so comparisons (“FY2023 vs FY2024”) get evidence for both. Proxy-flavoured questions
(pay ratio, directors, say-on-pay…) get a soft boost for proxy documents. A bare year also peeks at the next
fiscal year's report. `USE_RERANKER=1` adds cross-encoder reranking within each pool.

**Generation (`pipeline.py`).** Retrieved passages are labelled `[S1]…[Sn]` with file, period, PDF page,
kind and units. The system prompt forces: use only the excerpts, cite every claim, copy numbers exactly with
metric + unit + period, never mix fiscal/calendar years, show arithmetic, report conflicting values, and begin
with `INSUFFICIENT_EVIDENCE:` when the excerpts don't support an answer. Temperature 0.

**Citations.** The model cites `[S#]`; the app maps each id to `file · PDF page · period · kind` and shows the
exact passage in an expander. Invalid ids are stripped with a warning; an answer with no citations is flagged.
Cited pages are *PDF* page indices (1-based); the printed page number is stored too (`printed_page`).

**Abstention / failure handling** (layered):
1. Question names a year that isn't indexed (e.g. fiscal 2031) → abstain **without** calling the LLM.
2. Best dense similarity below `MIN_DENSE_SCORE` → abstain without calling the LLM.
3. LLM judges the excerpts insufficient → `INSUFFICIENT_EVIDENCE` → shown as an abstention.
4. Post-hoc **number check**: every number in the answer is looked up in the *cited* passages; numbers not
   found verbatim (computed, rounded, or hallucinated) trigger a visible warning.
5. Missing years from a multi-year question are passed to the LLM so it answers only what's covered.
Provider/key errors surface as readable messages in the UI/CLI.

## Inputs
Natural-language questions (follow-ups work: the previous turns are used to rewrite the question into a
standalone one). Sidebar: number of passages, optional year restriction, toggle for uncited passages.
Example questions:
* “What was total revenue in fiscal 2024, and what was it in fiscal 2023?”
* “How did Data Center revenue change between fiscal 2023 and fiscal 2024?”
* “What were the main risk factors related to export controls in the FY2023 annual report?”
* “What does the 2024 proxy statement say about the CEO pay ratio?”
* “Who served on the board of directors in 2020?”
* “What was NVIDIA’s revenue in fiscal 2031?” → abstains.

## Evaluation (`eval/`, `scripts/evaluate.py`, `tests/`)
* **Unit/integration tests (`pytest`)** – query/year parsing, metadata inference, table → markdown cleanup,
  chunk boundaries/headers, a real generated PDF with a ruled table (units + table detection + no duplicate text),
  citation/number verification, and the whole retrieve→generate→abstain flow with a fake embedder and fake LLM.
* **Retrieval metrics** – `evaluate.py --retrieval-only` reports hit@k and MRR; a hit requires the right year
  (all years for comparisons), doc type, optional page, and a keyword in the retrieved chunk.
* **Answer metrics** – correctness via regex on the final answer, citation validity, rate of unverified
  numbers, and abstention accuracy on unanswerable questions (future years, other companies, trivia, forecasts).
* `eval/questions.jsonl` is a small smoke set; **verify its expectations against your ingested PDFs and extend it**.
  It is not (and must not be) a copy of the private golden set.
* Suggested manual review when tuning: run with `USE_RERANKER=0/1`, `TOP_K=6/8/12` and compare hit@k.

## Deployment
* **Local / VM:** as above. **Docker:** `docker build -t nvidia-rag .` then
  `docker run -p 8501:8501 --env-file .env -v $PWD/index:/app/index nvidia-rag` (build the index first, or
  bake PDFs into the image).
* **Streamlit Community Cloud / Hugging Face Spaces:** push the repo with a prebuilt `index/` (remove it from
  `.gitignore`, or use Git LFS for `embeddings.npy`; ~tens of MB), set `ANTHROPIC_API_KEY` as a secret, entry point `app.py`.
* No secrets are in the repo; keys are read from the environment only.

## Design decisions
* **Per-page chunks** → verifiable page citations beat slightly better cross-page context.
* **Tables as first-class chunks with units/caption** → numeric questions retrieve a table, not a paragraph about it.
* **Hybrid search** → BM25 nails exact figures/names; embeddings catch paraphrases.
* **Per-year pools** → fixes the main failure mode of RAG over repeated annual filings.
* **Verification layer** instead of trusting the LLM → warnings on unsupported numbers/citations.
* **Local embeddings + flat index** → reproducible, no extra service, only the LLM needs a key.

## Known limitations
* Not run against the real PDFs in this repo's authoring environment – run `scripts/ingest.py` and check
  `docs/COVERAGE.md`. Table detection is heuristic; very wide or image-based tables may be imperfect.
* Pre-2000s filings may be scanned images → need `OCR=1` + Tesseract; OCR numbers are less reliable.
* Charts/images and the glossy “annual review” design pages are not interpreted (text only).
* Year inference for oddly named files can be wrong → use `data/manifest.csv`.
* Fiscal vs calendar year ambiguity for bare years is handled heuristically (next-year peek + prompt rules).
* The number check flags derived values too (growth %, billions conversions); it is a prompt for human review.
* Retrieval is flat/exact (fine for ≲100k chunks); switch to FAISS/pgvector beyond that.
* No automatic LLM-as-judge; correctness checks are regex-based.

---
### Submission text (copy into the form)
**Short description:** RAG chatbot over NVIDIA annual reports & proxy statements. PyMuPDF parsing with
table-aware chunking, hybrid (BGE embeddings + BM25) retrieval with per-fiscal-year pools, Claude/OpenAI
generation constrained to cited excerpts, page-level citations, layered abstention, and a post-hoc
citation/number verifier. Streamlit UI + CLI, tests and an evaluation harness.
**Detailed working / inputs / setup / limitations:** see the sections above.
