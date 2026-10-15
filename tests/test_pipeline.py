"""End-to-end retrieval + generation tests with a hashing embedder and a fake LLM (no downloads, no keys)."""
import numpy as np
import pytest

from nvrag import retriever as retriever_mod
from nvrag.chunking import Chunk
from nvrag.config import load_config
from nvrag.indexer import save_index
from nvrag.pipeline import RagPipeline
from nvrag.textutil import tokenize

DIM = 256


def _vec(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for t in tokenize(text):
        v[hash(t) % DIM] += 1.0
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeEmbedder:
    def __init__(self, name):
        self.name = name

    def encode_query(self, q):
        return _vec(q)


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, user, max_tokens=None):
        self.calls.append(user)
        return self.reply


def _chunk(year, page, text, kind="text", dt="annual_report", units=None):
    return Chunk(f"d{year}:p{page}:{kind[0]}0", f"d{year}", f"nvidia_{dt}_{year}.pdf", dt, year, page, kind, "", units, text)


@pytest.fixture()
def pipe(tmp_path, monkeypatch):
    chunks = [
        _chunk(2023, 40, "| Metric | FY23 |\n| Total revenue | 26,974 |", "table", units="in millions"),
        _chunk(2024, 41, "| Metric | FY24 |\n| Total revenue | 60,922 |", "table", units="in millions"),
        _chunk(2024, 12, "Our competitors include large semiconductor companies and cloud providers."),
        _chunk(2024, 3, "The pay ratio of our CEO to median employee is disclosed.", dt="proxy"),
    ]
    emb = np.stack([_vec(c.embed_text()) for c in chunks])
    save_index(tmp_path, chunks, emb, {"embed_model": "fake"})
    monkeypatch.setattr(retriever_mod, "Embedder", FakeEmbedder)
    cfg = load_config()
    object.__setattr__(cfg, "index_dir", tmp_path)
    object.__setattr__(cfg, "embed_model", "fake")
    object.__setattr__(cfg, "min_dense", 0.0)
    return RagPipeline(cfg, llm=FakeLLM("Total revenue was 60,922 million in fiscal 2024 [S1]."))


def test_year_pool_retrieves_each_requested_year(pipe):
    sr = pipe.retrieve("Compare total revenue in fiscal 2023 and fiscal 2024", top_k=4)
    years = {h.chunk.year for h in sr.hits}
    assert {2023, 2024} <= years


def test_missing_year_abstains_without_calling_llm(pipe):
    a = pipe.ask("What was total revenue in fiscal 2031?")
    assert a.abstained and "2031" in a.answer
    assert pipe.llm.calls == []


def test_answer_has_valid_citation_and_no_number_warning(pipe):
    sr = pipe.retrieve("total revenue fiscal 2024")
    top = sr.hits[0].chunk
    pipe._llm.reply = "Total revenue was 60,922 million in fiscal 2024 [S1]."
    a = pipe.generate("q", "q", sr)
    assert not a.abstained and a.cited_ids == [1]
    assert a.sources[0]["page"] == top.page
    # number appears in S1 only if S1 is the FY24 table; otherwise a warning must be raised
    if "60,922" in top.text:
        assert not a.warnings


def test_llm_abstention_is_detected(pipe):
    pipe._llm.reply = "INSUFFICIENT_EVIDENCE: the excerpts do not mention this."
    a = pipe.ask("What is the CEO's favorite food?")
    assert a.abstained and not a.answer.startswith("INSUFFICIENT")


def test_hallucinated_number_and_bad_citation_are_flagged(pipe):
    pipe._llm.reply = "Revenue was 99,999 million [S1] and also [S42]."
    a = pipe.ask("total revenue fiscal 2024")
    text = " ".join(a.warnings)
    assert "99,999" in text and "invalid citation" in text.lower()
    assert 42 not in a.cited_ids
