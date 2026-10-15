"""Embedding + on-disk index (numpy matrix + JSONL chunks; BM25 rebuilt at load).

A flat numpy matrix with exact cosine search is deliberate: the corpus is ~tens of thousands of
chunks, exact search takes milliseconds, and there is no vector-DB service to deploy.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .textutil import tokenize

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy: keeps tests light

        self.name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
        ).astype(np.float32)

    def encode_query(self, q: str) -> np.ndarray:
        prefix = QUERY_PREFIX if "bge" in self.name.lower() and "-en" in self.name.lower() else ""
        return self.model.encode([prefix + q], normalize_embeddings=True)[0].astype(np.float32)


@dataclass
class IndexData:
    chunks: list[Chunk]
    emb: np.ndarray
    bm25: BM25Okapi
    years: np.ndarray
    types: np.ndarray
    info: dict


def save_index(index_dir: Path, chunks: list[Chunk], emb: np.ndarray, info: dict) -> None:
    """Stored compactly (gzip JSONL + float16 vectors) so the index fits under GitHub's 100 MB file limit."""
    index_dir.mkdir(parents=True, exist_ok=True)
    old = index_dir / "chunks.jsonl"
    if old.exists():
        old.unlink()
    with gzip.open(index_dir / "chunks.jsonl.gz", "wt", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    np.save(index_dir / "embeddings.npy", emb.astype(np.float16))
    (index_dir / "index_info.json").write_text(json.dumps(info, indent=2))


def _open_chunks(index_dir: Path):
    gz = index_dir / "chunks.jsonl.gz"
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    return open(index_dir / "chunks.jsonl", encoding="utf-8")


def load_index(index_dir: Path) -> IndexData:
    index_dir = Path(index_dir)
    if not ((index_dir / "chunks.jsonl.gz").exists() or (index_dir / "chunks.jsonl").exists()):
        raise FileNotFoundError(
            f"No index at '{index_dir}'. Put PDFs in data/raw and run: python scripts/ingest.py"
        )
    with _open_chunks(index_dir) as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    emb = np.load(index_dir / "embeddings.npy").astype(np.float32)
    info = json.loads((index_dir / "index_info.json").read_text())
    bm25 = BM25Okapi([tokenize(c.embed_text()) for c in chunks])
    years = np.array([c.year if c.year is not None else -1 for c in chunks])
    types = np.array([c.doc_type for c in chunks])
    return IndexData(chunks, emb, bm25, years, types, info)