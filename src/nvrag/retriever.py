"""Hybrid retrieval (dense + BM25, reciprocal-rank fusion) with year-aware pools.

Why year-aware pools: filings are near-duplicates across years (same boilerplate, similar tables).
A plain top-k for "compare FY2023 and FY2024 data center revenue" tends to return one year only.
So for each year named in the question we retrieve a separate pool, plus a small global pool.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .config import Config, load_config
from .indexer import Embedder, load_index
from .query import QueryInfo, analyze_query
from .textutil import tokenize


@dataclass
class Hit:
    chunk: Chunk
    dense: float
    bm25: float
    fused: float
    rerank: float | None = None


@dataclass
class SearchResult:
    hits: list[Hit]
    info: QueryInfo
    missing_years: list[int] = field(default_factory=list)
    weak: bool = False
    reason: str | None = None  # set when we already know the question is unanswerable


def _rank_fuse(dense: np.ndarray, bm: np.ndarray, idx: np.ndarray, k_rrf: int = 60) -> np.ndarray:
    n = len(idx)
    rd = np.empty(n, dtype=int)
    rd[np.argsort(-dense[idx], kind="stable")] = np.arange(n)
    rb = np.empty(n, dtype=int)
    rb[np.argsort(-bm[idx], kind="stable")] = np.arange(n)
    return 1.0 / (k_rrf + rd + 1) + 1.0 / (k_rrf + rb + 1)


class Retriever:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.idx = load_index(self.cfg.index_dir)
        built_with = self.idx.info.get("embed_model")
        if built_with and built_with != self.cfg.embed_model:
            raise RuntimeError(
                f"Index was built with '{built_with}' but EMBED_MODEL is '{self.cfg.embed_model}'. "
                "Re-run ingestion or fix EMBED_MODEL."
            )
        self.embedder = Embedder(self.cfg.embed_model)
        self._reranker = None

    # ---- helpers -------------------------------------------------------
    @property
    def available_years(self) -> list[int]:
        return sorted({int(y) for y in set(self.idx.years.tolist()) if y >= 0})

    @property
    def reranker(self):
        if self.cfg.use_reranker and self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(self.cfg.rerank_model)
        return self._reranker

    def _select(self, query, idx, dense, bm, k, doc_type) -> list[Hit]:
        if len(idx) == 0 or k <= 0:
            return []
        fused = _rank_fuse(dense, bm, idx)
        if doc_type:
            fused = fused * np.where(self.idx.types[idx] == doc_type, 1.25, 1.0)
        rr = self.reranker
        take = min(len(idx), k * 3 if rr else k)
        order = np.argsort(-fused, kind="stable")[:take]
        hits = [
            Hit(self.idx.chunks[idx[o]], float(dense[idx[o]]), float(bm[idx[o]]), float(fused[o]))
            for o in order
        ]
        if rr and hits:
            scores = rr.predict([(query, h.chunk.header() + "\n" + h.chunk.text) for h in hits])
            for h, s in zip(hits, scores):
                h.rerank = float(s)
            hits.sort(key=lambda h: -h.rerank)
            hits = hits[:k]
        return hits

    # ---- main entry ----------------------------------------------------
    def search(self, query: str, top_k: int | None = None, years: list[int] | None = None) -> SearchResult:
        k = top_k or self.cfg.top_k
        info = analyze_query(query)
        if years:  # explicit UI filter wins over parsed years
            info = QueryInfo(years=sorted(years), plain=set(), doc_type=info.doc_type)

        qv = self.embedder.encode_query(query)
        dense = self.idx.emb @ qv
        bm = np.asarray(self.idx.bm25.get_scores(tokenize(query)))

        pools: list[list[Hit]] = []
        missing: list[int] = []
        found = 0
        ys = info.years
        if ys:
            per = max(3, math.ceil(k / len(ys)))
            for y in ys:
                idx = np.where(self.idx.years == y)[0]
                if len(idx) == 0:
                    missing.append(y)
                else:
                    found += 1
                    pools.append(self._select(query, idx, dense, bm, per, info.doc_type))
                if y in info.plain and len(ys) == 1:  # bare "2024": also peek at the next fiscal year
                    nidx = np.where(self.idx.years == y + 1)[0]
                    if len(nidx):
                        found += 1
                        pools.append(self._select(query, nidx, dense, bm, 2, info.doc_type))
            global_k = max(2, k // 3)
        else:
            global_k = k
        pools.append(self._select(query, np.arange(len(self.idx.chunks)), dense, bm, global_k, info.doc_type))

        seen, hits = set(), []
        for pool in pools:
            for h in pool:
                if h.chunk.chunk_id not in seen:
                    seen.add(h.chunk.chunk_id)
                    hits.append(h)

        res = SearchResult(hits=hits, info=info, missing_years=missing)
        if ys and found == 0:
            res.reason = (
                "No documents for " + ", ".join(map(str, ys)) + " are in the index "
                f"(indexed years: {self.available_years[0]}–{self.available_years[-1]})."
                if self.available_years
                else "The index is empty."
            )
        res.weak = (not hits) or max(h.dense for h in hits) < self.cfg.min_dense
        return res
