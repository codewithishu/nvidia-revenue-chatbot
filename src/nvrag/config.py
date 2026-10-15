"""Central configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass


def _bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    data_dir: Path
    index_dir: Path
    manifest_path: Path
    embed_model: str
    rerank_model: str
    use_reranker: bool
    top_k: int
    min_dense: float
    ocr: bool
    llm_provider: str
    llm_model: str
    llm_max_tokens: int = 4000
    target_chunk_chars: int = 1400
    max_chunk_chars: int = 2000
    table_max_chars: int = 3500


def load_config() -> Config:
    return Config(
        data_dir=Path(os.getenv("DATA_DIR", "data/raw")),
        index_dir=Path(os.getenv("INDEX_DIR", "index")),
        manifest_path=Path(os.getenv("MANIFEST_PATH", "data/manifest.csv")),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base"),
        use_reranker=_bool("USE_RERANKER"),
        top_k=int(os.getenv("TOP_K", "8")),
        min_dense=float(os.getenv("MIN_DENSE_SCORE", "0.25")),
        ocr=_bool("OCR"),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic").strip().lower(),
        llm_model=os.getenv("LLM_MODEL", "").strip(),
    )
