"""인덱스 계층: 벡터 인덱스(검색) + DocStore(본문) + BM25 인코더."""
from __future__ import annotations

from ..config import Settings
from .base import IndexHit, IndexRecord, QueryVectors, VectorIndex
from .bm25 import BM25Encoder
from .docstore import DocStore, PostgresDocStore, SQLiteDocStore
from .tokenizer import make_tokenizer

__all__ = ["VectorIndex", "IndexRecord", "IndexHit", "QueryVectors", "DocStore", "SQLiteDocStore",
           "PostgresDocStore", "BM25Encoder", "make_index", "make_bm25", "make_docstore"]


def make_docstore(settings: Settings) -> DocStore:
    dc = settings.docstore
    if dc.backend == "postgres":
        return PostgresDocStore(dc.resolved_dsn(), dc.schema)
    if dc.backend == "sqlite":
        return SQLiteDocStore(settings.storage.root_path / "docstore.db")
    raise ValueError(f"unknown docstore backend: {dc.backend} (postgres | sqlite)")


def make_bm25(settings: Settings) -> BM25Encoder:
    s = settings.sparse
    return BM25Encoder(make_tokenizer(s.tokenizer), s.k1, s.b, s.avgdl)


def make_index(settings: Settings, embed_key: str, dim: int, use_m3: bool) -> VectorIndex:
    """임베더 key 별로 분리된 인덱스를 연다 (모델이 다르면 다른 인덱스)."""
    ic = settings.index
    root = settings.storage.root_path / "index"
    if ic.backend == "qdrant":
        from .qdrant import QdrantIndex
        return QdrantIndex(f"{ic.collection_prefix}__{embed_key}", dim, use_m3, url=ic.qdrant_url,
                           path=root / "qdrant", api_key_env=ic.qdrant_api_key_env,
                           quantization=ic.quantization, on_disk=ic.on_disk, upsert_batch=ic.upsert_batch)
    if ic.backend == "memory":
        from .memory import InMemoryIndex
        return InMemoryIndex(root / "memory" / embed_key, dim)
    raise ValueError(f"unknown index backend: {ic.backend}")
