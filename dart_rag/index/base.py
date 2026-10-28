"""벡터 인덱스 인터페이스.

인덱스는 "어떤 chunk_id 가 질문과 가까운가" 만 답한다. 청크 본문은 DocStore 가 가진다.
  - 채널: dense (의미) / m3 (bge-m3 sparse) / bm25 (정확 어휘·숫자)
  - 채널별 상위 prefetch_limit 개를 뽑아 RRF 로 융합
구현: InMemoryIndex (numpy, 소규모·노트북) / QdrantIndex (로컬 모드 또는 서버)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..domain import SearchFilter, SparseVec

# 인덱스 payload 에 넣는 필드 (필터용). 본문은 넣지 않는다.
PAYLOAD_FIELDS = ("chunk_id", "rcept_no", "corp_code", "corp_name", "report_type", "period", "year",
                  "kind", "parent_id", "tags", "is_latest")
KEYWORD_FIELDS = ("rcept_no", "corp_code", "corp_name", "report_type", "period", "year", "kind",
                  "parent_id", "tags")


@dataclass
class IndexRecord:
    chunk_id: str
    payload: dict
    dense: np.ndarray
    bm25: SparseVec
    m3: SparseVec | None = None


@dataclass
class QueryVectors:
    dense: np.ndarray | None = None
    m3: SparseVec | None = None
    bm25: SparseVec | None = None

    def channels(self) -> list[str]:
        return [c for c in ("dense", "m3", "bm25") if getattr(self, c) is not None and len(getattr(self, c))]


@dataclass
class IndexHit:
    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def rrf(rankings: dict[str, list[str]], k: int = 60) -> list[IndexHit]:
    """채널별 순위 목록 → RRF 점수. 순위는 1부터."""
    acc: dict[str, IndexHit] = {}
    for ch, ids in rankings.items():
        for r, cid in enumerate(ids, start=1):
            h = acc.setdefault(cid, IndexHit(cid, 0.0))
            h.score += 1.0 / (k + r)
            h.ranks[ch] = r
    return sorted(acc.values(), key=lambda h: -h.score)


class VectorIndex(ABC):
    name = "base"

    @abstractmethod
    def upsert(self, records: list[IndexRecord]) -> int: ...

    @abstractmethod
    def delete_filing(self, rcept_no: str) -> None: ...

    @abstractmethod
    def set_latest(self, rcept_no: str, is_latest: bool) -> None: ...

    @abstractmethod
    def search(self, q: QueryVectors, flt: SearchFilter, limit: int, prefetch_limit: int = 50,
               channels: list[str] | None = None) -> list[IndexHit]: ...

    @abstractmethod
    def count(self) -> int: ...

    def flush(self) -> None:
        """영속화 (메모리 인덱스용). Qdrant 는 no-op."""

    def describe(self) -> dict:
        return {"backend": self.name, "count": self.count()}
