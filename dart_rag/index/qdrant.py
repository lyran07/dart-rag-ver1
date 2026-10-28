"""Qdrant 인덱스.

컬렉션 = `{prefix}__{embedder.key}`  (모델이 바뀌면 컬렉션도 바뀜 → 섞일 일이 없음)
  vectors         dense : 1024 (bge-m3) cosine, 원본은 디스크(on_disk), int8 양자화본만 RAM
  sparse_vectors  m3    : bge-m3 lexical weights
                  bm25  : TF 포화 가중치 + Modifier.IDF
  payload         필터 필드만 (본문은 DocStore)

연결: qdrant_url 이 있으면 서버(docker/Cloud), 없으면 로컬 모드(파일, 서버 불필요 — 노트북 데모용).
"""
from __future__ import annotations

import os
from pathlib import Path

from ..domain import SearchFilter, SparseVec
from .base import KEYWORD_FIELDS, IndexRecord, QueryVectors, VectorIndex, rrf


def _point_id(chunk_id: str) -> int:
    return int(chunk_id[:16], 16) & 0x7FFFFFFFFFFFFFFF


class QdrantIndex(VectorIndex):
    name = "qdrant"

    def __init__(self, collection: str, dim: int, use_m3: bool, url: str | None = None,
                 path: Path | None = None, api_key_env: str = "QDRANT_API_KEY",
                 quantization: str = "int8", on_disk: bool = True, upsert_batch: int = 256):
        from qdrant_client import QdrantClient, models
        self.m = models
        if url:
            self.client = QdrantClient(url=url, api_key=os.environ.get(api_key_env), timeout=60)
            self.local = False
        else:
            path = Path(path or "data/index/qdrant")
            path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path))
            self.local = True
        self.collection, self.dim, self.use_m3 = collection, dim, use_m3
        self.quantization, self.on_disk, self.batch = quantization, on_disk, upsert_batch
        self._ensure()

    # ------------------------------------------------------------------ schema
    def _ensure(self):
        m = self.m
        if self.client.collection_exists(self.collection):
            return
        sparse = {"bm25": m.SparseVectorParams(modifier=m.Modifier.IDF)}
        if self.use_m3:
            sparse["m3"] = m.SparseVectorParams()
        quant = None
        if self.quantization == "int8":
            quant = m.ScalarQuantization(scalar=m.ScalarQuantizationConfig(type=m.ScalarType.INT8, always_ram=True))
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": m.VectorParams(size=self.dim, distance=m.Distance.COSINE, on_disk=self.on_disk)},
            sparse_vectors_config=sparse,
            quantization_config=quant,
        )
        if not self.local:  # 로컬 모드는 payload index 미지원(불필요)
            for f in KEYWORD_FIELDS:
                self.client.create_payload_index(self.collection, f, m.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection, "is_latest", m.PayloadSchemaType.BOOL)

    # ------------------------------------------------------------------ helpers
    def _sv(self, v: SparseVec):
        return self.m.SparseVector(indices=list(v.keys()), values=[float(x) for x in v.values()])

    def _filter(self, flt: SearchFilter):
        if not flt:
            return None
        m = self.m
        conds = []
        for k, v in flt.must.items():
            if isinstance(v, (list, tuple, set)):
                conds.append(m.FieldCondition(key=k, match=m.MatchAny(any=list(v))))
            else:
                conds.append(m.FieldCondition(key=k, match=m.MatchValue(value=v)))
        return m.Filter(must=conds)

    def _by_rcept(self, rcept_no: str):
        m = self.m
        return m.Filter(must=[m.FieldCondition(key="rcept_no", match=m.MatchValue(value=rcept_no))])

    # ------------------------------------------------------------------ write
    def upsert(self, records: list[IndexRecord]) -> int:
        m = self.m
        for s in range(0, len(records), self.batch):
            pts = []
            for r in records[s:s + self.batch]:
                vec = {"dense": r.dense.astype(float).tolist()}
                if r.bm25:
                    vec["bm25"] = self._sv(r.bm25)
                if self.use_m3 and r.m3:
                    vec["m3"] = self._sv(r.m3)
                pts.append(m.PointStruct(id=_point_id(r.chunk_id), vector=vec, payload=r.payload))
            self.client.upsert(self.collection, points=pts, wait=True)
        return len(records)

    def delete_filing(self, rcept_no: str) -> None:
        self.client.delete(self.collection, points_selector=self.m.FilterSelector(filter=self._by_rcept(rcept_no)))

    def set_latest(self, rcept_no: str, is_latest: bool) -> None:
        self.client.set_payload(self.collection, payload={"is_latest": is_latest}, points=self._by_rcept(rcept_no))

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    # ------------------------------------------------------------------ read
    def search(self, q: QueryVectors, flt: SearchFilter, limit: int, prefetch_limit: int = 50, channels=None):
        m = self.m
        qf = self._filter(flt)
        use = [c for c in q.channels() if (channels is None or c in channels) and (c != "m3" or self.use_m3)]
        if not use:
            return []

        def qv(ch):
            return q.dense.astype(float).tolist() if ch == "dense" else self._sv(getattr(q, ch))

        # 채널별 검색을 한 번의 batch 요청으로 보내고 RRF 는 클라이언트에서 계산한다.
        #  → 메모리 인덱스와 점수 체계가 같고(k=60), 채널별 순위를 평가에 그대로 쓸 수 있다.
        #  (서버 측 융합이 필요하면 Prefetch + FusionQuery(RRF) 로 바꾸면 됨)
        reqs = [m.QueryRequest(query=qv(ch), using=ch, limit=prefetch_limit, filter=qf,
                               with_payload=["chunk_id"]) for ch in use]
        results = self.client.query_batch_points(self.collection, requests=reqs)
        rankings = {ch: [p.payload["chunk_id"] for p in r.points] for ch, r in zip(use, results)}
        return rrf(rankings)[:limit]

    def describe(self):
        return {"backend": "qdrant" + ("-local" if self.local else ""), "collection": self.collection,
                "count": self.count()}
