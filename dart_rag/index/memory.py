"""numpy 인메모리 인덱스 (수만~수십만 청크, 노트북 데모용).

- 쓰기는 버퍼에만 쌓고, 검색 시점에 한 번 행렬을 만든다 (기존: 보고서마다 BM25 전체 재구축 + 파일 전체 재저장)
- flush() 때만 디스크에 저장 → 단계 실행 1회당 1번
- BM25 IDF 는 Qdrant Modifier.IDF 와 같은 식:  ln(1 + (N − df + 0.5)/(df + 0.5))
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..domain import SearchFilter, SparseVec
from .base import IndexHit, IndexRecord, QueryVectors, VectorIndex, rrf


class InMemoryIndex(VectorIndex):
    name = "memory"

    def __init__(self, path: Path, dim: int):
        self.path = Path(path)
        self.dim = dim
        self.records: dict[str, IndexRecord] = {}
        self._dirty = True
        self._mat: np.ndarray | None = None
        self._ids: list[str] = []
        self._inv: dict[str, dict[int, list[tuple[int, float]]]] = {}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self):
        meta, vec = self.path / "records.jsonl.gz", self.path / "dense.npy"
        if not (meta.exists() and vec.exists()):
            return
        M = np.load(vec)
        with gzip.open(meta, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                self.records[d["id"]] = IndexRecord(
                    d["id"], d["payload"], M[i].astype(np.float32),
                    {int(k): v for k, v in d["bm25"].items()},
                    {int(k): v for k, v in d["m3"].items()} if d.get("m3") is not None else None)

    def flush(self):
        self.path.mkdir(parents=True, exist_ok=True)
        recs = list(self.records.values())
        M = np.stack([r.dense for r in recs]).astype(np.float16) if recs else np.zeros((0, self.dim), np.float16)
        np.save(self.path / "dense.npy", M)
        with gzip.open(self.path / "records.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({"id": r.chunk_id, "payload": r.payload, "bm25": r.bm25, "m3": r.m3},
                                   ensure_ascii=False) + "\n")
        (self.path / "info.json").write_text(json.dumps({"dim": self.dim, "n": len(recs)}))

    # ------------------------------------------------------------------ write
    def upsert(self, records):
        for r in records:  # payload 는 복사 (호출자 객체와 공유하지 않음)
            self.records[r.chunk_id] = IndexRecord(r.chunk_id, dict(r.payload), r.dense, r.bm25, r.m3)
        self._dirty = True
        return len(records)

    def delete_filing(self, rcept_no):
        for cid in [k for k, r in self.records.items() if r.payload.get("rcept_no") == rcept_no]:
            del self.records[cid]
        self._dirty = True

    def set_latest(self, rcept_no, is_latest):
        for r in self.records.values():
            if r.payload.get("rcept_no") == rcept_no:
                r.payload["is_latest"] = is_latest

    def count(self):
        return len(self.records)

    # ------------------------------------------------------------------ read
    def _build(self):
        if not self._dirty:
            return
        recs = list(self.records.values())
        self._ids = [r.chunk_id for r in recs]
        self._payloads = [r.payload for r in recs]
        self._mat = np.stack([r.dense for r in recs]).astype(np.float32) if recs else np.zeros((0, self.dim), np.float32)
        self._inv = {"bm25": defaultdict(list), "m3": defaultdict(list)}
        for row, r in enumerate(recs):
            for tid, w in r.bm25.items():
                self._inv["bm25"][tid].append((row, w))
            for tid, w in (r.m3 or {}).items():
                self._inv["m3"][tid].append((row, w))
        self._dirty = False

    def _sparse_scores(self, ch: str, q: SparseVec, mask: np.ndarray) -> np.ndarray:
        n = len(self._ids)
        s = np.zeros(n, dtype=np.float32)
        inv = self._inv[ch]
        for tid, qw in q.items():
            postings = inv.get(tid)
            if not postings:
                continue
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5)) if ch == "bm25" else 1.0
            rows = np.fromiter((p[0] for p in postings), dtype=np.int64, count=len(postings))
            ws = np.fromiter((p[1] for p in postings), dtype=np.float32, count=len(postings))
            np.add.at(s, rows, qw * idf * ws)
        s[~mask] = 0
        return s

    def search(self, q: QueryVectors, flt: SearchFilter, limit: int, prefetch_limit: int = 50, channels=None):
        self._build()
        if not self._ids:
            return []
        mask = np.fromiter((flt.matches(p) for p in self._payloads), dtype=bool, count=len(self._ids)) \
            if flt else np.ones(len(self._ids), dtype=bool)
        if not mask.any():
            return []
        use = [c for c in q.channels() if channels is None or c in channels]
        rankings: dict[str, list[str]] = {}
        for ch in use:
            if ch == "dense":
                sc = self._mat @ q.dense.astype(np.float32)
                sc[~mask] = -np.inf
                valid = mask
            else:
                sc = self._sparse_scores(ch, getattr(q, ch), mask)
                valid = mask & (sc > 0)
            order = np.argsort(-sc)[:prefetch_limit]
            rankings[ch] = [self._ids[i] for i in order if valid[i]]
        return rrf(rankings)[:limit]
