"""아티팩트 저장소: 단계별 산출물의 경로 규칙 + 원자적 쓰기.

root(영구 저장소, 예: 2TB 스토리지)/
  raw/{corp_code}/{rcept_no}.xml.gz                       L0 원문 (유일한 원본)
  parsed/{parser_ver}/{corp_code}/{rcept_no}.json.gz       L1 섹션 트리
  chunks/{chunker_ver}/{corp_code}/{rcept_no}.jsonl.gz     L2 청크
  emb/{embed_key}/{corp_code}/{rcept_no}.npz               L3 임베딩 (content_hash 로 캐시)
  index/  docstore.db  catalog.db  wiki/                   L4/L5 (재생성 가능)

- 모든 쓰기는 임시 파일 → os.replace (대여 시간이 끝나 강제 종료돼도 반쯤 쓴 파일이 남지 않음)
- 버전이 경로에 들어가므로 파서/청커/모델을 바꿔도 이전 산출물을 덮어쓰지 않는다
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from ..domain import Chunk, EmbeddingBatch, Filing, Section


class ArtifactStore:
    LAYERS = ("raw", "parsed", "chunks", "emb", "index", "wiki")

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def raw_path(self, f: Filing) -> Path:
        return self.root / "raw" / f.corp_code / f"{f.rcept_no}.xml.gz"

    def parsed_path(self, f: Filing, ver: str) -> Path:
        return self.root / "parsed" / ver / f.corp_code / f"{f.rcept_no}.json.gz"

    def chunks_path(self, f: Filing, ver: str) -> Path:
        return self.root / "chunks" / ver / f.corp_code / f"{f.rcept_no}.jsonl.gz"

    def emb_path(self, f: Filing, key: str) -> Path:
        return self.root / "emb" / key / f.corp_code / f"{f.rcept_no}.npz"

    # ------------------------------------------------------------------ atomic io
    @staticmethod
    def _atomic(path: Path, write_fn, suffix: str = ".tmp") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".part-", suffix=suffix)
        os.close(fd)
        try:
            write_fn(tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _write_gz_text(self, path: Path, text: str) -> None:
        def w(tmp):
            with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
                fh.write(text)
        self._atomic(path, w)

    @staticmethod
    def _read_gz_text(path: Path) -> str:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()

    # ------------------------------------------------------------------ L0 raw
    def has_raw(self, f: Filing) -> bool:
        return self.raw_path(f).exists()

    def save_raw(self, f: Filing, xml: str) -> Path:
        p = self.raw_path(f)
        self._write_gz_text(p, xml)
        return p

    def load_raw(self, f: Filing) -> str:
        return self._read_gz_text(self.raw_path(f))

    # ------------------------------------------------------------------ L1 parsed
    def save_parsed(self, f: Filing, ver: str, root: Section) -> Path:
        p = self.parsed_path(f, ver)
        self._write_gz_text(p, json.dumps(root.to_dict(), ensure_ascii=False))
        return p

    def load_parsed(self, f: Filing, ver: str) -> Section:
        return Section.from_dict(json.loads(self._read_gz_text(self.parsed_path(f, ver))))

    # ------------------------------------------------------------------ L2 chunks
    def save_chunks(self, f: Filing, ver: str, chunks: list[Chunk]) -> Path:
        p = self.chunks_path(f, ver)
        self._write_gz_text(p, "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in chunks))
        return p

    def load_chunks(self, f: Filing, ver: str) -> list[Chunk]:
        text = self._read_gz_text(self.chunks_path(f, ver))
        return [Chunk.from_dict(json.loads(line)) for line in text.splitlines() if line]

    # ------------------------------------------------------------------ L3 embeddings
    def save_embeddings(self, f: Filing, key: str, hashes: list[str], batch: EmbeddingBatch) -> Path:
        p = self.emb_path(f, key)
        arrays = {"hashes": np.asarray(hashes, dtype="U32"), "dense": batch.dense.astype(np.float16)}
        if batch.sparse is not None:
            indptr = np.zeros(len(batch.sparse) + 1, dtype=np.int64)
            for i, sv in enumerate(batch.sparse):
                indptr[i + 1] = indptr[i] + len(sv)
            arrays["sp_indptr"] = indptr
            arrays["sp_idx"] = np.fromiter((k for sv in batch.sparse for k in sv), dtype=np.uint32, count=int(indptr[-1]))
            arrays["sp_val"] = np.fromiter((v for sv in batch.sparse for v in sv.values()), dtype=np.float16,
                                           count=int(indptr[-1]))
        self._atomic(p, lambda tmp: np.savez(tmp, **arrays), suffix=".npz")
        return p

    def load_embeddings(self, f: Filing, key: str) -> dict[str, tuple[np.ndarray, dict | None]]:
        """content_hash → (dense float32, sparse dict | None)."""
        p = self.emb_path(f, key)
        if not p.exists():
            return {}
        z = np.load(p)
        dense = z["dense"].astype(np.float32)
        has_sp = "sp_indptr" in z.files
        out = {}
        for i, h in enumerate(z["hashes"]):
            sp = None
            if has_sp:
                a, b = z["sp_indptr"][i], z["sp_indptr"][i + 1]
                sp = {int(k): float(v) for k, v in zip(z["sp_idx"][a:b], z["sp_val"][a:b])}
            out[str(h)] = (dense[i], sp)
        return out

    # ------------------------------------------------------------------ usage
    def usage(self) -> dict[str, int]:
        out = {}
        for layer in self.LAYERS:
            d = self.root / layer
            out[layer] = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) if d.exists() else 0
        for name in ("docstore.db", "catalog.db"):
            p = self.root / name
            out[name] = p.stat().st_size if p.exists() else 0
        return out
