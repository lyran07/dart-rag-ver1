"""파이프라인 단계.

  fetch  (네트워크)  DART → raw/
  parse  (CPU)      raw → parsed/
  chunk  (CPU)      parsed → chunks/
  embed  (GPU 권장)  chunks → emb/        ← GPU 가 필요한 유일한 단계
  index  (CPU)      chunks + emb → DocStore + 벡터 인덱스 (서빙 머신에서 실행)
  wiki   (CPU)      parsed + chunks → wiki/ (선택)

각 단계는 version 을 가진다. catalog 에 같은 version 으로 done 이면 건너뛴다.
"""
from __future__ import annotations

import socket
from abc import ABC, abstractmethod
from functools import cached_property

from ..config import Settings
from ..domain import EmbeddingBatch, Filing
from ..ingestion.chunker import DartChunker
from ..ingestion.parser import DartXmlParser
from .catalog import Catalog
from .storage import ArtifactStore


class StageContext:
    """단계들이 공유하는 자원. 무거운 것(임베더·인덱스)은 처음 쓸 때 만든다."""

    def __init__(self, settings: Settings, source=None):
        self.settings = settings
        self.store = ArtifactStore(settings.storage.root_path)
        self.catalog = Catalog(settings.storage.root_path / "catalog.db")
        self.source = source
        self.parser = DartXmlParser()
        self.chunker = DartChunker(settings.chunking)
        self.host = socket.gethostname()

    @cached_property
    def embedder(self):
        from ..embedding import make_embedder
        e = make_embedder(self.settings.embedding)
        print(f"[embed] {e.describe()}")
        return e

    @cached_property
    def embed_key(self) -> str:
        """임베더를 로드하지 않고도 key 를 알 수 있어야 index 단계가 가볍다 → 설정에서 계산."""
        from ..embedding.base import configured_key
        return configured_key(self.settings.embedding.provider, self.settings.embedding.model)

    @cached_property
    def bm25(self):
        from ..index import make_bm25
        return make_bm25(self.settings)

    @cached_property
    def docstore(self):
        from ..index import make_docstore
        return make_docstore(self.settings)

    def open_index(self, dim: int, use_m3: bool):
        if not hasattr(self, "_index"):
            from ..index import make_index
            self._index = make_index(self.settings, self.embed_key, dim, use_m3)
        return self._index

    @cached_property
    def wiki(self):
        from ..wiki import WikiBuilder
        return WikiBuilder(self.settings.storage.root_path / "wiki")


class Stage(ABC):
    name = "base"
    needs_gpu = False

    def __init__(self, ctx: StageContext):
        self.ctx = ctx

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def run(self, f: Filing) -> int:
        """처리한 항목 수 반환."""

    def finalize(self) -> None:
        """실행 끝(또는 마감 중단) 시 1회."""


class FetchStage(Stage):
    name = "fetch"

    def version(self):
        return "f1"

    def run(self, f):
        if self.ctx.store.has_raw(f):
            return 0
        if self.ctx.source is None:
            raise RuntimeError("fetch 단계에는 source(dart/sample) 가 필요합니다.")
        xml = self.ctx.source.fetch(f)
        self.ctx.store.save_raw(f, xml)
        return len(xml)


class ParseStage(Stage):
    name = "parse"

    def version(self):
        return self.ctx.parser.version

    def run(self, f):
        root = self.ctx.parser.parse(self.ctx.store.load_raw(f))
        self.ctx.store.save_parsed(f, self.version(), root)
        return sum(1 for _ in root.walk()) - 1


class ChunkStage(Stage):
    name = "chunk"

    def version(self):
        return f"{self.ctx.parser.version}.{self.ctx.chunker.version}"

    def run(self, f):
        root = self.ctx.store.load_parsed(f, self.ctx.parser.version)
        chunks = self.ctx.chunker.chunk(root, f)
        self.ctx.store.save_chunks(f, self.version(), chunks)
        return len(chunks)


class EmbedStage(Stage):
    """embed_kinds(기본 text, table_kv)만 임베딩. 표 마크다운(table)은 DocStore 에만 → 벡터 약 40% 절감.
    같은 보고서의 이전 임베딩 파일에서 content_hash 가 같은 벡터는 재사용 (GPU 시간 절약)."""
    name = "embed"
    needs_gpu = True
    CHECKPOINT = 32   # 이 개수마다 중간 저장 (CPU 기준 1~2분 분량)

    def version(self):
        ec = self.ctx.settings.embedding
        sel = ""
        if ec.skip_sections or sorted(ec.embed_kinds) != ["table_kv", "text"]:
            import hashlib
            sel = ".s" + hashlib.md5(repr((sorted(ec.embed_kinds), ec.skip_sections)).encode()).hexdigest()[:4]
        return f"{self.ctx.embed_key}.{self.ctx.parser.version}.{self.ctx.chunker.version}{sel}"

    def run(self, f):
        ctx = self.ctx
        ec = ctx.settings.embedding
        kinds = set(ec.embed_kinds)
        chunks = [c for c in ctx.store.load_chunks(f, ChunkStage(ctx).version())
                  if c.kind in kinds and not any(s in c.meta.get("section_path", "") for s in ec.skip_sections)]
        hashes = list(dict.fromkeys(c.content_hash for c in chunks))
        text_of = {c.content_hash: c.text for c in chunks}
        cache = ctx.store.load_embeddings(f, ctx.embed_key)   # 이전 실행의 중간 저장분 포함
        todo = [h for h in hashes if h not in cache]
        import numpy as np

        def save(keys):
            dense = np.stack([cache[h][0] for h in keys]) if keys else np.zeros((0, 1), np.float32)
            has_sp = bool(keys) and all(cache[h][1] is not None for h in keys)
            ctx.store.save_embeddings(f, ctx.embed_key, keys,
                                      EmbeddingBatch(dense, [cache[h][1] for h in keys] if has_sp else None))

        if todo:
            # 긴 텍스트부터(패딩 낭비 감소) CHECKPOINT 개씩 임베딩 → 매번 중간 저장. 끊겨도 다시 실행하면 이어서.
            import sys
            import time
            todo.sort(key=lambda h: len(text_of[h]), reverse=True)
            total, base, t0 = len(hashes), len(hashes) - len(todo), time.perf_counter()
            if base:
                print(f"[embed] 중간 저장분 {base}/{total} 재사용, 남은 {len(todo)}개부터 이어서")
            for s in range(0, len(todo), self.CHECKPOINT):
                part = todo[s:s + self.CHECKPOINT]
                b = ctx.embedder.embed_documents([text_of[h] for h in part])
                for i, h in enumerate(part):
                    cache[h] = (b.dense[i], b.sparse[i] if b.sparse else None)
                save([h for h in hashes if h in cache])
                done, el = s + len(part), time.perf_counter() - t0
                eta = el / done * (len(todo) - done)
                sys.stderr.write(f"\r[embed] {base + done}/{total} ({(base + done) * 100 // total}%) 저장됨"
                                 f" · 경과 {int(el // 60)}분 {int(el % 60)}초 · 남은 예상 {int(eta // 60)}분 {int(eta % 60)}초   ")
                sys.stderr.flush()
            sys.stderr.write("\n")
        save(hashes)
        return len(todo)


class IndexStage(Stage):
    """아티팩트 → DocStore + 벡터 인덱스. GPU 불필요 (BM25 도 여기서 계산)."""
    name = "index"

    def version(self):
        ic = self.ctx.settings.index
        return f"{EmbedStage(self.ctx).version()}.{ic.backend}.{ic.collection_prefix}.{self.ctx.bm25.key}"

    def run(self, f):
        import numpy as np
        from ..index import IndexRecord
        from ..index.base import PAYLOAD_FIELDS
        ctx = self.ctx
        chunks = ctx.store.load_chunks(f, ChunkStage(ctx).version())
        embs = ctx.store.load_embeddings(f, ctx.embed_key)
        if not embs:
            raise RuntimeError(f"임베딩 없음: {ctx.store.emb_path(f, ctx.embed_key)} (embed 단계 먼저)")
        dim = int(next(iter(embs.values()))[0].shape[0])
        use_m3 = next(iter(embs.values()))[1] is not None
        index = ctx.open_index(dim, use_m3)
        ctx.docstore.replace_filing(f.rcept_no, chunks)
        recs = []
        for c in chunks:
            e = embs.get(c.content_hash)
            if e is None:
                continue  # 임베딩 대상 아닌 kind (table) → DocStore 에만
            payload = {k: c.meta.get(k) for k in PAYLOAD_FIELDS if k in c.meta}
            payload.update(chunk_id=c.id, kind=c.kind, is_latest=f.is_latest)
            recs.append(IndexRecord(c.id, payload, np.asarray(e[0], np.float32), ctx.bm25.encode_document(c.text), e[1]))
        index.delete_filing(f.rcept_no)
        index.upsert(recs)
        return len(recs)

    def finalize(self):
        if not hasattr(self.ctx, "_index"):
            return
        # 기재정정으로 최신본이 바뀐 공시의 is_latest 동기화
        for f in self.ctx.catalog.filings():
            if self.ctx.catalog.is_done(f.rcept_no, self.name, self.version()):
                self.ctx._index.set_latest(f.rcept_no, f.is_latest)
        self.ctx._index.flush()


class WikiStage(Stage):
    name = "wiki"

    def version(self):
        return f"{ChunkStage(self.ctx).version()}.{self.ctx.wiki.version}"

    def run(self, f):
        root = self.ctx.store.load_parsed(f, self.ctx.parser.version)
        chunks = self.ctx.store.load_chunks(f, ChunkStage(self.ctx).version())
        return self.ctx.wiki.add_report(root, f, chunks)

    def finalize(self):
        if "wiki" in self.ctx.__dict__:
            self.ctx.wiki.flush()


class ReportStage(Stage):
    """공시 1건 → 기업분석 보고서 1건 (전년 동기 공시가 있으면 비교 포함). data/reports/ 에 저장.
    LLM(GPT-6 Luna)을 쓰므로 OPENAI_API_KEY 가 없으면 실패로 기록하고, 키를 넣고 다시 run 하면 이어서 만든다.
    기재정정 이전본(is_latest=False)은 만들지 않는다."""
    name = "report"
    template = "company_analysis"

    @cached_property
    def builder(self):
        from types import SimpleNamespace

        from ..agent import load_agent_config
        from ..report import ReportBuilder
        from ..routing import load_routes
        # 파이프라인 안에서는 서비스 대신 ctx 의 위키(이번 실행에서 막 만든 페이지 포함)를 쓴다
        svc = SimpleNamespace(wiki=self.ctx.wiki, routes=load_routes(), agent_config=load_agent_config(),
                              settings=self.ctx.settings)
        return getattr(self.ctx, "report_builder", None) or ReportBuilder(svc)

    def version(self):
        return f"{WikiStage(self.ctx).version()}.{self.builder.version(self.template)}"

    def _ensure_wiki(self, f) -> bool:
        """비교 공시의 위키가 아직 없으면 (최신 공시부터 처리하므로 흔함) 여기서 만든다."""
        ws = WikiStage(self.ctx)
        if self.ctx.catalog.is_done(f.rcept_no, ws.name, ws.version()):
            return True
        if not self.ctx.catalog.is_done(f.rcept_no, ChunkStage.name, ChunkStage(self.ctx).version()):
            return False
        n = ws.run(f)
        self.ctx.catalog.mark(f.rcept_no, ws.name, ws.version(), "done", n_items=n, host=self.ctx.host)
        return True

    def run(self, f):
        import os
        if not f.is_latest:
            return 0
        if not getattr(self.ctx, "report_builder", None) and not os.environ.get(
                self.builder.svc.agent_config.backends.get("responses", {}).get("api_key_env", "OPENAI_API_KEY")):
            raise RuntimeError("OPENAI_API_KEY 가 없어 보고서를 만들 수 없습니다 (키를 넣고 다시 run 하면 이어서 생성)")
        want = f"{int(f.period[:4]) - 1}{f.period[4:]}"
        prev = next((x for x in self.ctx.catalog.filings([f.corp_code], latest_only=True)
                     if x.period == want and x.report_type == f.report_type), None)
        compare = want if prev is not None and self._ensure_wiki(prev) else None
        job = self.builder.run(f.corp_name, f.period, compare, self.template)
        bad = [s.id for s in job.sections if s.status == "error"]
        if job.status != "done" or bad:
            raise RuntimeError(f"보고서 생성 실패: {job.error or ''} 오류 섹션 {bad}")
        return sum(s.status == "done" for s in job.sections)

    def finalize(self):
        if "wiki" in self.ctx.__dict__:
            self.ctx.wiki.flush()


STAGES: dict[str, type[Stage]] = {s.name: s for s in (FetchStage, ParseStage, ChunkStage, EmbedStage, IndexStage,
                                                     WikiStage, ReportStage)}
DEFAULT_ORDER = ["fetch", "parse", "chunk", "embed", "index", "wiki", "report"]
