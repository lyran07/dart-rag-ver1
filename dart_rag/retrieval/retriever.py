"""하이브리드 검색기: 질의 분석 → 3채널 검색(RRF) → 리랭크/가산점 → small-to-big 확장.

의존성은 모두 생성자로 주입 (Embedder, VectorIndex, DocStore, BM25Encoder, Reranker, QueryAnalyzer).
→ 백엔드(memory/qdrant), 모델(bge-m3/e5/…), 리랭커 유무를 바꿔도 이 클래스는 그대로.
"""
from __future__ import annotations

import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ..config import RetrievalCfg
from ..domain import Chunk, ScoredChunk
from ..embedding import Embedder
from ..index import BM25Encoder, DocStore, QueryVectors, VectorIndex
from ..index.tokenizer import RegexTokenizer
from .query import QueryAnalyzer, QueryPlan
from .reranker import NoopReranker, Reranker

_tok = RegexTokenizer()


def _content_tokens(s: str) -> set[str]:
    return {t for t in _tok.tokenize(s) if len(t) >= 2 and not t[0].isdigit()}


@dataclass
class RetrievalResult:
    hits: list[ScoredChunk]
    plan: QueryPlan
    used_filter: dict
    timings: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, embedder: Embedder, index: VectorIndex, docstore: DocStore, bm25: BM25Encoder,
                 cfg: RetrievalCfg, analyzer: QueryAnalyzer | None = None, reranker: Reranker | None = None):
        self.embedder, self.index, self.docs, self.bm25 = embedder, index, docstore, bm25
        self.cfg = cfg
        self.analyzer = analyzer or QueryAnalyzer(latest_only=cfg.latest_only, multi_facet=cfg.multi_facet)
        self.reranker = reranker or NoopReranker(cfg.candidates)
        self._qcache: "OrderedDict[str, object]" = OrderedDict()   # 질의 임베딩 캐시 (같은 질의는 API 재호출 안 함)
        self._qcache_max = 512
        self.refresh_corp_names()

    def refresh_corp_names(self) -> None:
        self.analyzer.set_corp_names(self.docs.corp_names())

    # ------------------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None, filters: dict | None = None,
                 channels: list[str] | None = None) -> RetrievalResult:
        """channels: 이번 검색에 쓸 채널 (None → 설정값). "dense" 가 없으면 질의 임베딩(API 호출)을 건너뛴다."""
        k = k or self.cfg.top_k
        channels = list(channels) if channels else list(self.cfg.channels)
        t0 = time.perf_counter()
        plan = self.analyzer.analyze(question, filters)
        qv = self._query_vectors(plan, use_dense="dense" in channels or "m3" in channels)
        t1 = time.perf_counter()

        n_cand = max(self.reranker.candidates if self.reranker.enabled else self.cfg.candidates, k)
        facets = plan.facets or [{}]
        per = max(n_cand // len(facets), k)
        groups: list[list[ScoredChunk]] = []
        used_filters = []
        for fc in facets:
            base = plan.filters if not fc else type(plan.filters)({**plan.filters.must, **fc})
            hits, used = [], base
            for flt in plan.filter_ladder(base):
                hits = self.index.search(qv, flt, limit=per, prefetch_limit=max(self.cfg.prefetch_limit, per),
                                         channels=channels)
                used = flt
                if hits:
                    break
            used_filters.append(used.must)
            chunks = self.docs.get_many([h.chunk_id for h in hits])
            groups.append([ScoredChunk(chunks[h.chunk_id], h.score,
                                       {"rrf": h.score, **{f"rank_{c}": r for c, r in h.ranks.items()}})
                           for h in hits if h.chunk_id in chunks])
        t2 = time.perf_counter()

        ranked_groups = [self._rank(question, plan, g) for g in groups]
        ranked = self._interleave(ranked_groups, k)
        t3 = time.perf_counter()

        ranked = self._expand(question, ranked, narrative=bool({"outlook", "cause"} & set(plan.intents)))
        t4 = time.perf_counter()
        used_out = used_filters[0] if len(used_filters) == 1 else {"facets": used_filters}
        return RetrievalResult(ranked, plan, used_out, {
            "encode": t1 - t0, "search": t2 - t1, "rank": t3 - t2, "expand": t4 - t3})

    # ------------------------------------------------------------------
    def _query_vectors(self, plan: QueryPlan, use_dense: bool = True) -> QueryVectors:
        bm25 = self.bm25.encode_query(plan.lexical_text)
        if not use_dense:
            return QueryVectors(dense=None, m3=None, bm25=bm25)
        key = plan.dense_text
        eb = self._qcache.get(key)
        if eb is None:
            eb = self.embedder.embed_query(key)
            self._qcache[key] = eb
            if len(self._qcache) > self._qcache_max:
                self._qcache.popitem(last=False)
        else:
            self._qcache.move_to_end(key)
        return QueryVectors(dense=eb.dense[0], m3=eb.sparse[0] if eb.sparse else None, bm25=bm25)

    def _features(self, question: str, plan: QueryPlan, c: Chunk, q_toks: set[str]) -> dict[str, float]:
        path = c.meta.get("section_path", "")
        title_toks = set(_tok.tokenize(path))
        return {
            "section_hint": float(any(h in path for h in plan.section_hints)),
            "title_overlap": len(q_toks & title_toks) / len(q_toks) if q_toks else 0.0,
            "is_kv": float(c.kind == "table_kv"),
            "is_short": float(len(c.raw) < 80),
            "is_footnote": float("주석" in c.meta.get("section_title", "") and "주석" not in question),
        }

    def _rank(self, question: str, plan: QueryPlan, cands: list[ScoredChunk]) -> list[ScoredChunk]:
        q_toks = _content_tokens(question)
        if self.reranker.enabled and cands:
            # 리랭커가 문맥을 판단하므로 규칙 가산은 약하게 (덧셈)
            rr = self.reranker.score(question, [c.chunk.text for c in cands])
            w = self.cfg.boosts_after_rerank
            for c, s in zip(cands, rr):
                f = self._features(question, plan, c.chunk, q_toks)
                c.score = s + w.get("section_hint", 0) * f["section_hint"] + w.get("title_overlap", 0) * f["title_overlap"] \
                    - w.get("footnote_penalty", 0) * f["is_footnote"]
                c.why.update(rerank=s, **f)
        else:
            # 리랭커 없음: RRF 점수에 규칙 가산 (곱셈, 기존 store.py 로직을 설정값으로)
            w = self.cfg.boosts
            for c in cands:
                f = self._features(question, plan, c.chunk, q_toks)
                s = c.score
                s *= 1 + w.get("section_hint", 0) * f["section_hint"]
                s *= 1 + w.get("title_overlap", 0) * f["title_overlap"]
                s *= 1 + w.get("table_kv", 0) * f["is_kv"]
                s *= w.get("short_penalty", 1) if f["is_short"] else 1
                s *= 1 - w.get("footnote_penalty", 0) * f["is_footnote"]
                c.score = s
                c.why.update(f)
        return sorted(cands, key=lambda c: -c.score)

    @staticmethod
    def _interleave(groups: list[list[ScoredChunk]], k: int) -> list[ScoredChunk]:
        """facet 별 순위 목록을 번갈아 뽑아 k 개 (비교 질문에서 한쪽이 독식하지 않게)."""
        if len(groups) == 1:
            return groups[0][:k]
        out, seen = [], set()
        for row in itertools.zip_longest(*groups):
            for h in row:
                if h is not None and h.chunk.id not in seen:
                    out.append(h)
                    seen.add(h.chunk.id)
                    if len(out) == k:
                        return out
        return out

    def _expand(self, question: str, hits: list[ScoredChunk], narrative: bool = False) -> list[ScoredChunk]:
        """small-to-big: (1) 상위 섹션의 표 KV 동반 (2) KV → 원본 표 (3) 선택: 섹션 본문."""
        cfg = self.cfg
        seen = {h.chunk.id for h in hits}
        out = list(hits)
        qt = set(_tok.tokenize(question))

        parents: list[str] = []
        for h in hits:
            p = h.chunk.meta.get("parent_id")
            if p and p not in parents:
                parents.append(p)
        for h_parent in parents[:cfg.sibling_kv_parents]:
            base = next(h.score for h in hits if h.chunk.meta.get("parent_id") == h_parent)
            kvs = [c for c in self.docs.by_parent(h_parent, "table_kv") if c.id not in seen]
            kvs.sort(key=lambda c: -len(qt & set(_tok.tokenize(c.raw))))
            for c in kvs[:cfg.sibling_kv_per_parent]:
                out.append(ScoredChunk(c, base * 0.95, {"expanded": 1.0}))
                seen.add(c.id)
            n_text = max(cfg.section_text_chunks, cfg.narrative_text_chunks if narrative else 0)
            if n_text:
                for c in [c for c in self.docs.by_parent(h_parent, "text") if c.id not in seen][:n_text]:
                    out.append(ScoredChunk(c, base * 0.9, {"expanded": 1.0}))
                    seen.add(c.id)

        if cfg.table_for_kv:
            for h in list(out):
                m = h.chunk.meta
                if h.chunk.kind != "table_kv":
                    continue
                t = self.docs.table_group(m.get("parent_id"), m.get("table_index"), m.get("row_start"), m.get("col_group", 0))
                if t and t.id not in seen:
                    out.append(ScoredChunk(t, h.score * 0.99, {"expanded": 1.0}))
                    seen.add(t.id)
        return out
