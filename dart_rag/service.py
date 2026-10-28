"""서비스 조립 (composition root).

설정 → 구현체 선택 → 의존성 주입을 이 파일에서만 한다.
CLI / FastAPI / 평가 스크립트 / 테스트는 모두 RAGService 하나만 쓴다.
"""
from __future__ import annotations

from functools import cached_property

from .config import Settings
from .domain import Answer


class RAGService:
    def __init__(self, settings: Settings | None = None, router=None, reranker=None, embedder=None):
        self.settings = settings or Settings.load()
        self._router, self._reranker, self._embedder = router, reranker, embedder

    # ------------------------------------------------------------------ 구성 요소
    @cached_property
    def embedder(self):
        if self._embedder is not None:
            return self._embedder
        from .embedding import make_embedder
        return make_embedder(self.settings.embedding)

    @cached_property
    def docstore(self):
        from .index import make_docstore
        return make_docstore(self.settings)

    @cached_property
    def index(self):
        from .index import make_index
        e = self.embedder
        return make_index(self.settings, e.key, e.dim, e.supports_sparse)

    @cached_property
    def router(self):
        if self._router is not None:
            return self._router
        from .llm import LLMRouter
        return LLMRouter.from_file(self.settings.generation.providers_file)

    @cached_property
    def retriever(self):
        from .index import make_bm25
        from .retrieval import HybridRetriever, QueryAnalyzer, make_reranker
        rc = self.settings.retrieval
        rr = self._reranker if self._reranker is not None else make_reranker(self.settings.rerank, rc.candidates)
        return HybridRetriever(self.embedder, self.index, self.docstore, make_bm25(self.settings), rc,
                               QueryAnalyzer(latest_only=rc.latest_only, multi_facet=rc.multi_facet,
                                             aliases=self.agent_config.aliases), rr)

    @cached_property
    def wiki(self):
        from .wiki import WikiBuilder
        return WikiBuilder(self.settings.storage.root_path / "wiki")

    @cached_property
    def agent_config(self):
        from .agent import load_agent_config
        return load_agent_config()

    @cached_property
    def agent(self):
        from .agent import AgentLoop
        from .agent.llm import make_agent_llm
        from .agent.tools import ToolSession
        cfg = self.agent_config
        return AgentLoop(make_agent_llm(cfg), lambda: ToolSession(self, cfg), cfg)

    @cached_property
    def report(self):
        """기업분석 보고서 생성기 (공시 1건 기준, data/reports/ 저장본 우선)."""
        from .report import ReportBuilder
        return ReportBuilder(self)

    # ------------------------------------------------------------------ auto 모드 (라우팅)
    @cached_property
    def filings(self):
        from .pipeline.catalog import Catalog
        cat = Catalog(self.settings.storage.root_path / "catalog.db")
        try:
            return cat.filings(latest_only=True)
        finally:
            cat.db.close()

    @cached_property
    def routes(self):
        from .routing import load_routes
        return load_routes()

    @cached_property
    def rule_router(self):
        from .routing import RuleRouter
        return RuleRouter(self.routes, self.filings, self.agent_config.aliases)

    def make_llm_router(self, name: str):
        """name: gemini (chat 백엔드) | luna (responses 백엔드, reasoning none)"""
        from .agent.llm import make_llm
        from .routing import LLMQueryRouter
        if name == "gemini":
            llm = make_llm(self.agent_config, backend="chat", max_output_tokens=400, timeout=8)
        elif name == "luna":
            llm = make_llm(self.agent_config, backend="responses", reasoning="none", max_output_tokens=400, timeout=8)
        else:
            raise ValueError(f"알 수 없는 라우터: {name} (gemini | luna | rules)")
        return LLMQueryRouter(llm, self.routes, self.filings, self.agent_config.aliases, name=name)

    @cached_property
    def auto(self):
        from .agent.llm import make_llm
        from .routing.auto import AutoAnswerer
        a = self.agent_config.auto or {}
        name = a.get("router", "gemini")
        llm_router = None
        if name != "rules":
            try:
                llm_router = self.make_llm_router(name)
            except Exception:  # noqa: BLE001  키가 없으면 규칙 라우터만 사용
                llm_router = None
        answer = make_llm(self.agent_config, backend=a.get("answer_backend", "responses"),
                          reasoning=a.get("answer_reasoning", "none"),
                          max_output_tokens=int(a.get("answer_max_output_tokens", 1200)))
        return AutoAnswerer(self, self.agent_config, self.rule_router, llm_router, answer)

    def warmup(self, verbose: bool = True) -> dict:
        """서버 시작 시 1회: 인덱스·BM25·DB·위키 로드 + 질의 임베딩 1회 (첫 사용자 대기 제거)."""
        import time as _t
        out: dict = {}

        def step(name, fn):
            if verbose:
                print(f"[warmup] {name} …", flush=True)
            t = _t.perf_counter()
            try:
                fn()
                out[name] = round(_t.perf_counter() - t, 2)
            except Exception as e:  # noqa: BLE001  한 단계 실패가 나머지를 막지 않게
                out[name + "_error"] = f"{type(e).__name__}: {str(e)[:150]}"
            if verbose:
                print(f"[warmup] {name} {out.get(name, out.get(name + '_error'))}", flush=True)

        step("index", lambda: self.retrieve("워밍업", 1, channels=["bm25"]))   # 인덱스·BM25 로드 (API 없음)
        step("query_embedding", lambda: self.embedder.embed_query("워밍업"))  # API 연결 (타임아웃 10초)
        step("wiki", lambda: (self.wiki.index, self.filings, self.routes))
        if self.agent_config.default_mode == "auto":
            step("auto", lambda: self.auto)
        return out

    @cached_property
    def generator(self):
        from .generation import AnswerGenerator
        return AnswerGenerator(self.router, self.settings.generation)

    # ------------------------------------------------------------------ 사용
    def retrieve(self, question: str, k: int | None = None, filters: dict | None = None,
                 channels: list[str] | None = None):
        return self.retriever.retrieve(question, k, filters, channels)

    def ask(self, question: str, k: int | None = None, filters: dict | None = None,
            mode: str | None = None) -> Answer:
        """mode: auto (라우터가 wiki·hybrid·vector·agent 선택) | agent (tool calling 에이전트)
        | rag (검색 1회 + LLM 1회). None 이면 config/agent.yaml 기본값.
        filters 를 직접 준 요청은 rag 로 처리한다 (에이전트는 조건을 스스로 정한다)."""
        mode = mode or self.agent_config.default_mode
        if mode == "auto" and not filters:
            try:
                return self.auto.ask(question)
            except Exception as e:  # noqa: BLE001  auto 실패 → 기존 방식으로 1회
                if self.agent_config.fallback != "rag":
                    raise
                ans = self.ask_rag(question, k, filters)
                ans.plan["auto_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                ans.plan["mode"] = "rag (auto fallback)"
                return ans
        if mode == "agent" and not filters:
            try:
                return self.agent.run(question)
            except Exception as e:  # noqa: BLE001  에이전트 실패 → 설정에 따라 기존 방식으로 1회
                if self.agent_config.fallback != "rag":
                    raise
                ans = self.ask_rag(question, k, filters)
                ans.plan["agent_error"] = f"{type(e).__name__}: {str(e)[:300]}"
                ans.plan["mode"] = "rag (agent fallback)"
                return ans
        return self.ask_rag(question, k, filters)

    def ask_rag(self, question: str, k: int | None = None, filters: dict | None = None) -> Answer:
        ans = self.generator.answer(question, self.retrieve(question, k, filters))
        ans.plan.setdefault("mode", "rag")
        return ans

    def health(self) -> dict:
        from .embedding import gpu_info
        return {
            "embedder": self.embedder.describe(),
            "index": self.index.describe(),
            "docstore": self.docstore.describe(),
            "docstore_chunks": self.docstore.count(),
            "reranker": self.retriever.reranker.name,
            "llm": self.router.available,
            "gpu": gpu_info(),
        }
