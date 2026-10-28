"""HTTP API (FastAPI).  uvicorn dart_rag.api:app --host 0.0.0.0 --port 8080

GET  /health                 구성 상태 (임베더, 인덱스 크기, LLM 목록, GPU)
POST /search                 검색만 (LLM 호출 없음, 평가·디버깅용)
POST /ask                    검색 + 답변  → {status, answer, sources(실제 사용 근거), ...}
GET  /companies              검색 가능한 기업 목록
GET  /companies/{corp}/wiki  기업 Wiki 허브 (보고서 목록) — wiki 단계 실행 후
GET  /filings/{rcept_no}/wiki 보고서 섹션 페이지 목록
GET  /evidence/{chunk_id}    근거 청크 본문·메타 (답변 출처 확인용)

보완 사항: 질문 길이 검증(1~500자), filters 허용 키 제한, strict 옵션, 응답 상태 코드 구분.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .service import RAGService

app = FastAPI(title="DART RAG", version="0.3")
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# ---- 서비스 보호: 접근 토큰(선택) + IP 별 분당 요청 제한 (LLM 호출 API 에만 적용)
ACCESS_TOKEN = os.environ.get("API_ACCESS_TOKEN")          # 설정 시 X-Access-Token 헤더 또는 ?token= 필요
RATE_PER_MIN = int(os.environ.get("ASK_RATE_PER_MIN", "6"))
_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def _guard(request: Request, call_next):
    path = request.url.path
    if path in ("/ask", "/search", "/feedback"):
        if ACCESS_TOKEN and request.headers.get("x-access-token", request.query_params.get("token")) != ACCESS_TOKEN:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "접근 토큰이 필요합니다."}, status_code=401)
        if path == "/ask":
            ip = request.client.host if request.client else "?"
            now, q = time.time(), _hits[ip]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= RATE_PER_MIN:
                from fastapi.responses import JSONResponse
                return JSONResponse({"detail": f"분당 {RATE_PER_MIN}회를 초과했습니다. 잠시 후 다시 시도하세요."},
                                    status_code=429, headers={"Retry-After": "60"})
            q.append(now)
    return await call_next(request)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=str(WEB_DIR)), name="web")

MAX_QUESTION_CHARS = 500
ALLOWED_FILTERS = {"corp_name", "corp_code", "year", "period", "report_type", "rcept_no", "is_latest"}


@lru_cache(maxsize=1)
def service() -> RAGService:
    return RAGService()


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=MAX_QUESTION_CHARS)
    k: int | None = Field(None, ge=1, le=30)
    filters: dict | None = None
    strict: bool = False     # True: 지정 조건이 맞는 공시가 없으면 조건을 완화하지 않고 '근거 없음'
    mode: Literal["auto", "agent", "rag"] | None = None   # None → config/agent.yaml 의 default_mode

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("질문이 비어 있습니다.")
        if len(v) > MAX_QUESTION_CHARS:
            raise ValueError(f"질문은 {MAX_QUESTION_CHARS}자 이내로 입력해 주세요.")
        return v

    @field_validator("filters")
    @classmethod
    def _filters(cls, v: dict | None) -> dict | None:
        if v is None:
            return None
        bad = set(v) - ALLOWED_FILTERS
        if bad:
            raise ValueError(f"허용되지 않는 filters 키: {sorted(bad)} (허용: {sorted(ALLOWED_FILTERS)})")
        return v

    def effective_filters(self) -> dict | None:
        f = dict(self.filters or {})
        if self.strict:
            f["strict"] = True
        return f or None


class Feedback(BaseModel):
    question: str = Field(..., max_length=MAX_QUESTION_CHARS)
    rating: Literal["up", "down"]
    reason: Literal["wrong_number", "wrong_company_period", "no_evidence", "format", "other"] | None = None
    comment: str | None = Field(None, max_length=1000)


@app.on_event("startup")
def _warmup():
    """서버 시작 시 인덱스·위키·라우터를 미리 로드 (WARMUP=0 이면 생략)."""
    if os.environ.get("WARMUP", "1") == "0":
        return
    try:
        print("[warmup]", service().warmup(), flush=True)
    except Exception as e:  # noqa: BLE001  warm-up 실패가 서버 시작을 막지 않게
        print(f"[warmup] 실패: {type(e).__name__}: {e}", flush=True)


@app.get("/health")
def health():
    return service().health()


@app.post("/search")
def search(q: Query):
    r = service().retrieve(q.question, q.k, q.effective_filters())
    return {
        "plan": {**r.plan.to_dict(), "used_filter": r.used_filter},
        "timings": r.timings,
        "hits": [{"chunk_id": h.chunk.id, "kind": h.chunk.kind, "score": round(h.score, 4),
                  "section": h.chunk.meta.get("section_path"), "rcept_no": h.chunk.meta.get("rcept_no"),
                  "period": h.chunk.meta.get("period"), "why": h.why, "text": h.chunk.raw[:500]}
                 for h in r.hits],
    }


@app.post("/ask")
def ask(q: Query):
    a = service().ask(q.question, q.k, q.effective_filters(), mode=q.mode)
    return {"status": a.status, "answer": a.answer, "sources": a.sources, "llm": a.llm,
            "plan": a.plan, "timings": a.timings}


@app.get("/companies")
def companies():
    return {"companies": sorted(service().docstore.corp_names())}


@app.get("/companies/{corp}/wiki")
def company_wiki(corp: str):
    w = service().wiki.company(corp)
    if not w:
        raise HTTPException(404, "Wiki 가 없는 기업입니다 (wiki 단계를 먼저 실행하세요).")
    return w


@app.get("/filings/{rcept_no}/wiki")
def filing_wiki(rcept_no: str):
    secs = service().wiki.sections(rcept_no)
    if not secs:
        raise HTTPException(404, "해당 접수번호의 Wiki 섹션이 없습니다.")
    return {"rcept_no": rcept_no, "sections": [{k: v for k, v in s.items() if k != "path"} for s in secs]}


@app.get("/evidence/{chunk_id}")
def evidence(chunk_id: str):
    c = service().docstore.get_many([chunk_id]).get(chunk_id)
    if not c:
        raise HTTPException(404, "청크를 찾을 수 없습니다.")
    m = c.meta
    return {"chunk_id": c.id, "kind": c.kind, "text": c.raw, "section": m.get("section_path"),
            "corp_name": m.get("corp_name"), "period": m.get("period"), "rcept_no": m.get("rcept_no"),
            "url": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={m.get('rcept_no')}" if m.get("rcept_no") else None}


@app.post("/feedback")
def feedback(f: Feedback):
    """테스터 피드백 기록 (초기: 저장소 루트의 feedback.jsonl 에 append)."""
    import json
    import time
    path = service().settings.storage.root_path / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({**f.model_dump(), "ts": time.time()}, ensure_ascii=False) + "\n")
    return {"ok": True}
