"""LLM 이 호출하는 tool 4개. 모두 기존 코드를 감싼다 (새 검색 엔진 없음).

  resolve_company     회사명 → DART 등록명·고유번호   (catalog.db + config/agent.yaml 별칭)
  list_filings        회사의 보유 공시 목록            (catalog.db)
  search_disclosures  공시 본문·표 검색                (기존 HybridRetriever)
  get_section         섹션 원문을 원래 순서대로        (data/parsed/ 파싱 결과)

ToolSession 은 질문 1개 동안 유지되며, 조회한 근거(seen)를 모아 인용 검증에 쓴다.
"""
from __future__ import annotations

import difflib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from .llm import ToolCall

REPORT_TYPES = ["사업보고서", "반기보고서", "분기보고서"]

TOOL_SCHEMAS = [
    {"type": "function", "name": "resolve_company",
     "description": "회사 이름(줄임말 가능)을 서비스가 보유한 기업의 DART 등록명과 corp_code 로 확인한다.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["name"],
                    "properties": {"name": {"type": "string", "description": "예: 하이닉스, 현대차"}}}},
    {"type": "function", "name": "list_filings",
     "description": "회사가 보유한 공시(사업·반기·분기보고서) 목록을 최신순으로 돌려준다. 기간 존재 여부 확인용.",
     "parameters": {"type": "object", "additionalProperties": False, "required": ["corp_code"],
                    "properties": {"corp_code": {"type": "string"},
                                   "report_type": {"type": "string", "enum": REPORT_TYPES},
                                   "year": {"type": "string", "description": "예: 2026"}}}},
    {"type": "function", "name": "search_disclosures",
     "description": ("공시 본문과 표에서 관련 부분을 검색한다. query 에는 찾을 내용만 쓰고 회사·기간은 인자로 지정한다. "
                     "결과의 chunk_id 를 답변 evidence 에 인용한다."),
     "parameters": {"type": "object", "additionalProperties": False, "required": ["query", "corp_code"],
                    "properties": {"query": {"type": "string", "description": "예: 연결 매출액 요약재무정보"},
                                   "corp_code": {"type": "string"},
                                   "period": {"type": "string", "description": "보고서 기준월 예: 2026.06"},
                                   "report_type": {"type": "string", "enum": REPORT_TYPES},
                                   "k": {"type": "integer", "minimum": 1, "maximum": 10}}}},
    {"type": "function", "name": "get_section",
     "description": ("공시의 한 섹션을 원문 순서대로(소제목·문단·표) 돌려준다. 표의 연결/별도·당기/전기 구분처럼 "
                     "문맥이 필요할 때 쓴다. 길면 offset 으로 이어 읽는다."),
     "parameters": {"type": "object", "additionalProperties": False, "required": ["rcept_no", "section_path"],
                    "properties": {"rcept_no": {"type": "string"},
                                   "section_path": {"type": "string",
                                                    "description": "검색 결과의 section_path 그대로"},
                                   "offset": {"type": "integer", "minimum": 0}}}},
]


def _norm(s: str) -> str:
    return re.sub(r"\s+|\(주\)|㈜|주식회사", "", (s or "")).lower()


def match_company(name: str, corps: dict[str, str], aliases: dict[str, str]) -> list[dict]:
    """corps: {corp_name: corp_code}. 정확 → 별칭 → 부분 문자열 → 유사도 순으로 찾는다."""
    n = _norm(name)
    by_norm = {_norm(c): c for c in corps}
    al = {_norm(k): v for k, v in (aliases or {}).items()}
    hits: list[str] = []
    if n in by_norm:
        hits = [by_norm[n]]
    elif n in al and al[n] in corps:
        hits = [al[n]]
    elif corps.get(name):
        hits = [name]
    if not hits and n:
        hits = [c for k, c in by_norm.items() if n in k or k in n]
    if not hits and n:
        hits = [by_norm[k] for k in difflib.get_close_matches(n, list(by_norm), n=3, cutoff=0.6)]
    return [{"corp_name": c, "corp_code": corps[c]} for c in dict.fromkeys(hits)][:5]


class ToolSession:
    def __init__(self, service, cfg):
        self.svc, self.cfg = service, cfg
        self.seen: dict[str, dict] = {}          # chunk_id → {text, source, rcept_no, ...}
        self.tool_timings: dict[str, dict] = {}  # call_id → 도구 내부 세부 시간 (LLM 에는 보내지 않음)
        self._filings = self._load_filings()      # 메인 스레드에서 한 번 (catalog 연결은 스레드 공유 불가)
        self.corps = {f.corp_name: f.corp_code for f in self._filings}

    # ------------------------------------------------------------------ 데이터
    def _load_filings(self):
        from ..pipeline.catalog import Catalog
        cat = Catalog(self.svc.settings.storage.root_path / "catalog.db")
        try:
            return cat.filings(latest_only=True)
        finally:
            db = getattr(cat, "db", None)
            if db is not None:
                db.close()

    def _remember(self, cid: str, text: str, **meta) -> None:
        self.seen[cid] = {"text": text, **meta}

    # ------------------------------------------------------------------ tools
    def resolve_company(self, name: str) -> dict:
        c = match_company(name, self.corps, self.cfg.aliases)
        out: dict = {"candidates": c}
        if not c:
            out["note"] = "보유 기업 중 일치하는 회사가 없습니다."
            out["available"] = sorted(self.corps)
        elif len(c) > 1:
            out["note"] = "후보가 여러 개입니다. 사용자에게 확인하세요."
        return out

    def list_filings(self, corp_code: str, report_type: str | None = None, year: str | None = None) -> dict:
        fs = [f for f in self._filings if f.corp_code == corp_code]
        if report_type:
            fs = [f for f in fs if f.report_type == report_type]
        if year:
            fs = [f for f in fs if f.period.startswith(str(year))]
        fs.sort(key=lambda f: (f.period, f.rcept_dt), reverse=True)
        return {"filings": [{"rcept_no": f.rcept_no, "report_nm": f.report_nm, "report_type": f.report_type,
                             "period": f.period, "rcept_dt": f.rcept_dt} for f in fs[:20]],
                "total": len(fs)}

    def search_disclosures(self, query: str, corp_code: str, period: str | None = None,
                           report_type: str | None = None, k: int | None = None) -> dict:
        from ..generation.answerer import source_label
        filters: dict = {"corp_code": corp_code, "is_latest": True}
        if period:
            filters["period"] = period
            filters["year"] = period[:4]
        if report_type:
            filters["report_type"] = report_type
        k = max(1, min(int(k or self.cfg.search_k), 10))
        r = self.svc.retrieve(query, k, filters)
        hits = r.hits
        tables = {(h.chunk.meta.get("parent_id"), h.chunk.meta.get("table_index"), h.chunk.meta.get("row_start"),
                   h.chunk.meta.get("col_group")) for h in hits if h.chunk.kind == "table"}
        out = []
        for h in hits:
            m = h.chunk.meta
            key = (m.get("parent_id"), m.get("table_index"), m.get("row_start"), m.get("col_group"))
            if h.chunk.kind == "table_kv" and key in tables:      # 같은 표의 마크다운이 있으면 kv 는 생략
                continue
            text = h.chunk.raw[: self.cfg.result_chars]
            src = source_label(h.chunk)
            self._remember(h.chunk.id, h.chunk.raw, source=src, rcept_no=m.get("rcept_no"),
                           period=m.get("period"), section=m.get("section_path"), kind=h.chunk.kind)
            out.append({"chunk_id": h.chunk.id, "kind": h.chunk.kind, "rcept_no": m.get("rcept_no"),
                        "period": m.get("period"), "section_path": m.get("section_path"), "text": text})
            if len(out) >= k:
                break
        res: dict = {"results": out, "_timings": {k: round(v, 3) for k, v in r.timings.items()}}
        if not out:
            res["hint"] = "결과 없음. 검색어를 바꾸거나 period·report_type 조건을 빼고 다시 시도하세요."
        return res

    def get_section(self, rcept_no: str, section_path: str, offset: int = 0) -> dict:
        from ..ingestion.parser import DartXmlParser
        from ..pipeline.storage import ArtifactStore
        f = next((x for x in self._filings if x.rcept_no == rcept_no), None)
        if f is None:
            return {"error": f"보유하지 않은 rcept_no: {rcept_no}"}
        root = ArtifactStore(self.svc.settings.storage.root_path).load_parsed(f, DartXmlParser.version)
        secs = [(" > ".join(s.path), s) for s in root.walk() if s.level > 0]
        want = section_path.strip()
        sec = next((s for p, s in secs if p == want), None) \
            or next((s for p, s in secs if p.endswith(want) or want.endswith(p.split(" > ")[-1])), None)
        if sec is None:
            return {"error": "섹션을 찾지 못했습니다.", "available_sections": [p for p, _ in secs][:60]}
        parts = []
        for b in sec.blocks:
            if b.kind == "table" and b.table is not None:
                parts.append(b.table.to_markdown())
            elif b.text:
                parts.append(b.text)
        full = "\n\n".join(parts)
        offset = max(0, int(offset or 0))
        n = self.cfg.section_chars
        text = full[offset: offset + n]
        path = " > ".join(sec.path)
        cid = f"sec:{rcept_no}:{path}#{offset}"
        src = f"{f.corp_name} {f.report_type}({f.period}) · {path}"
        self._remember(cid, text, source=src, rcept_no=rcept_no, period=f.period, section=path, kind="section")
        out = {"chunk_id": cid, "section_path": path, "text": text, "total_chars": len(full),
               "subsections": [c.title for c in sec.children]}
        if offset + n < len(full):
            out["next_offset"] = offset + n
        return out

    # ------------------------------------------------------------------ 실행
    def call(self, tc: ToolCall) -> str:
        fn = {"resolve_company": self.resolve_company, "list_filings": self.list_filings,
              "search_disclosures": self.search_disclosures, "get_section": self.get_section}.get(tc.name)
        if fn is None:
            res: dict = {"error": f"알 수 없는 도구: {tc.name}"}
        elif "__parse_error__" in tc.arguments:
            res = {"error": tc.arguments["__parse_error__"]}
        else:
            try:
                res = fn(**tc.arguments)
            except TypeError as e:
                res = {"error": f"인자 오류: {e}"}
            except Exception as e:  # noqa: BLE001  도구 오류는 LLM 에게 돌려줘 다시 시도하게 한다
                res = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        if isinstance(res, dict) and "_timings" in res:
            self.tool_timings[tc.call_id] = res.pop("_timings")
        s = json.dumps(res, ensure_ascii=False)
        lim = self.cfg.tool_output_chars
        return s if len(s) <= lim else s[:lim] + '…(출력 길이 제한으로 잘림)"'

    def run_parallel(self, calls: list[ToolCall], timeout: float) -> list[tuple[ToolCall, str, float]]:
        import time
        if len(calls) == 1:
            t = time.perf_counter()
            return [(calls[0], self.call(calls[0]), time.perf_counter() - t)]
        out: list = []
        with ThreadPoolExecutor(max_workers=min(4, len(calls))) as ex:
            t = time.perf_counter()
            futs = [(c, ex.submit(self.call, c)) for c in calls]
            for c, fu in futs:
                try:
                    out.append((c, fu.result(timeout=timeout), time.perf_counter() - t))
                except FutureTimeout:
                    out.append((c, json.dumps({"error": f"도구 실행 시간 초과({timeout}s)"}, ensure_ascii=False),
                                time.perf_counter() - t))
        return out
