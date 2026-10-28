"""경로 선택기.

RuleRouter     : 규칙 (LLM 없음, ~1ms). LLM 라우터의 실패·시간 초과 대체, 평가 기준선.
LLMQueryRouter : LLM 이 질문을 읽고 경로·회사·기간·섹션을 JSON 하나로 돌려준다 (Gemini Lite 또는 Luna).
                 회사·기간은 코드가 보유 데이터와 대조해 검증한다 (없는 회사·기간은 버림).

경로
  wiki   : 섹션 하나(또는 몇 개)에서 바로 답이 나오는 질문 — 수치·목록·현황. 여러 회사·기간이어도 같은 섹션이면 wiki
  hybrid : 섹션 목록에 없지만 특정 용어·고유명사로 찾을 수 있는 질문 (dense + BM25)
  vector : 의미·관계 질문 — 원인·영향·전략·전망처럼 여러 부분의 내용을 엮어야 하는 질문 (dense)
  agent  : 단계적 조사가 필요한 복합 질문, 회사가 불분명한 질문
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date

from ..agent.llm import parse_final_json
from .resolve import find_companies, resolve_company_names, resolve_periods
from .routes import RouteTable

ROUTES = ("wiki", "hybrid", "vector", "agent")
RELATION_RE = re.compile(r"왜|원인|이유|영향|전망|전략|계획|배경|관계|어떻게\s*될|리스크\s*요인|시사점")


@dataclass
class RouteDecision:
    route: str
    companies: list[dict] = field(default_factory=list)     # [{corp_name, corp_code}]
    periods: dict[str, list[str]] = field(default_factory=dict)   # corp_name → ["2026.06", ...]
    sections: list[str] = field(default_factory=list)       # wiki_routes id
    search_query: str = ""
    reason: str = ""
    router: str = ""
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _Base:
    def __init__(self, routes: RouteTable, filings: list, aliases: dict, today: date | None = None):
        self.routes, self.filings, self.aliases = routes, filings, aliases or {}
        self.today = today or date.today()
        self.corps = {f.corp_name: f.corp_code for f in filings}

    def filings_of(self, corp_code: str) -> list:
        return [f for f in self.filings if f.corp_code == corp_code]

    def periods_for(self, question: str, companies: list[dict]) -> dict[str, list[str]]:
        return {c["corp_name"]: resolve_periods(question, self.filings_of(c["corp_code"]), self.today)
                for c in companies}


class RuleRouter(_Base):
    name = "rules"

    def decide(self, question: str) -> RouteDecision:
        t = time.perf_counter()
        comps = resolve_company_names(find_companies(question, self.corps, self.aliases), self.corps, self.aliases)
        secs = self.routes.match_keywords(question)
        relation = bool(RELATION_RE.search(question))
        if not comps:
            route, why = "agent", "회사를 특정하지 못함 → 에이전트가 확인·되묻기"
        elif relation:
            route, why = "vector", "원인·영향·전망 등 관계 질문"
        elif secs:
            route, why = "wiki", f"섹션 매핑: {', '.join(secs[:3])}"
        else:
            route, why = "hybrid", "매핑표에 없는 주제"
        return RouteDecision(route, comps, self.periods_for(question, comps), secs[:3] if route == "wiki" else [],
                             question, why, self.name, time.perf_counter() - t)


ROUTE_SCHEMA = {
    "type": "json_schema", "name": "route_decision", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
               "required": ["route", "companies", "periods", "sections", "search_query", "reason"],
               "properties": {
                   "route": {"type": "string", "enum": list(ROUTES)},
                   "companies": {"type": "array", "items": {"type": "string"}},
                   "periods": {"type": "array", "items": {"type": "string"}},
                   "sections": {"type": "array", "items": {"type": "string"}},
                   "search_query": {"type": "string"},
                   "reason": {"type": "string"}}}}


class LLMQueryRouter(_Base):
    """llm: agent.llm 의 ResponsesLLM / ChatToolsLLM (tool 없이 JSON 한 번)."""

    def __init__(self, llm, routes: RouteTable, filings: list, aliases: dict, today: date | None = None,
                 name: str = "llm"):
        super().__init__(routes, filings, aliases, today)
        self.llm, self.name = llm, name
        self.instructions = self._instructions()

    def _instructions(self) -> str:
        by_corp: dict[str, list[str]] = {}
        for f in sorted(self.filings, key=lambda f: f.period, reverse=True):
            by_corp.setdefault(f.corp_name, []).append(f"{f.period}({f.report_type})")
        corp_lines = "\n".join(f"- {c}: {', '.join(ps)}" for c, ps in by_corp.items())
        alias_lines = ", ".join(f"{a}→{t}" for a, t in self.aliases.items())
        return f"""당신은 DART 공시 질의응답 서비스의 질문 라우터입니다. 질문을 읽고 답을 찾을 경로를 고릅니다.
직접 답하지 말고 아래 JSON 만 출력합니다.

[경로]
- wiki: 아래 섹션 목록 중 하나(또는 몇 개)에서 바로 답이 나오는 질문 (수치·목록·현황). 여러 회사·여러 기간 비교라도 같은 섹션으로 답할 수 있으면 wiki.
- hybrid: 섹션 목록에는 없지만 특정 용어·고유명사·수치로 찾을 수 있는 질문.
- vector: 원인·영향·전략·전망처럼 여러 부분의 내용을 의미로 엮어야 하는 관계 질문.
- agent: 여러 단계 조사가 필요한 복합 질문, 또는 회사를 특정할 수 없는 질문.

[보유 기업과 공시 기간]
{corp_lines}
별칭: {alias_lines}

[위키 섹션 목록 (sections 에는 이 id 만)]
{self.routes.describe()}

[출력 규칙]
- companies: 위 보유 기업의 정식 이름. 없거나 불분명하면 빈 배열.
- periods: "YYYY.MM" 형식, 위 목록에 있는 기간만. 상대 기간(작년·올해)은 오늘 날짜 기준. 기간 언급이 없으면 빈 배열(=최신).
- sections: route 가 wiki 일 때만 1~3개, 아니면 빈 배열.
- search_query: hybrid·vector 에서 쓸 검색어 (회사명·기간 제외, 찾을 내용만).
- reason: 경로를 고른 이유 한 문장."""

    def decide(self, question: str) -> RouteDecision:
        t = time.perf_counter()
        conv = self.llm.start(self.instructions, f"오늘 날짜: {self.today.isoformat()}\n질문: {question}")
        st = self.llm.step(conv, None, ROUTE_SCHEMA, allow_tools=False)
        d = parse_final_json(st.final_text)
        notes: list[str] = []
        route = d.get("route") if d.get("route") in ROUTES else "agent"
        comps = resolve_company_names([str(x) for x in d.get("companies") or []], self.corps, self.aliases)
        if len(comps) < len(d.get("companies") or []):
            notes.append(f"보유하지 않은/모호한 회사 제외: {d.get('companies')}")
        want = [str(p) for p in d.get("periods") or []]
        periods: dict[str, list[str]] = {}
        for c in comps:
            have = {f.period for f in self.filings_of(c["corp_code"])}
            ok = [p for p in want if p in have]
            if want and len(ok) < len(want):
                notes.append(f"{c['corp_name']}: 없는 기간 제외 {sorted(set(want) - have)}")
            periods[c["corp_name"]] = ok or (sorted(have, reverse=True)[:1] if not want else [])
        known = self.routes.by_id
        secs = [s for s in d.get("sections") or [] if s in known][:3]
        if route == "wiki" and not secs:
            route, notes = "hybrid", notes + ["wiki 섹션이 목록에 없어 hybrid 로 변경"]
        if route in ("wiki", "hybrid", "vector") and not comps:
            route, notes = "agent", notes + ["회사를 확정하지 못해 agent 로 변경"]
        return RouteDecision(route, comps, periods, secs if route == "wiki" else [],
                             str(d.get("search_query") or question), str(d.get("reason") or ""), self.name,
                             time.perf_counter() - t, notes + [f"usage {json.dumps(st.usage)}"])
