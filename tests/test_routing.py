"""경로 선택 · auto 모드 테스트 (가짜 LLM → API 비용 0원)."""
from __future__ import annotations

import json
import time
import types
from datetime import date
from pathlib import Path

from dart_rag.agent.config import AgentConfig
from dart_rag.agent.llm import StepResult
from dart_rag.domain import Answer, Filing
from dart_rag.routing import LLMQueryRouter, RuleRouter
from dart_rag.routing.auto import AutoAnswerer, select_relevant
from dart_rag.routing.resolve import resolve_periods
from dart_rag.routing.routes import RouteTable, WikiRoute
from dart_rag.wiki import WikiBuilder

TODAY = date(2026, 10, 27)
FILINGS = [Filing(f"r{c}{p}", code, c, f"{t} ({p})", p.replace(".", "") + "14")
           for c, code in [("SK하이닉스", "00164779"), ("삼성전자", "00126380")]
           for p, t in [("2026.06", "반기보고서"), ("2026.03", "분기보고서"), ("2025.09", "분기보고서"),
                        ("2025.06", "반기보고서"), ("2024.12", "사업보고서")]]
ROUTES = RouteTable([WikiRoute("summary_financials", "요약재무정보", ["매출", "영업이익"], ["1. 요약재무정보"]),
                     WikiRoute("dividends", "배당", ["배당"], ["6. 배당에 관한 사항"])])
ALIASES = {"하이닉스": "SK하이닉스", "삼성": "삼성전자"}


def rules():
    return RuleRouter(ROUTES, FILINGS, ALIASES, today=TODAY)


def test_rule_router_routes():
    r = rules()
    d = r.decide("하이닉스 작년 반기랑 올해 반기 매출 비교")
    assert d.route == "wiki" and d.sections == ["summary_financials"]
    assert d.periods["SK하이닉스"] == ["2025.06", "2026.06"]
    assert r.decide("하이닉스 매출이 왜 늘었어").route == "vector"
    assert r.decide("SK하이닉스 HBM 관련 내용").route == "hybrid"
    assert r.decide("LG전자 매출").route == "agent"
    d = r.decide("삼성전자와 SK하이닉스 영업이익")
    assert d.route == "wiki" and [c["corp_name"] for c in d.companies] == ["삼성전자", "SK하이닉스"]


def test_resolve_periods():
    fs = [f for f in FILINGS if f.corp_name == "SK하이닉스"]
    assert resolve_periods("매출", fs, TODAY) == ["2026.06"]                    # 단서 없음 → 최신
    assert resolve_periods("2025년 3분기 매출", fs, TODAY) == ["2025.09"]
    assert resolve_periods("작년 사업보고서", fs, TODAY) == []                     # 2025.12 없음
    assert resolve_periods("최근 3개 분기 추이", fs, TODAY) == ["2026.06", "2026.03", "2025.09"]


class FakeLLM:
    backend, model, reasoning = "fake", "fake", "none"

    def __init__(self, outputs, delay=0.0):
        self.outputs, self.delay, self.calls = list(outputs), delay, 0

    def start(self, instructions, user):
        return {"i": instructions, "u": user}

    def step(self, conv, tools, schema, allow_tools=True):
        self.calls += 1
        time.sleep(self.delay)
        o = self.outputs.pop(0)
        return StepResult([], o if isinstance(o, str) else json.dumps(o, ensure_ascii=False),
                          {"input_tokens": 10, "cached_tokens": 0, "output_tokens": 5, "usd": 0.0}, self.delay)


def test_llm_router_validates_against_data():
    out = {"route": "wiki", "companies": ["SK하이닉스", "LG전자"], "periods": ["2026.06", "2023.06"],
           "sections": ["summary_financials", "없는섹션"], "search_query": "매출", "reason": "요약재무"}
    r = LLMQueryRouter(FakeLLM([out]), ROUTES, FILINGS, ALIASES, today=TODAY, name="fake")
    d = r.decide("하이닉스 매출")
    assert d.route == "wiki" and [c["corp_name"] for c in d.companies] == ["SK하이닉스"]
    assert d.periods["SK하이닉스"] == ["2026.06"] and d.sections == ["summary_financials"]
    assert any("없는 기간" in n for n in d.notes)
    assert "보유 기업과 공시 기간" in r.instructions and "summary_financials" in r.instructions


def test_llm_router_wiki_without_section_falls_back_to_hybrid():
    out = {"route": "wiki", "companies": ["SK하이닉스"], "periods": [], "sections": [], "search_query": "HBM",
           "reason": ""}
    d = LLMQueryRouter(FakeLLM([out]), ROUTES, FILINGS, ALIASES, today=TODAY).decide("q")
    assert d.route == "hybrid" and d.periods["SK하이닉스"] == ["2026.06"]


def test_select_relevant_keeps_matching_parts():
    page = "# 제목\n\n> 출처\n\n" + "\n\n".join(f"임원 {i} 명단 ..." * 20 for i in range(60)) + \
           "\n\n나. 직원 등 현황\n\n| 직원수 | 1인평균 급여 |\n|---|---|\n| 100 | 120,000 |"
    s = select_relevant(page, "직원 수랑 평균 급여", ["평균 급여"], 2000)
    assert "1인평균 급여" in s and "나. 직원 등 현황" in s and len(s) < 2300


class FakeWiki:
    def find_sections(self, corp, period, patterns):
        return [{"id": f"section:{corp}{period}", "corp": corp, "report_type": "반기보고서", "period": period,
                 "title": "III. 재무에 관한 사항 > 1. 요약재무정보", "rcept_no": "r1", "path": "x.md"}]

    def page_text(self, n):
        return f"가. 요약연결재무정보\n(단위: 백만원)\n| 매출액 | 131,895,033 |  ({n['period']})"


def make_auto(answer_outputs, router=None, agent_answer=None, rules_first=True):
    agent = types.SimpleNamespace(run=lambda q: agent_answer or Answer(q, "에이전트 답", [], {"mode": "agent"},
                                                                        "agent:x", {}, status="answered"))
    svc = types.SimpleNamespace(wiki=FakeWiki(), routes=ROUTES, agent=agent, retrieve=None)
    cfg = AgentConfig(auto={"router_timeout_sec": 0.3, "rules_first": rules_first})
    return AutoAnswerer(svc, cfg, rules(), router, FakeLLM(answer_outputs), today=TODAY)


def ok_answer(cid):
    return {"conclusion": "연결 매출액 131,895,033백만원 [1]", "key_points": ["131,895,033백만원 [1]"],
            "interpretation": [], "evidence": [{"chunk_id": cid, "quote": "매출액 | 131,895,033"}],
            "needs_clarification": False}


def test_auto_wiki_path_answers_and_verifies():
    a = make_auto([ok_answer("wiki:SK하이닉스2026.06")]).ask("SK하이닉스 2026년 반기 연결 매출액은?")
    assert a.status == "answered" and a.plan["route"] == "wiki" and a.sources[0]["verified"]
    assert set(a.timings) >= {"route", "gather", "answer", "total"}


def test_auto_escalates_to_agent_when_no_evidence():
    empty = {**ok_answer("x"), "evidence": []}
    a = make_auto([empty]).ask("SK하이닉스 2026년 반기 연결 매출액은?")
    assert a.plan["mode"] == "agent" and a.plan["via"] == "auto" and "escalated" in a.plan


def test_auto_router_timeout_uses_rules():
    slow = FakeLLM([{"route": "agent", "companies": [], "periods": [], "sections": [], "search_query": "",
                     "reason": ""}], delay=1.0)
    router = LLMQueryRouter(slow, ROUTES, FILINGS, ALIASES, today=TODAY)
    a = make_auto([ok_answer("wiki:SK하이닉스2026.06")], router=router, rules_first=False).ask("SK하이닉스 2026년 반기 매출")
    assert a.plan["route"] == "wiki" and any("초과" in n for n in a.plan["router_notes"])


def test_auto_rules_first_skips_llm_router_only_when_rules_are_sure():
    class Boom:
        calls = 0

        def decide(self, q):
            Boom.calls += 1
            raise RuntimeError("부르면 안 됨")
    a = make_auto([ok_answer("wiki:SK하이닉스2026.06")], router=Boom()).ask("SK하이닉스 2026년 반기 매출")
    assert a.plan["route"] == "wiki" and Boom.calls == 0 and any("생략" in n for n in a.plan["router_notes"])
    # 규칙이 wiki 로 확정하지 못하는 질문(동의어)은 LLM 라우터에게 묻는다 (여기서는 실패 → 규칙으로 대체)
    auto = make_auto([ok_answer("x")], router=Boom())
    d, notes = auto.route("하이닉스 2026년 반기 순익은?")
    assert Boom.calls == 1 and any("실패" in n for n in notes)


def test_wiki_paths_are_relative(tmp_path: Path):
    w = WikiBuilder(tmp_path / "wiki")
    assert w.rel(r"C:\Users\x\proj\data\wiki\SK하이닉스\a.md") == "SK하이닉스/a.md"
    assert w.rel(tmp_path / "wiki" / "삼성전자" / "b.md") == "삼성전자/b.md"
    assert w.abs("삼성전자/b.md") == tmp_path / "wiki" / "삼성전자" / "b.md"


def test_auto_retries_once_when_json_truncated():
    class Trunc(FakeLLM):
        def add_user(self, conv, text):
            conv["u"] += "\n" + text
    llm = Trunc(['{"conclusion": "잘린 출', ok_answer("wiki:SK하이닉스2026.06")])
    auto = make_auto([])
    auto.answer_llm = llm
    a = auto.ask("SK하이닉스 2026년 반기 연결 매출액은?")
    assert a.status == "answered" and llm.calls == 2 and a.plan["mode"] == "auto"
