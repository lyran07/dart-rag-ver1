"""비교·추이 계산 검증 (가짜 LLM → API 비용 0원)."""
from __future__ import annotations

from dart_rag.agent.compare import analyze, check_and_fix, find_issues, parse_value, render_lines
from dart_rag.agent.llm import StepResult
from dart_rag.agent.prompts import ANSWER_SCHEMA

ALIASES = {"현대차": "현대자동차", "하이닉스": "SK하이닉스", "삼성": "삼성전자"}
DIV = [{"subject": "삼성전자", "item": "보통주 주당 현금배당금", "period": "2026.06", "value": "746", "unit": "원"},
       {"subject": "SK하이닉스", "item": "보통주 주당 현금배당금", "period": "2026.06", "value": "750", "unit": "원"},
       {"subject": "현대자동차", "item": "보통주 주당 현금배당금", "period": "2026.06", "value": "5,000", "unit": "원"}]
SEEN = {"a": {"text": "주당 현금배당금 746 / 750 / 5,000"}}


def test_parse_value():
    assert parse_value("5,000") == 5000 and parse_value("27.0%") == 27.0
    assert parse_value("△716") == -716 and parse_value("(4,751,654)") == -4751654 and parse_value("-") is None


def test_compare_ranking_and_issue():
    info = analyze(DIV, [SEEN["a"]["text"]])
    g = info["groups"][0]
    assert g["kind"] == "compare" and g["max"] == "현대자동차" and g["min"] == "삼성전자"
    assert "현대자동차 5,000 > SK하이닉스 750 > 삼성전자 746" in render_lines(info)[0]
    wrong = ["금액은 SK하이닉스가 가장 크고, 삼성전자가 가장 작습니다."]
    issues = find_issues(info, wrong, ALIASES)
    assert len(issues) == 1 and "현대자동차" in issues[0] and "SK하이닉스" in issues[0]
    assert find_issues(info, ["현대차가 5,000원으로 가장 많고 삼성전자가 가장 적다."], ALIASES) == []


def test_trend_change_and_direction_issue():
    rows = [{"subject": "현대자동차", "item": "매출액", "period": "2025.06", "value": "92,694,438", "unit": "백만원"},
            {"subject": "현대자동차", "item": "매출액", "period": "2026.06", "value": "95,154,214", "unit": "백만원"}]
    info = analyze(rows)
    assert info["groups"][0]["kind"] == "trend" and "+2.7%" in render_lines(info)[0]
    assert find_issues(info, ["매출은 전년 반기보다 약 8.1% 감소했다."]) != []
    assert find_issues(info, ["매출은 약 2.7% 증가했다."]) == []


class TwoShotLLM:
    """첫 답은 틀린 결론, 계산 결과를 받으면 고친 답."""
    backend, model = "fake", "fake"

    def __init__(self):
        self.added = []

    def add_user(self, conv, text):
        self.added.append(text)

    def step(self, conv, tools, schema, allow_tools=True):
        fixed = {"conclusion": "현대자동차가 5,000원으로 가장 많고 삼성전자가 746원으로 가장 적다 [1].",
                 "key_points": [], "interpretation": [], "evidence": [{"chunk_id": "a", "quote": "5,000"}],
                 "needs_clarification": False, "comparison": DIV}
        import json
        return StepResult([], json.dumps(fixed, ensure_ascii=False), {"input_tokens": 5, "usd": 0.0001}, 0.5)


def test_check_and_fix_rewrites_once():
    llm = TwoShotLLM()
    data = {"conclusion": "SK하이닉스가 가장 크다 [1].", "key_points": [], "interpretation": [],
            "evidence": [{"chunk_id": "a", "quote": "750"}], "needs_clarification": False, "comparison": DIV}
    out, meta = check_and_fix(llm, object(), data, SEEN, ALIASES, ANSWER_SCHEMA)
    assert meta["retried"] and len(llm.added) == 1 and "현대자동차 5,000" in llm.added[0]
    assert "현대자동차" in out["conclusion"] and out["_calc_issues"] == [] and out["_calc_lines"]


def test_no_comparison_no_change():
    data = {"conclusion": "x", "comparison": []}
    out, meta = check_and_fix(None, None, data, SEEN, ALIASES, ANSWER_SCHEMA)
    assert meta == {} and "_calc_lines" not in out
