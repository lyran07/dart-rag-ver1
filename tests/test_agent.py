"""에이전트 루프 테스트 (가짜 LLM · 가짜 도구 → API 비용 0원)."""
from __future__ import annotations

import json

import pytest

from dart_rag.agent.config import AgentConfig
from dart_rag.agent.llm import StepResult, ToolCall, parse_final_json
from dart_rag.agent.loop import AgentError, AgentLoop, verify_evidence
from dart_rag.agent.tools import match_company

CORPS = {"삼성전자": "00126380", "SK하이닉스": "00164779", "현대자동차": "00164742"}
KV = "[SK하이닉스 | 반기보고서 2026.06 | 요약재무정보]\n매출액(제79기 반기)=131,895,033 백만원"


def final(conclusion="연결 매출액은 131,895,033백만원 [1]", cid="c1", quote="매출액(제79기 반기)=131,895,033", clar=False):
    return json.dumps({"conclusion": conclusion, "key_points": ["연결 매출액 131,895,033백만원 [1]"],
                       "interpretation": [], "evidence": [{"chunk_id": cid, "quote": quote}] if cid else [],
                       "needs_clarification": clar}, ensure_ascii=False)


class FakeLLM:
    backend, model, reasoning = "fake", "fake-model", "low"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def start(self, instructions, user_text):
        return {"msgs": [user_text]}

    def add_user(self, conv, text):
        conv["msgs"].append(text)

    def add_tool_results(self, conv, results):
        conv["msgs"] += [out for _, out in results]

    def step(self, conv, tools, schema, allow_tools=True):
        self.calls.append(allow_tools)
        item = self.script.pop(0)
        u = {"input_tokens": 100, "cached_tokens": 50, "output_tokens": 10, "reasoning_tokens": 0, "usd": 0.00001}
        if isinstance(item, list):
            return StepResult(item, None, u, 0.01)
        return StepResult([], item, u, 0.01)


class FakeSession:
    def __init__(self, fail=False):
        self.seen, self.fail, self.ran = {}, fail, []

    def run_parallel(self, calls, timeout):
        out = []
        for c in calls:
            self.ran.append(c.name)
            if self.fail:
                out.append((c, json.dumps({"error": "boom"}), 0.0))
            else:
                self.seen["c1"] = {"text": KV, "source": "SK하이닉스 반기보고서(2026.06) · 요약재무정보",
                                   "rcept_no": "20260814003509", "period": "2026.06", "section": "요약재무정보",
                                   "kind": "table_kv"}
                out.append((c, json.dumps({"results": [{"chunk_id": "c1", "text": KV}]}, ensure_ascii=False), 0.0))
        return out


def tc(name="search_disclosures", **args):
    return ToolCall(f"id-{name}", name, args or {"query": "연결 매출액", "corp_code": "00164779"})


def make(script, cfg=None, session=None):
    s = session or FakeSession()
    return AgentLoop(FakeLLM(script), lambda: s, cfg or AgentConfig(), today="2026-10-27"), s


def test_tool_then_answer_verified():
    loop, s = make([[tc()], final()])
    a = loop.run("SK하이닉스 2026년 반기 연결 매출액은?")
    assert a.status == "answered" and s.ran == ["search_disclosures"]
    assert a.sources[0]["verified"] and a.plan["evidence_check"]["n_unverified"] == 0
    assert "131,895,033" in a.answer and a.plan["rounds"] == 2
    assert a.plan["usage"]["input_tokens"] == 200


def test_number_not_in_source_is_flagged():
    loop, _ = make([[tc()], final(quote="매출액=131,895,000")])
    a = loop.run("q")
    assert not a.sources[0]["verified"] and "원문에 없는 숫자" in a.sources[0]["problems"][0]
    assert "검증되지 않았습니다" in a.answer


def test_unseen_chunk_is_flagged():
    srcs, chk = verify_evidence([{"chunk_id": "nope", "quote": "x"}], {})
    assert chk["n_unverified"] == 1 and srcs[0]["problems"] == ["조회하지 않은 chunk_id"]


def test_round_limit_forces_final_without_tools():
    cfg = AgentConfig(max_rounds=2)
    loop, _ = make([[tc()], [tc()], final()], cfg)
    a = loop.run("q")
    assert loop.llm.calls == [True, True, False]          # 마지막은 도구 없이
    assert a.plan["trace"][-1]["round"] == "final(forced)"


def test_tool_error_is_returned_to_llm_and_loop_continues():
    loop, _ = make([[tc()], final(cid=None, conclusion="찾지 못했습니다")], session=FakeSession(fail=True))
    a = loop.run("q")
    assert a.status == "no_evidence" and a.plan["trace"][0]["tools"][0]["error"]


def test_clarification_status():
    loop, _ = make([final(cid=None, conclusion="어느 회사인가요?", clar=True)])
    assert loop.run("삼성 매출").status == "needs_clarification"


def test_bad_json_raises_agent_error():
    loop, _ = make(["이건 JSON 이 아님"])
    with pytest.raises(AgentError):
        loop.run("q")


def test_parse_final_json_tolerates_fences():
    assert parse_final_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_final_json('답변: {"a": 2} 끝') == {"a": 2}


def test_match_company():
    al = {"하이닉스": "SK하이닉스", "현대차": "현대자동차"}
    assert match_company("하이닉스", CORPS, al)[0]["corp_code"] == "00164779"
    assert match_company("SK 하이닉스", CORPS, al)[0]["corp_name"] == "SK하이닉스"
    assert match_company("현대차", CORPS, al)[0]["corp_name"] == "현대자동차"
    assert match_company("삼성", CORPS, al)[0]["corp_name"] == "삼성전자"       # 부분 일치
    assert match_company("LG전자", CORPS, al) == []
