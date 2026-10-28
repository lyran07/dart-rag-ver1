"""보고서 생성 테스트 (합성 샘플 + 가짜 LLM → API 비용 0원)."""
from __future__ import annotations

import json
import re
import threading

from tests.conftest import build_sample

from dart_rag.agent.llm import StepResult
from dart_rag.report import ReportBuilder, to_markdown
from dart_rag.report.builder import unverified_table_numbers
from dart_rag.report.templates import ReportTemplate, SectionSpec
from dart_rag.service import RAGService

TPL = {"t": ReportTemplate("t", "테스트 보고서", [
    SectionSpec("sum", "한눈에 보기", "종합한다", summary=True),
    SectionSpec("fin", "실적", "요약재무를 정리한다", routes=["summary_financials"], table=True),
    SectionSpec("sales", "매출", "매출을 정리한다", routes=["sales_orders"]),
    SectionSpec("none", "없는 섹션", "없다", routes=["audit"]),
    SectionSpec("mg", "경영진", "사업보고서만", routes=["mdna"], report_types=["사업보고서"]),
])}


class EchoLLM:
    """[근거] 의 첫 블록 id 와 그 안의 첫 숫자를 인용해 답한다. 표에는 원문 숫자 1개 + 만든 숫자 1개."""
    backend, model, reasoning = "fake", "fake-luna", "none"

    def __init__(self):
        self.prompts, self.lock = [], threading.Lock()

    def start(self, instructions, user):
        with self.lock:
            self.prompts.append((instructions, user))
        return {"i": instructions, "u": user}

    def step(self, conv, tools, schema, allow_tools=True):
        u = conv["u"]
        m = re.search(r"<근거 id=([^>]+)>\n출처: [^\n]*\n(.*?)</근거>", u, re.S)
        if m is None:   # 한눈에 보기
            out = {"headline": "종합 특징", "overview": "요약 서술", "table_markdown": "",
                   "interpretation": ["점검 1", "점검 2", "점검 3", "넘침"], "watch": ["a", "b"], "evidence": []}
        else:
            num = (re.findall(r"\d{1,3}(?:,\d{3})+", m.group(2)) or ["0"])[0]
            out = {"headline": "특징 한 줄", "overview": f"값은 {num} [1]",
                   "table_markdown": "" if "표가 꼭 필요하지 않다" in u
                   else f"단위: 백만원\n| 항목 | 값 |\n|---|---|\n| 원문 | {num} |\n| 만든값 | 987,654,321 |",
                   "interpretation": ["약 3.3배 [1]"], "watch": [],
                   "evidence": [{"chunk_id": m.group(1), "quote": f"원문 숫자 {num}"}]}
        return StepResult([], json.dumps(out, ensure_ascii=False),
                          {"input_tokens": len(u), "output_tokens": 50, "usd": 0.001}, 0.0)


def _builder(tmp_path, llm=None):
    s = build_sample(tmp_path)
    svc = RAGService(s)
    return svc, ReportBuilder(svc, llm=llm or EchoLLM(), templates=TPL)


def test_report_sections_compare_and_cache(tmp_path):
    svc, rb = _builder(tmp_path)
    assert rb.default_compare("삼성전자", "2024.09") is None          # 2023.09 없음
    job = rb.run("삼성전자", "2024.09", "2023.12", "t")
    st = {s.id: s for s in job.sections}
    assert job.status == "done" and not job.from_cache
    assert st["fin"].status == "done" and st["fin"].sources[0]["verified"]
    assert st["fin"].table_markdown and st["fin"].check["table_unverified_numbers"] == ["987,654,321"]
    assert st["none"].status == "no_evidence" and st["none"].diag["pages_found"] == 0
    assert "mg" not in st                                              # 분기보고서 → 사업보고서 전용 섹션 제외
    assert st["sum"].status == "done" and st["sum"].kind == "summary"
    assert len(st["sum"].interpretation) == 3 and st["sum"].watch == ["a"]   # 분량 상한
    # 비교 공시 두 기간의 위키가 모두 근거로 들어갔는지
    fin_prompt = next(u for i, u in rb._llm.prompts if "보고서 섹션: 실적" in u)
    assert "(2024.09)" in fin_prompt and "(2023.12)" in fin_prompt and "비교: " in fin_prompt
    # 종합 의견은 다른 섹션 뒤에, 다른 섹션 결과만 읽는다
    assert "[섹션 요약]" in rb._llm.prompts[-1][1] and "## 실적" in rb._llm.prompts[-1][1]
    assert job.usage["usd"] == 0.003
    md = to_markdown(job)
    assert "987,654,321" in md and "원문에서 찾지 못한 숫자" in md
    assert md.index("## 한눈에 보기") < md.index("## 실적")             # 한눈에 보기가 맨 앞
    assert "## 없는 섹션" not in md and "생략한 섹션: 없는 섹션" in md     # 빈 섹션은 빠지고 맨 끝에 표시

    # 같은 조건 → 저장본 (LLM 호출 없음)
    n = len(rb._llm.prompts)
    again = rb.run("삼성전자", "2024.09", "2023.12", "t")
    assert again.from_cache and len(rb._llm.prompts) == n and again.sections[1].overview == job.sections[1].overview
    assert rb.has_cached("삼성전자", "2024.09", "2023.12", "t")
    assert not rb.has_cached("삼성전자", "2024.09", None, "t")

    # 프롬프트·템플릿이 바뀌면 버전이 달라져 저장본을 쓰지 않는다
    TPL["t"].sections[0].instruction = "바뀐 지시"
    try:
        assert not rb.has_cached("삼성전자", "2024.09", "2023.12", "t")
    finally:
        TPL["t"].sections[0].instruction = "요약재무를 정리한다"


def test_report_validation(tmp_path):
    _, rb = _builder(tmp_path)
    for args in [("없는회사", "2024.09", None, "t"), ("삼성전자", "2020.06", None, "t"),
                 ("삼성전자", "2024.09", "2024.09", "t"), ("삼성전자", "2024.09", None, "없는템플릿")]:
        try:
            rb.run(*args)
            raise AssertionError(f"ValueError 가 나야 함: {args}")
        except ValueError:
            pass


def test_section_error_is_isolated_and_not_cached(tmp_path):
    class Boom(EchoLLM):
        def step(self, conv, tools, schema, allow_tools=True):
            if "보고서 섹션: 매출" in conv["u"]:
                raise TimeoutError("느림")
            return super().step(conv, tools, schema, allow_tools)
    _, rb = _builder(tmp_path, Boom())
    job = rb.run("삼성전자", "2024.09", None, "t")
    st = {s.id: s for s in job.sections}
    assert st["sales"].status == "error" and "TimeoutError" in st["sales"].error
    assert st["fin"].status == "done" and st["sum"].status == "done"
    assert not rb.has_cached("삼성전자", "2024.09", None, "t")          # 오류 섹션이 있으면 저장 안 함


def test_unverified_table_numbers():
    src = ["매출액 258,935,494 백만원, 영업이익 6,566,976"]
    t = "| 항목 | 값 |\n|---|---|\n| 매출액 | 258935494 |\n| 영업이익 | 6,566,976 |\n| 이상 | 1,234 |\n| 기 | 55 |"
    assert unverified_table_numbers(t, src) == ["1,234"]


def test_report_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import dart_rag.api as api
    svc, rb = _builder(tmp_path)
    svc.__dict__["report"] = rb
    monkeypatch.setattr(api, "service", lambda: svc)
    c = TestClient(api.app)
    reps = c.get("/companies/삼성전자/reports").json()["reports"]
    assert {r["period"] for r in reps} == {"2024.09", "2023.12"} and not any(r["cached"] for r in reps)
    j = c.post("/report", json={"corp": "삼성전자", "period": "2024.09", "compare_period": "none",
                                "template": "t"}).json()
    for _ in range(100):
        if j["status"] != "running":
            break
        import time
        time.sleep(0.05)
        j = c.get(f"/report/{j['id']}").json()
    assert j["status"] == "done" and j["progress"] == {"done": 4, "total": 4}      # mg 제외
    md = c.get(f"/report/{j['id']}/markdown")
    assert md.status_code == 200 and "## 실적" in md.text
    sid = next(s for s in j["sections"] if s["id"] == "fin")["sources"][0]["chunk_id"]
    w = c.get(f"/wiki/sections/{sid}").json()
    assert w["corp"] == "삼성전자" and w["text"]
    j2 = c.post("/report", json={"corp": "삼성전자", "period": "2024.09", "compare_period": "none",
                                 "template": "t"}).json()
    assert j2["from_cache"] and j2["status"] == "done"
    assert c.post("/report", json={"corp": "삼성전자", "period": "2024-09"}).status_code == 422
    assert c.post("/report", json={"corp": "삼성전자", "period": "2020.06", "template": "t"}).status_code == 400


def test_truncated_json_is_retried_once(tmp_path):
    class Truncating(EchoLLM):
        cut = {"실적": 1}

        def step(self, conv, tools, schema, allow_tools=True):
            for k in list(self.cut):
                if f"보고서 섹션: {k}" in conv["u"] and self.cut[k] > 0:
                    self.cut[k] -= 1
                    return StepResult([], '{"headline": "잘린 출', {"input_tokens": 1, "output_tokens": 1, "usd": 0.0}, 0.0)
            return super().step(conv, tools, schema, allow_tools)
    _, rb = _builder(tmp_path, Truncating())
    job = rb.run("삼성전자", "2024.09", None, "t")
    fin = next(s for s in job.sections if s.id == "fin")
    assert fin.status == "done" and "재시도" in fin.diag["retried"]


def test_version_ignores_runtime_options(tmp_path):
    _, rb = _builder(tmp_path)
    v, orig = rb.version("t"), TPL["t"].options
    try:
        TPL["t"].options = {**orig, "parallel": 1, "max_output_tokens": 999}
        assert rb.version("t") == v
        TPL["t"].options = {**orig, "page_chars": 100}
        assert rb.version("t") != v
    finally:
        TPL["t"].options = orig


def test_pipeline_report_stage_makes_single_version(tmp_path):
    from dart_rag.pipeline import PipelineRunner, StageContext
    s = build_sample(tmp_path)
    ctx = StageContext(s)
    svc = RAGService(s)
    llm = EchoLLM()
    ctx.report_builder = ReportBuilder(svc, llm=llm, templates={"company_analysis": TPL["t"]})
    rep = PipelineRunner(ctx, ["report"], verbose=False).run(ctx.catalog.filings(latest_only=True))
    assert not rep.failed and rep.processed["report"] == 2
    rb = ctx.report_builder
    assert rb.has_cached("삼성전자", "2024.09", None, "company_analysis")
    assert not rb.has_cached("삼성전자", "2024.09", "2023.12", "company_analysis")   # 비교 버전은 요청 시
    assert all("비교: " not in u for _, u in llm.prompts)
