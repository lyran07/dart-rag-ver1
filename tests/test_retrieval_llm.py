from types import SimpleNamespace

from dart_rag.llm import LLMRouter, ProviderSpec
from dart_rag.retrieval import QueryAnalyzer
from dart_rag.service import RAGService

from .conftest import build_sample


def test_query_analyzer_filters_and_expansion():
    qa = QueryAnalyzer(corp_names=["삼성전자", "삼성전자우"])
    p = qa.analyze("2024년 3분기 삼성전자 인건비")
    assert p.filters.must["report_type"] == "분기보고서"
    assert p.filters.must["period"] == "2024.09"
    assert p.filters.must["corp_name"] == "삼성전자"
    assert "연간급여총액" in p.lexical_text and p.dense_text == "2024년 3분기 삼성전자 인건비"
    assert "직원" in " ".join(p.section_hints)
    ladder = [f.must for f in p.filter_ladder()]
    assert "period" not in ladder[1] and "report_type" not in ladder[2]


def test_filter_relaxation(tmp_path):
    s = build_sample(tmp_path)
    r = RAGService(s).retrieve("2019년 사업보고서 배당", k=3)   # 2019 데이터 없음 → 완화
    assert r.hits and "period" not in r.used_filter


def test_small_to_big_adds_table(tmp_path):
    s = build_sample(tmp_path)
    r = RAGService(s).retrieve("2023년 연결 매출액", k=4)
    kinds = [h.chunk.kind for h in r.hits]
    assert "table_kv" in kinds and "table" in kinds


class _FakeErr(Exception):
    def __init__(self, status):
        self.status_code = status


def _client(behaviour):
    def create(**kw):
        b = behaviour.pop(0)
        if isinstance(b, Exception):
            raise b
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=b))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_llm_router_falls_back():
    plans = {"a": [_FakeErr(429), _FakeErr(429)], "b": [_FakeErr(500)], "c": ["답변 [1]"]}
    specs = [ProviderSpec(n, "m", api_key="k", retry_on_429=1, retry_wait=0) for n in "abc"]
    r = LLMRouter(specs, client_factory=lambda s: _client(plans[s.name]), sleep=lambda _: None)
    res = r.complete("sys", lambda budget: "user")
    assert res.provider == "c" and res.text == "답변 [1]"
    assert [a["provider"] for a in res.attempts] == ["a", "a", "b", "c"]


def test_router_from_file_uses_only_available(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    r = LLMRouter.from_file()
    assert [s.split(":")[0] for s in r.available] == ["gemini", "gemini_lite"]   # 같은 키를 쓰는 두 제공자
    monkeypatch.setenv("LLM_ORDER", "groq")
    assert LLMRouter.from_file().available == []


def test_ask_without_llm_uses_extractive_fallback(tmp_path):
    s = build_sample(tmp_path)
    a = RAGService(s, router=LLMRouter([])).ask("2023년 배당금")
    assert a.llm == "none" and a.status == "extractive_fallback" and a.sources
    assert all(src["url"] and src["chunk_id"] for src in a.sources)


def test_ask_strict_filter_returns_no_evidence(tmp_path):
    s = build_sample(tmp_path)
    a = RAGService(s, router=LLMRouter([])).ask("2023년 배당금", filters={"corp_name": "없는회사", "strict": True})
    assert a.status == "no_evidence" and a.sources == []


def test_wiki_built_from_sample(tmp_path):
    s = build_sample(tmp_path)
    svc = RAGService(s)
    corps = svc.docstore.corp_names()
    w = svc.wiki.company(corps[0])
    assert w and w["reports"], "wiki 단계가 기업·보고서 노드를 만들어야 함"
    secs = svc.wiki.sections(w["reports"][0]["rcept_no"])
    assert secs and any(sec["chunk_ids"] for sec in secs)
    assert (s.storage.root_path / "wiki" / "_topics").exists()


def test_ask_with_fake_llm_budget(tmp_path):
    s = build_sample(tmp_path)
    seen = {}

    def create(**kw):
        seen["user"] = kw["messages"][1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    router = LLMRouter([ProviderSpec("tiny", "m", api_key="k", max_ctx_chars=600)], client_factory=lambda s: fake)
    a = RAGService(s, router=router).ask("2023년 배당금")
    assert a.llm == "tiny:m" and len(seen["user"]) < 800


def test_api_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from dart_rag import api
    s = build_sample(tmp_path)
    svc = RAGService(s, router=LLMRouter([]))
    monkeypatch.setattr(api, "service", lambda: svc)
    c = TestClient(api.app)
    assert c.get("/health").json()["docstore_chunks"] > 0
    j = c.post("/search", json={"question": "2023년 배당금", "k": 3}).json()
    assert j["hits"] and "plan" in j
    assert c.post("/ask", json={"question": "2023년 배당금"}).json()["llm"] == "none"


def test_multi_facet_years_and_corps():
    qa = QueryAnalyzer(corp_names=["삼성전자", "SK하이닉스"])
    p = qa.analyze("삼성전자와 SK하이닉스 2023년 대비 2024년 영업이익 변화")
    assert p.filters.must["year"] == ["2023", "2024"] and "period" not in p.filters.must
    assert p.filters.must["corp_name"] == ["삼성전자", "SK하이닉스"]
    assert len(p.facets) == 4 and {"compare", "trend"} <= set(p.intents)
    basic = QueryAnalyzer(corp_names=["삼성전자", "SK하이닉스"], multi_facet=False).analyze(
        "삼성전자와 SK하이닉스 2023년 대비 2024년")
    assert basic.filters.must["corp_name"] == "삼성전자" and basic.filters.must["year"] == "2023" and not basic.facets


def test_intent_hints_for_outlook():
    p = QueryAnalyzer().analyze("반도체 시장 전망은?")
    assert "outlook" in p.intents and "기타 참고사항" in p.section_hints


def test_facet_search_balances_results(tmp_path):
    s = build_sample(tmp_path)
    r = RAGService(s).retrieve("2023년과 2024년 매출 비교", k=4)
    periods = {h.chunk.meta["period"] for h in r.hits if "expanded" not in h.why}
    assert periods == {"2023.12", "2024.09"}
