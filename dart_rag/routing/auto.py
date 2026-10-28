"""auto 모드: 경로 선택 → 근거 수집 → LLM 답변 1회.

  wiki   : 위키 섹션 페이지를 메타데이터(회사·기간·섹션)로 바로 읽는다. 검색·임베딩 없음.
  hybrid : dense + BM25 검색 (회사·기간 필터)
  vector : dense 검색만
  agent  : 에이전트 루프로 위임
단일 답변이 근거를 하나도 인용하지 못하면 (escalate_to_agent=true 일 때) 에이전트로 넘긴다.
LLM 라우터가 실패하거나 router_timeout_sec 를 넘기면 규칙 라우터 결과로 진행한다.
rules_first(기본 true): 규칙 라우터가 회사·기간·섹션을 모두 잡아 wiki 로 판정하면 LLM 라우터를 부르지 않는다
  (평가셋에서 규칙의 wiki 판정은 전부 맞았음 → 약 1초 단축, Gemini 무료 한도 절약).
  규칙이 hybrid/vector/agent 로 판정한 질문(동의어·개념·복합)만 LLM 라우터에게 묻는다.
"""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import date

from ..agent.compare import check_and_fix
from ..agent.llm import QuotaExhausted, parse_final_json
from ..agent.loop import render, verify_evidence
from ..agent.prompts import ANSWER_SCHEMA
from ..domain import Answer
from .router import RouteDecision

SINGLE_SYSTEM = """당신은 한국 상장회사 DART 공시 분석가입니다. 아래 [근거] 로 주어진 공시 원문만으로 답합니다.
- 수치는 원문 값을 그대로 쓰고 연결/별도, 기간(제N기, 누적·분기), 단위를 함께 적습니다.
  표 바로 위의 소제목(예: "가. 요약연결재무정보", "나. 요약별도재무정보")으로 연결/별도를 구분합니다.
  원문에 단위가 보이지 않으면 추측하지 말고 "단위 미표시"라고 적습니다.
- key_points 에는 원문에서 확인되는 사실만, interpretation 에는 그로부터의 해석만 적습니다.
- 문장 안에서 근거를 evidence 배열 순서 번호 [1], [2] 로 인용합니다. evidence 의 chunk_id 는 [근거] 의 id 를 그대로,
  quote 에는 근거 원문 일부를 숫자까지 그대로 200자 이내로 옮깁니다.
- 근거로 답할 수 없으면 evidence 를 빈 배열로 두고 conclusion 에 무엇이 부족한지 적습니다.
- 여러 회사·기간의 값을 비교하거나 증감을 묻는 질문이면 comparison 에 비교한 값을 원문 그대로 모두 넣습니다
  (subject=회사명, item=항목명, period=기준월 예: 2026.06, value=원문 숫자, unit=단위). 같은 항목은 같은 item·unit 으로 씁니다.
  대소·순위·증감 결론은 값을 크기순으로 정렬해 확인한 뒤 씁니다. 비교가 아닌 질문은 comparison 을 빈 배열로 둡니다.
- 최종 출력은 JSON 객체 하나: {"conclusion", "key_points", "interpretation", "evidence", "needs_clarification", "comparison"}"""


_PARTICLE = re.compile(r"(은|는|이|가|을|를|의|랑|이랑|과|와|도|만|에|에서|으로|로|야|요|해줘|알려줘)$")
_STOP = {"얼마", "알려", "알려줘", "몇", "어떻게", "무엇", "뭐야", "있어", "비교", "올해", "작년", "최근"}


def _terms(question: str, keywords: list[str]) -> list[str]:
    ts = []
    for w in re.findall(r"[가-힣A-Za-z0-9%]{2,}", question):
        w = _PARTICLE.sub("", w)
        if len(w) >= 2 and w not in _STOP:
            ts.append(w)
    ts += [re.sub(r"\s+", "", k) for k in keywords]
    return list(dict.fromkeys(t.lower() for t in ts if t))


def select_relevant(page: str, question: str, keywords: list[str], limit: int) -> str:
    """긴 위키 페이지에서 질문과 관련된 문단·표만 골라 limit 안에 담는다 (원래 순서 유지).
    고른 표 바로 앞 문단(소제목·단위)도 함께 넣는다. 짧은 페이지는 그대로."""
    if len(page) <= limit:
        return page
    parts = page.split("\n\n")
    terms = _terms(question, keywords)
    score = [sum(1 for t in terms if t in re.sub(r"\s+", "", p.lower())) for p in parts]
    keep = {0, 1}                                         # 제목·출처 줄
    for i in sorted(range(len(parts)), key=lambda i: -score[i]):
        if score[i] == 0:
            break
        keep |= {i - 1, i} if i > 0 else {i}
        if sum(min(len(parts[j]), 3000) for j in keep) > limit:
            keep.discard(i)
            break
    out, used = [], 0
    for j in sorted(keep):
        piece = parts[j] if len(parts[j]) <= 3000 else parts[j][:3000] + " …(표 일부 생략)"
        if used + len(piece) > limit:
            break
        out.append(piece)
        used += len(piece)
    return "\n\n".join(out) + f"\n\n(※ 긴 섹션이라 질문 관련 부분만 발췌: {used:,}/{len(page):,}자)"


class AutoAnswerer:
    def __init__(self, service, cfg, rule_router, llm_router, answer_llm, today: date | None = None):
        self.svc, self.cfg = service, cfg
        self.rules, self.llm_router, self.answer_llm = rule_router, llm_router, answer_llm
        self.today = today
        a = cfg.auto or {}
        self.router_timeout = float(a.get("router_timeout_sec", 6))
        self.page_chars = int(a.get("wiki_page_chars", 8000))
        self.ctx_chars = int(a.get("context_chars", 16000))
        self.search_k = int(a.get("search_k", 8))
        self.escalate = bool(a.get("escalate_to_agent", True))
        self.rules_first = bool(a.get("rules_first", True))
        self.cooldown_sec = float(a.get("router_cooldown_sec", 600))
        self._router_off_until = 0.0          # 한도 소진 시 이 시각까지 LLM 라우터 건너뜀
        self._pool = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------ 1. 경로
    def route(self, question: str) -> tuple[RouteDecision, list[str]]:
        notes: list[str] = []
        if self.llm_router is None:
            return self.rules.decide(question), notes
        if self.rules_first:
            rd = self.rules.decide(question)
            if (rd.route == "wiki" and rd.companies and rd.sections
                    and all(rd.periods.get(c["corp_name"]) for c in rd.companies)):
                notes.append("규칙 라우터가 wiki 로 확정 → LLM 라우터 생략")
                return rd, notes
        if time.time() < self._router_off_until:
            notes.append(f"LLM 라우터 한도 소진으로 {int(self._router_off_until - time.time())}s 동안 규칙 라우터 사용")
            return self.rules.decide(question), notes
        fu = self._pool.submit(self.llm_router.decide, question)
        try:
            return fu.result(timeout=self.router_timeout), notes
        except FutureTimeout:
            notes.append(f"LLM 라우터 {self.router_timeout}s 초과 → 규칙 라우터")
        except QuotaExhausted as e:
            self._router_off_until = time.time() + self.cooldown_sec
            notes.append(f"LLM 라우터 한도 소진 → {int(self.cooldown_sec)}s 동안 규칙 라우터 ({str(e)[:80]})")
        except Exception as e:  # noqa: BLE001
            notes.append(f"LLM 라우터 실패({type(e).__name__}: {str(e)[:120]}) → 규칙 라우터")
        return self.rules.decide(question), notes

    # ------------------------------------------------------------------ 2. 근거
    def gather_wiki(self, d: RouteDecision) -> tuple[list[dict], dict]:
        wiki, pats = self.svc.wiki, self.svc.routes.patterns(d.sections)
        kws = [k for i in d.sections for k in (self.svc.routes.by_id.get(i).keywords if self.svc.routes.by_id.get(i) else [])]
        nodes = [n for c in d.companies for period in d.periods.get(c["corp_name"], [])
                 for n in wiki.find_sections(c["corp_name"], period, pats)]
        # 글자 예산을 페이지 수로 나눈다: 앞 회사의 긴 페이지가 예산을 다 써서 뒤 회사가 잘려 나가지 않게
        per_page = max(1500, min(self.page_chars, (self.ctx_chars - 200 * len(nodes)) // max(1, len(nodes))))
        items: list[dict] = []
        for n in nodes:
            text = select_relevant(wiki.page_text(n), d.search_query or "", kws, per_page)
            items.append({"id": f"wiki:{n['id'].split(':', 1)[1]}", "text": text,
                          "source": f"{n['corp']} {n['report_type']}({n['period']}) · {n['title']}",
                          "rcept_no": n.get("rcept_no"), "period": n.get("period"),
                          "section": n.get("title"), "kind": "wiki"})
        return items, {}

    def gather_search(self, d: RouteDecision, channels: list[str]) -> tuple[list[dict], dict]:
        from ..generation.answerer import source_label
        items, timings = [], {}
        for c in d.companies:
            for period in d.periods.get(c["corp_name"], []) or [None]:
                flt = {"corp_code": c["corp_code"], "is_latest": True}
                if period:
                    flt.update(period=period, year=period[:4])
                r = self.svc.retrieve(d.search_query or "", self.search_k, flt, channels)
                for k, v in r.timings.items():
                    timings[k] = round(timings.get(k, 0) + v, 3)
                tables = {(h.chunk.meta.get("parent_id"), h.chunk.meta.get("table_index"),
                           h.chunk.meta.get("row_start")) for h in r.hits if h.chunk.kind == "table"}
                for h in r.hits:
                    m = h.chunk.meta
                    if h.chunk.kind == "table_kv" and (m.get("parent_id"), m.get("table_index"),
                                                       m.get("row_start")) in tables:
                        continue
                    items.append({"id": h.chunk.id, "text": h.chunk.raw, "source": source_label(h.chunk),
                                  "rcept_no": m.get("rcept_no"), "period": m.get("period"),
                                  "section": m.get("section_path"), "kind": h.chunk.kind})
        return items, timings

    # ------------------------------------------------------------------ 3. 답변
    def answer_once(self, question: str, items: list[dict]) -> tuple[dict, dict, float]:
        data, usage, sec, _ = self._answer(question, items)
        return data, usage, sec

    def _answer(self, question: str, items: list[dict]):
        """answer_once + 대화 객체 (비교 서술을 다시 쓰게 할 때 같은 대화를 이어 쓴다). 요청마다 지역 변수로 둔다."""
        parts, used = [], 0
        for it in items:
            block = f"<근거 id={it['id']}>\n출처: {it['source']}\n{it['text']}\n</근거>"
            if used + len(block) > self.ctx_chars:
                break
            parts.append(block)
            used += len(block)
        today = (self.today or date.today()).isoformat()
        conv = self.answer_llm.start(SINGLE_SYSTEM, f"오늘 날짜: {today}\n질문: {question}\n\n[근거]\n" + "\n\n".join(parts))
        st = self.answer_llm.step(conv, None, ANSWER_SCHEMA, allow_tools=False)
        return parse_final_json(st.final_text), st.usage, st.seconds, conv

    # ------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        d, notes = self.route(question)
        t_route = time.perf_counter() - t0
        plan = {"mode": "auto", "route": d.route, "decision": d.to_dict(), "router_notes": notes}
        if d.route == "agent":
            return self._agent(question, plan, t_route, t0)

        t1 = time.perf_counter()
        if d.route == "wiki":
            items, st_t = self.gather_wiki(d)
        else:
            items, st_t = self.gather_search(d, ["dense", "bm25"] if d.route == "hybrid" else ["dense"])
        t_gather = time.perf_counter() - t1
        if not items:
            plan["router_notes"].append(f"{d.route} 경로에서 근거 0건")
            if self.escalate:
                return self._agent(question, plan, t_route, t0, reason="근거 없음")
            return Answer(question, "요청한 조건에 맞는 공시 근거를 찾지 못했습니다.", [], plan, "none",
                          {"route": round(t_route, 3), "gather": round(t_gather, 3)}, status="no_evidence")

        data, usage, t_ans, conv = self._answer(question, items)
        seen = {it["id"]: it for it in items}
        try:
            data, comp = check_and_fix(self.answer_llm, conv, data, seen,
                                       getattr(self.cfg, "aliases", {}), ANSWER_SCHEMA)
        except Exception as e:  # noqa: BLE001  비교 검증 실패가 답변을 막지 않게
            comp = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        if comp.get("retried"):
            t_ans += comp.get("seconds", 0.0)
            usage = {k: (usage.get(k) or 0) + (comp["usage"].get(k) or 0) for k in set(usage) | set(comp["usage"])
                     if k != "usd"} | {"usd": (None if usage.get("usd") is None or comp["usage"].get("usd") is None
                                               else round(usage["usd"] + comp["usage"]["usd"], 6))}
        sources, check = verify_evidence(data.get("evidence") or [], seen)
        if not sources and not data.get("needs_clarification") and self.escalate:
            return self._agent(question, plan, t_route, t0, reason="단일 답변이 근거를 인용하지 못함")
        status = "needs_clarification" if data.get("needs_clarification") else ("answered" if sources else "no_evidence")
        if comp:
            plan["comparison"] = {k: v for k, v in comp.items() if k != "usage"}
        plan.update(evidence_check=check, usage=usage, n_context=len(items), search_timings=st_t,
                    answer_model=f"{getattr(self.answer_llm, 'backend', '?')}:{getattr(self.answer_llm, 'model', '?')}")
        return Answer(question, render(data, check), sources, plan,
                      f"auto:{d.route}:{getattr(self.answer_llm, 'model', '?')}",
                      {"route": round(t_route, 3), "gather": round(t_gather, 3), "answer": round(t_ans, 3),
                       "total": round(time.perf_counter() - t0, 3)}, status=status)

    def _agent(self, question, plan, t_route, t0, reason: str | None = None) -> Answer:
        if reason:
            plan["escalated"] = reason
        a = self.svc.agent.run(question)
        a.plan = {**a.plan, **{k: v for k, v in plan.items() if k not in ("mode",)}, "mode": "agent",
                  "via": "auto"}
        a.timings = {**a.timings, "route": round(t_route, 3), "total": round(time.perf_counter() - t0, 3)}
        return a
