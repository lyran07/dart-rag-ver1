"""답변 생성: 검색 결과 → 컨텍스트 조립(예산) → LLM → Answer.

보완 사항
- sources 는 "실제로 프롬프트에 들어간" 청크만 (예산 때문에 잘린 청크는 제외) + DART 원문 링크.
- 답변의 출처 번호 [n] 가 실제 컨텍스트 번호인지 검사 (없는 번호는 missing_citations 로 보고).
- 검색 결과가 없으면 LLM 을 호출하지 않고 status=no_evidence.
- LLM 이 전부 실패하면 status=extractive_fallback (오류 문구를 답변 본문에 섞지 않음).
"""
from __future__ import annotations

import re
import time

from ..config import GenerationCfg
from ..domain import Answer, ScoredChunk
from ..llm import LLMRouter, NoLLMAvailable
from ..retrieval import RetrievalResult

SYSTEM = """당신은 한국 상장회사의 DART 공시(사업보고서·분기보고서) 전문 애널리스트입니다.
아래 [컨텍스트] 에 제공된 공시 발췌만을 근거로 답하세요.
규칙:
1. 수치는 컨텍스트에 있는 값·단위·기수(제N기)·기간을 그대로 인용하고, 반드시 출처 번호 [n] 를 붙입니다.
2. 컨텍스트에 없는 내용은 추측하지 말고 "제공된 공시 발췌에서 확인되지 않습니다" 라고 말합니다.
3. 여러 보고서(기간)가 섞여 있으면 어느 보고서 기준인지 명시합니다.
4. 사실(공시에 적힌 내용)과 해석(그로부터 추론한 내용)을 구분해 적습니다.
5. 표 데이터는 필요하면 간단한 표로 재정리합니다. 답변은 한국어로 간결하게."""

NO_EVIDENCE = "요청한 기업·기간 조건에 해당하는 공시 근거를 찾지 못했습니다. 조건을 넓히거나 질문을 바꿔 주세요."


def dart_url(rcept_no: str | None) -> str | None:
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}" if rcept_no else None


def source_label(c) -> str:
    m = c.meta
    return f"{m.get('corp_name')} {m.get('report_type')}({m.get('period')}) · {m.get('section_path')}"


def source_entry(i: int, h: ScoredChunk) -> dict:
    m = h.chunk.meta
    return {"n": i, "source": source_label(h.chunk), "kind": h.chunk.kind, "chunk_id": h.chunk.id,
            "rcept_no": m.get("rcept_no"), "period": m.get("period"), "section": m.get("section_path"),
            "url": dart_url(m.get("rcept_no")), "score": round(h.score, 4)}


class AnswerGenerator:
    def __init__(self, router: LLMRouter, cfg: GenerationCfg):
        self.router, self.cfg = router, cfg

    def build_context(self, hits: list[ScoredChunk], max_ctx: int) -> tuple[str, list[int]]:
        """컨텍스트 문자열과 '실제 포함된 hit 번호(1-base)' 를 함께 돌려준다."""
        per, parts, used, total = self.cfg.per_chunk_chars, [], [], 0
        for i, h in enumerate(hits, 1):
            p = f"[{i}] ({source_label(h.chunk)})\n{h.chunk.raw}"
            if len(p) > per:
                p = p[:per] + "\n...(중략)"
            if total + len(p) > max_ctx:
                break
            parts.append(p)
            used.append(i)
            total += len(p)
        return "\n\n".join(parts), used

    @staticmethod
    def cited(text: str) -> list[int]:
        return sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", text)})

    def answer(self, question: str, r: RetrievalResult) -> Answer:
        plan = {**r.plan.to_dict(), "used_filter": r.used_filter}
        t = time.perf_counter()
        if not r.hits:
            return Answer(question, NO_EVIDENCE, [], plan, "none",
                          {**r.timings, "generate": time.perf_counter() - t}, status="no_evidence")

        used_holder: dict[str, list[int]] = {}

        def build_user(budget: int) -> str:
            ctx, used = self.build_context(r.hits, budget)
            used_holder["used"] = used
            return f"[컨텍스트]\n{ctx}\n\n[질문]\n{question}"

        try:
            res = self.router.complete(SYSTEM, build_user)
            text, llm, status = res.text, f"{res.provider}:{res.model}", "answered"
            plan["llm_attempts"] = res.attempts
        except NoLLMAvailable as e:
            _, used = self.build_context(r.hits[:3], self.cfg.max_ctx_chars)
            used_holder["used"] = used
            text, llm, status = "답변 생성용 LLM 을 사용할 수 없어 관련 공시 발췌만 제공합니다.", "none", "extractive_fallback"
            plan["llm_error"] = str(e)[:300]

        used = used_holder.get("used", [])
        sources = [source_entry(i, r.hits[i - 1]) for i in used]
        if status == "answered":
            missing = [n for n in self.cited(text) if n not in used]
            if missing:
                plan["missing_citations"] = missing
        return Answer(question, text, sources, plan, llm, {**r.timings, "generate": time.perf_counter() - t},
                      status=status)
