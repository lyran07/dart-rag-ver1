"""에이전트 루프.

  질문 → [LLM: 도구 호출 결정] → 도구 병렬 실행 → 결과 전달 → … (최대 max_rounds)
       → 최종 JSON 답변 → 인용 검증 → Answer

제한: 라운드 상한(max_rounds), 질문당 시간 상한(deadline_sec). 어느 쪽이든 닿으면
도구 없이 지금까지의 근거로 최종 답변을 한 번 강제한다.
인용 검증: evidence 의 chunk_id 가 이번 질문에서 실제로 조회한 근거인지, quote 안의 숫자가
그 근거 원문에 글자 그대로 있는지 확인한다. 실패한 근거는 verified=False 로 표시한다.
"""
from __future__ import annotations

import re
import time
from datetime import date

from ..domain import Answer
from .compare import check_and_fix
from .llm import AgentLLMError, parse_final_json
from .prompts import ANSWER_SCHEMA, FORCE_FINAL, SYSTEM, user_message
from .tools import TOOL_SCHEMAS

NUM_RE = re.compile(r"\d[\d,.]*\d|\d")


class AgentError(RuntimeError):
    pass


def _dart_url(rcept_no):
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}" if rcept_no else None


def verify_evidence(evidence: list[dict], seen: dict[str, dict]) -> tuple[list[dict], dict]:
    sources, n_bad = [], 0
    for i, ev in enumerate(evidence or [], 1):
        cid, quote = str(ev.get("chunk_id", "")), str(ev.get("quote", ""))
        src = seen.get(cid)
        problems = []
        if src is None:
            problems.append("조회하지 않은 chunk_id")
        else:
            body = src["text"]
            missing = [x for x in NUM_RE.findall(quote) if x not in body]
            if missing:
                problems.append(f"원문에 없는 숫자: {missing[:5]}")
        n_bad += bool(problems)
        sources.append({"n": i, "source": (src or {}).get("source", cid), "kind": (src or {}).get("kind"),
                        "chunk_id": cid, "rcept_no": (src or {}).get("rcept_no"),
                        "period": (src or {}).get("period"), "section": (src or {}).get("section"),
                        "url": _dart_url((src or {}).get("rcept_no")), "score": None,
                        "quote": quote, "verified": not problems, "problems": problems})
    return sources, {"n_evidence": len(sources), "n_unverified": n_bad}


def render(data: dict, check: dict) -> str:
    out = [str(data.get("conclusion", "")).strip()]
    if data.get("key_points"):
        out += ["", "**주요 내용 (공시 확인 사항)**"] + [f"- {x}" for x in data["key_points"]]
    if data.get("interpretation"):
        out += ["", "**해석**"] + [f"- {x}" for x in data["interpretation"]]
    if data.get("_calc_lines"):
        out += ["", "**계산 확인 (코드)**"] + list(data["_calc_lines"])
    if data.get("_calc_issues"):
        out += ["", "⚠ 비교 서술이 계산 결과와 다릅니다: " + "; ".join(data["_calc_issues"])]
    if check.get("n_unverified"):
        out += ["", f"⚠ 근거 {check['n_unverified']}건이 원문과 대조 검증되지 않았습니다. 출처를 직접 확인하세요."]
    return "\n".join(out).strip()


class AgentLoop:
    def __init__(self, llm, session_factory, cfg, today: str | None = None):
        self.llm, self.session_factory, self.cfg = llm, session_factory, cfg
        self.today = today

    def run(self, question: str) -> Answer:
        cfg, t0 = self.cfg, time.perf_counter()
        session = self.session_factory()
        conv = self.llm.start(SYSTEM, user_message(question, self.today or date.today().isoformat()))
        trace: list[dict] = []
        usage = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "usd": 0.0}
        t_llm = t_tools = 0.0
        final: str | None = None

        def add_usage(u: dict) -> None:
            for k in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens"):
                usage[k] += u.get(k, 0) or 0
            usage["usd"] = None if (u.get("usd") is None or usage["usd"] is None) else round(usage["usd"] + u["usd"], 6)

        try:
            for rnd in range(1, cfg.max_rounds + 1):
                if time.perf_counter() - t0 > cfg.deadline_sec:
                    trace.append({"round": rnd, "stop": "deadline"})
                    break
                st = self.llm.step(conv, TOOL_SCHEMAS, ANSWER_SCHEMA)
                t_llm += st.seconds
                add_usage(st.usage)
                rec: dict = {"round": rnd, "llm_sec": round(st.seconds, 2)}
                trace.append(rec)
                if not st.tool_calls:
                    final = st.final_text
                    rec["final"] = True
                    break
                results = session.run_parallel(st.tool_calls, cfg.tool_timeout_sec)
                t_tools += max((r[2] for r in results), default=0.0)
                self.llm.add_tool_results(conv, [(c, out) for c, out, _ in results])
                detail = getattr(session, "tool_timings", {})
                rec["tools"] = [{"name": c.name, "args": c.arguments, "sec": round(sec, 2),
                                 "error": out.startswith('{"error"'), "out_chars": len(out),
                                 **({"detail": detail[c.call_id]} if c.call_id in detail else {})}
                                for c, out, sec in results]
            if final is None:                              # 라운드·시간 상한 → 도구 없이 마무리
                self.llm.add_user(conv, FORCE_FINAL)
                st = self.llm.step(conv, None, ANSWER_SCHEMA, allow_tools=False)
                t_llm += st.seconds
                add_usage(st.usage)
                trace.append({"round": "final(forced)", "llm_sec": round(st.seconds, 2)})
                final = st.final_text
            data = parse_final_json(final)
            try:
                data, comp = check_and_fix(self.llm, conv, data, session.seen, getattr(cfg, "aliases", {}),
                                           ANSWER_SCHEMA)
            except AgentLLMError:
                raise
            except Exception as e:  # noqa: BLE001  비교 검증 실패가 답변을 막지 않게
                comp = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if comp.get("retried"):
                add_usage(comp["usage"])
                t_llm += comp.get("seconds", 0.0)
                trace.append({"round": "comparison-fix", "llm_sec": round(comp.get("seconds", 0.0), 2)})
        except AgentLLMError as e:
            raise AgentError(str(e)) from e

        sources, check = verify_evidence(data.get("evidence") or [], session.seen)
        status = "needs_clarification" if data.get("needs_clarification") else \
            ("answered" if sources else "no_evidence")
        total = time.perf_counter() - t0
        plan = {"mode": "agent", "backend": getattr(self.llm, "backend", "?"), "model": getattr(self.llm, "model", "?"),
                "reasoning": getattr(self.llm, "reasoning", None), "trace": trace, "usage": usage,
                "evidence_check": check, "rounds": sum(1 for t in trace if isinstance(t.get("round"), int)),
                "n_seen": len(session.seen),
                **({"comparison": {k: v for k, v in comp.items() if k != "usage"}} if comp else {})}
        return Answer(question, render(data, check), sources, plan,
                      f"agent:{getattr(self.llm, 'model', '?')}",
                      {"agent_total": round(total, 3), "llm": round(t_llm, 3), "tools": round(t_tools, 3)},
                      status=status)
