"""CLI.  python -m dart_rag <command> ...

수집·처리 (파이프라인)
  discover  --corp 삼성전자 SK하이닉스 | --corp-file kospi200.txt  --from 20230101 --to 20241231
  run       --stages fetch,parse,chunk,embed   [--until 06:30 | --max-minutes 600]   (GPU 서버)
  run       --stages index,wiki                                                    (서빙 머신)
  status    카탈로그 진행 상황 + 단계별 처리 속도 + 저장 용량
  sample    키 없이 합성 샘플로 전체 파이프라인 (discover+run)

질의
  ask "질문"   chat   search "질문"   health
  report SK하이닉스 2026.06 [--compare auto|2025.06] [--out report.md]   기업분석 보고서 (기본: 비교 없음)
  toc <rcept_no>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import Settings


def _settings(a) -> Settings:
    ov = {}
    if getattr(a, "data_root", None):
        ov["storage.root"] = a.data_root
    if getattr(a, "embed", None):
        ov["embedding.provider"] = a.embed
    if getattr(a, "index_backend", None):
        ov["index.backend"] = a.index_backend
    return Settings.load(getattr(a, "config", None), ov)


def _corps(a) -> list[str]:
    out = list(a.corp or [])
    if getattr(a, "corp_file", None):
        out += [l.strip() for l in open(a.corp_file, encoding="utf-8") if l.strip() and not l.startswith("#")]
    return out


def _source(kind: str, s: Settings):
    from .ingestion.sources import make_source
    if kind == "dart":
        return make_source("dart", corp_cache=s.storage.root_path / "raw" / "corpCode.xml")
    return make_source(kind)


# ---------------------------------------------------------------- pipeline commands
def cmd_discover(a):
    from .pipeline import Catalog
    s = _settings(a)
    src = _source(a.source, s)
    fs = src.discover(_corps(a), a.from_, a.to, skip_amend=a.skip_amend,
                      report_types=tuple(a.report_types) if a.report_types else None)
    n = Catalog(s.storage.root_path / "catalog.db").upsert_filings(fs, src.name)
    print(f"[discover] 카탈로그 등록 {n}건 → {s.storage.root_path / 'catalog.db'}")


def cmd_run(a):
    from .pipeline import PipelineRunner, StageContext, parse_deadline
    s = _settings(a)
    stages = [x.strip() for x in a.stages.split(",")] if a.stages else None
    src = _source(a.source, s) if (stages is None or "fetch" in stages) else None
    ctx = StageContext(s, source=src)
    fs = ctx.catalog.filings(_corps(a) or None, latest_only=a.latest_only, limit=a.limit)
    if getattr(a, "rcept", None):
        fs = [f for f in fs if f.rcept_no in set(a.rcept)]
    if getattr(a, "period", None):
        fs = [f for f in fs if f.period in set(a.period)]
    deadline = parse_deadline(a.until, a.max_minutes)
    print(f"[run] 공시 {len(fs)}건 · 단계 {stages or 'all'} · 저장소 {s.storage.root_path}"
          + (f" · 마감 {deadline:%m-%d %H:%M} (여유 {a.stop_margin}분)" if deadline else ""))
    rep = PipelineRunner(ctx, stages, deadline, a.stop_margin, force=a.force).run(fs)
    print("\n".join(rep.lines()) or "  (할 일 없음 — 모두 완료 상태)")
    return rep


def cmd_status(a):
    from .pipeline import ArtifactStore, Catalog
    s = _settings(a)
    cat = Catalog(s.storage.root_path / "catalog.db")
    print(f"저장소: {s.storage.root_path}")
    print("카탈로그:", cat.counts())
    print(f"\n{'stage':6s} {'version':38s} {'status':7s} {'공시':>5s} {'items':>9s} {'sec':>8s} {'items/s':>8s}")
    for r in cat.summary():
        print(f"{r['stage']:6s} {r['version'][:38]:38s} {r['status']:7s} {r['filings']:5d} {r['items']:9d} "
              f"{r['seconds']:8.1f} {str(r['items_per_sec'] or '-'):>8s}")
    for rc, st, err in cat.failures(5):
        print(f"  ✘ {rc} [{st}] {err}")
    u = ArtifactStore(s.storage.root_path).usage()
    print("\n용량:", ", ".join(f"{k} {v / 1024 ** 2:.1f}MB" for k, v in u.items()),
          f"| 합계 {sum(u.values()) / 1024 ** 2:.1f}MB")


def cmd_sample(a):
    a.source, a.corp, a.corp_file, a.from_, a.to, a.skip_amend = "sample", [], None, "", "", False
    cmd_discover(a)
    # 보고서(report) 단계는 LLM 키가 필요해서 샘플 실행에서는 뺀다
    a.stages = "fetch,parse,chunk,embed,index,wiki"
    a.latest_only, a.limit, a.until, a.max_minutes, a.stop_margin, a.force = False, None, None, None, 0, False
    a.rcept = a.period = None
    cmd_run(a)


# ---------------------------------------------------------------- query commands
def _service(a):
    from .service import RAGService
    return RAGService(_settings(a))


def _route_line(p: dict) -> str:
    d = p.get("decision") or {}
    if not d:
        return ""
    per = "; ".join(f"{k} {','.join(v)}" for k, v in (d.get("periods") or {}).items())
    s = (f"경로 {d.get('route')} (라우터 {d.get('router')}, {d.get('seconds', 0):.2f}s) · 기간 {per or '-'}"
         + (f" · 섹션 {','.join(d.get('sections') or [])}" if d.get("sections") else "")
         + (f" · 검색어 '{d.get('search_query')}'" if d.get("route") in ("hybrid", "vector") else ""))
    extra = [*(p.get("router_notes") or []), *(n for n in d.get("notes") or [] if not n.startswith("usage"))]
    if p.get("escalated"):
        extra.append(f"에이전트로 전환: {p['escalated']}")
    return s + ("\n  ※ " + " / ".join(extra) if extra else "")


def _print_agent(ans, trace: bool):
    p = ans.plan
    if p.get("mode") == "auto":
        head = f"모드 auto · 답변 {p.get('answer_model')}"
    else:
        head = (f"모드 agent{' (auto 경유)' if p.get('via') == 'auto' else ''} · {p.get('backend')}:{p.get('model')}"
                f" (reasoning {p.get('reasoning')}) · 라운드 {p.get('rounds')}")
    rl = _route_line(p)
    print(f"\n질문: {ans.question}\n{head} · 상태 {ans.status}" + (f"\n{rl}" if rl else "")
          + f"\n\n{ans.answer}\n\n근거:")
    for s in ans.sources:
        mark = "검증됨" if s.get("verified") else "미검증 " + "; ".join(s.get("problems") or [])
        print(f"  [{s['n']}] {s['source']} ({s.get('kind')}, {mark})\n      \"{(s.get('quote') or '')[:150]}\"")
    if trace and p.get("search_timings"):
        print("\n검색 세부 시간:", p["search_timings"])
    if trace and p.get("trace"):
        print("\n과정:")
        for t in p.get("trace", []):
            tools = " | ".join(f"{x['name']}({json.dumps(x['args'], ensure_ascii=False)}){' 오류' if x['error'] else ''}"
                               f" {x['sec']}s" + (f" {x['detail']}" if x.get("detail") else "")
                               for x in t.get("tools", []))
            print(f"  라운드 {t.get('round')}: LLM {t.get('llm_sec', '-')}s" + (f" → {tools}" if tools else "")
                  + (" → 최종 답변" if t.get("final") else "") + (f" ({t['stop']})" if t.get("stop") else ""))
    u = p.get("usage", {})
    usd = f" · ${u['usd']:.4f}" if u.get("usd") is not None else ""
    print(f"시간: {ans.timings} · 토큰 입력 {u.get('input_tokens', 0):,}(캐시 {u.get('cached_tokens', 0):,})"
          f" / 출력 {u.get('output_tokens', 0):,}{usd}")


def _print_answer(ans, trace: bool = False):
    if (ans.plan or {}).get("mode") in ("agent", "auto"):
        return _print_agent(ans, trace)
    print(f"\n질문: {ans.question}\n계획: {json.dumps(ans.plan.get('used_filter'), ensure_ascii=False)}"
          f" · 모드 {ans.plan.get('mode', 'rag')} · LLM {ans.llm} · 상태 {ans.status}\n\n{ans.answer}\n\n출처:")
    for s in ans.sources:
        print(f"  [{s['n']}] {s['source']} ({s['kind']}, {s['score']})")
    if ans.plan.get("agent_error"):
        print("에이전트 실패 → rag 로 대체:", ans.plan["agent_error"])
    if ans.plan.get("auto_error"):
        print("auto 실패 → rag 로 대체:", ans.plan["auto_error"])
    print("시간:", {k: round(v, 3) for k, v in ans.timings.items()})
    plan = ans.plan or {}
    if plan.get("llm_attempts"):          # 성공: 어떤 제공자를 거쳤는지
        print("LLM 시도:", " → ".join(
            f"{x['provider']}({'OK' if x.get('ok') else x.get('status') or x.get('error')}, {x.get('sec', '-')}s)"
            if x.get("ok") else f"{x['provider']}({x.get('status') or x.get('error')})"
            for x in plan["llm_attempts"]))
    if plan.get("llm_error"):             # 전부 실패: 제공자별 상태·오류 종류
        print("LLM 실패:", plan["llm_error"])


def cmd_route(a):
    """경로 판정만 확인 (답변 생성 없음). 규칙 라우터는 0원, LLM 라우터는 질문당 약 0.1원."""
    import time as _t
    svc = _service(a)
    qs = [a.question] if a.question else [
        ln.strip() for ln in Path(a.file).read_text("utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    names = ["rules"] + ([] if a.rules_only else (a.routers or [svc.agent_config.auto.get("router", "gemini")]))
    routers = {}
    for n in names:
        if n == "rules":
            routers[n] = svc.rule_router
            continue
        try:
            routers[n] = svc.make_llm_router(n)
        except Exception as e:  # noqa: BLE001
            print(f"[{n}] 사용 불가: {type(e).__name__}: {e}")
    rows = []
    for q in qs:
        print(f"\n질문: {q}")
        for n, r in routers.items():
            t = _t.perf_counter()
            try:
                d = r.decide(q)
                per = "; ".join(f"{k} {','.join(v)}" for k, v in d.periods.items())
                print(f"  [{n:6s}] {d.route:6s} {(_t.perf_counter() - t):5.2f}s · 회사 "
                      f"{','.join(c['corp_name'] for c in d.companies) or '-'} · 기간 {per or '-'}"
                      + (f" · 섹션 {','.join(d.sections)}" if d.sections else "")
                      + (f" · 검색어 '{d.search_query}'" if d.route in ("hybrid", "vector") else "")
                      + f"\n           이유: {d.reason}"
                      + "".join(f"\n           ※ {x}" for x in d.notes if not x.startswith("usage")))
                rows.append({"question": q, "router": n, "route": d.route, "sec": round(_t.perf_counter() - t, 3),
                             "companies": [c["corp_name"] for c in d.companies], "periods": d.periods,
                             "sections": d.sections})
            except Exception as e:  # noqa: BLE001
                print(f"  [{n:6s}] 실패 {type(e).__name__}: {str(e)[:150]}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", "utf-8")
        print(f"\n저장: {a.out}")


def cmd_ask(a):
    _print_answer(_service(a).ask(a.question, a.k, mode=a.mode), trace=a.trace)


def cmd_search(a):
    r = _service(a).retrieve(a.question, a.k)
    print(f"필터: {r.used_filter} · 확장: {r.plan.lexical_text} · 섹션힌트: {r.plan.section_hints}")
    for i, h in enumerate(r.hits, 1):
        m = h.chunk.meta
        print(f"[{i}] {h.score:.4f} {h.chunk.kind:8s} {m.get('period')} {m.get('section_path')}")
    print("시간:", {k: round(v, 3) for k, v in r.timings.items()})


def cmd_chat(a):
    svc = _service(a)
    print("질문을 입력하세요 (quit 종료)")
    while True:
        try:
            q = input("\n> ").strip()
        except EOFError:
            break
        if q in {"quit", "exit", ""}:
            break
        _print_answer(svc.ask(q))


def cmd_report(a):
    """보고서 1건 생성 (저장본이 있으면 그대로). 섹션별 진단을 출력하고 마크다운을 저장한다."""
    from .report import to_markdown
    svc = _service(a)
    rb = svc.report
    compare = None if a.compare == "none" else (rb.default_compare(a.corp, a.period) if a.compare == "auto"
                                                else a.compare)
    print(f"[report] {a.corp} {a.period}" + (f" vs {compare}" if compare else " (비교 없음)") + " 생성 중…")
    job = rb.run(a.corp, a.period, compare, a.template, use_cache=not a.no_cache)
    print(f"상태 {job.status} · {'저장본 사용' if job.from_cache else f'생성 {job.generated_sec}s'}"
          f" · 토큰 입력 {job.usage.get('input_tokens')} 출력 {job.usage.get('output_tokens')}"
          f" · 비용 ${job.usage.get('usd')}")
    for s in job.sections:
        d, c = s.diag, s.check
        print(f"  {s.title:10s} {s.status:11s} {d.get('total_sec', 0):5.1f}s 근거 {d.get('n_context', '-')}"
              f" ({d.get('context_chars', '-')}자, 잘린 페이지 {d.get('pages_trimmed', '-')})"
              f" 미검증 {c.get('n_unverified', 0)} 표숫자불일치 {len(c.get('table_unverified_numbers') or [])}"
              + (f" 오류 {s.error}" if s.error else ""))
    md = to_markdown(job)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md, "utf-8")
        print(f"저장: {a.out}")
    else:
        print("\n" + md)


def cmd_health(a):
    print(json.dumps(_service(a).health(), ensure_ascii=False, indent=1))


def cmd_toc(a):
    from .ingestion.parser import DartXmlParser
    from .pipeline import Catalog, ArtifactStore
    s = _settings(a)
    f = next((f for f in Catalog(s.storage.root_path / "catalog.db").filings() if f.rcept_no == a.rcept_no), None)
    if not f:
        sys.exit(f"카탈로그에 없음: {a.rcept_no}")
    print(DartXmlParser.toc(DartXmlParser().parse(ArtifactStore(s.storage.root_path).load_raw(f)), a.level))


# ---------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="settings.yaml 경로 (기본 config/settings.yaml)")
    common.add_argument("--data-root", help="영구 저장소 경로 (예: /mnt/storage/dart)")
    common.add_argument("--embed", help="bge-m3 | e5 | openai | hashing")
    common.add_argument("--index-backend", help="memory | qdrant")

    p = argparse.ArgumentParser(prog="dart_rag")
    sp = p.add_subparsers(dest="cmd", required=True)

    d = sp.add_parser("discover", parents=[common])
    d.add_argument("--corp", nargs="*")
    d.add_argument("--corp-file")
    d.add_argument("--from", dest="from_", default="20230101")
    d.add_argument("--to", default="20241231")
    d.add_argument("--skip-amend", action="store_true")
    d.add_argument("--report-types", nargs="*", choices=["A001", "A002", "A003"],
                   help="A001 사업 / A002 반기 / A003 분기 (기본: 전부)")
    d.add_argument("--source", default="dart", choices=["dart", "sample"])
    d.set_defaults(fn=cmd_discover)

    r = sp.add_parser("run", parents=[common])
    r.add_argument("--stages", help="쉼표 구분: fetch,parse,chunk,embed,index,wiki (기본 전부)")
    r.add_argument("--corp", nargs="*")
    r.add_argument("--corp-file")
    r.add_argument("--latest-only", action="store_true", help="기재정정 이전본 제외")
    r.add_argument("--limit", type=int)
    r.add_argument("--rcept", nargs="*", help="특정 접수번호만")
    r.add_argument("--period", nargs="*", help="특정 기간만 (예: 2023.12 2024.09)")
    r.add_argument("--until", help="마감 시각 'HH:MM' 또는 ISO (대여 종료 시각)")
    r.add_argument("--max-minutes", type=float)
    r.add_argument("--stop-margin", type=float, default=20, help="마감 N분 전부터 새 공시 시작 안 함")
    r.add_argument("--force", action="store_true", help="완료된 단계도 다시 실행")
    r.add_argument("--source", default="dart", choices=["dart", "sample"])
    r.set_defaults(fn=cmd_run)

    for name, fn in [("status", cmd_status), ("sample", cmd_sample), ("health", cmd_health), ("chat", cmd_chat)]:
        sp.add_parser(name, parents=[common]).set_defaults(fn=fn)
    for name, fn in [("ask", cmd_ask), ("search", cmd_search)]:
        x = sp.add_parser(name, parents=[common])
        x.add_argument("question")
        x.add_argument("-k", type=int)
        if name == "ask":
            x.add_argument("--mode", choices=["auto", "agent", "rag"], default=None,
                           help="auto: 라우터가 wiki·hybrid·vector·agent 선택 / agent: tool calling 에이전트"
                                " / rag: 검색 1회+LLM 1회 (기본: config/agent.yaml)")
            x.add_argument("--trace", action="store_true", help="에이전트의 도구 호출 과정 출력")
        x.set_defaults(fn=fn)
    r = sp.add_parser("route", parents=[common], help="경로 판정만 확인 (답변 생성 없음)")
    r.add_argument("question", nargs="?")
    r.add_argument("--file", default="eval/route_questions.txt", help="질문 목록 파일 (한 줄에 하나)")
    r.add_argument("--routers", nargs="*", choices=["gemini", "luna"], help="비교할 LLM 라우터 (기본: 설정값)")
    r.add_argument("--rules-only", action="store_true", help="규칙 라우터만 (0원)")
    r.add_argument("--out", help="결과 JSONL 저장 경로")
    r.set_defaults(fn=cmd_route)
    rp = sp.add_parser("report", parents=[common], help="기업분석 보고서 1건 생성 (공시 1건 기준)")
    rp.add_argument("corp", help="DART 등록명 (예: SK하이닉스)")
    rp.add_argument("period", help="기준 보고서 기간 (예: 2026.06)")
    rp.add_argument("--compare", default="none", help="비교 기간 (예: 2025.06) | auto: 전년 동기 | none: 비교 없음(기본)")
    rp.add_argument("--template", default="company_analysis")
    rp.add_argument("--out", help="마크다운 저장 경로")
    rp.add_argument("--no-cache", action="store_true", help="저장본이 있어도 새로 생성")
    rp.set_defaults(fn=cmd_report)
    t = sp.add_parser("toc", parents=[common])
    t.add_argument("rcept_no")
    t.add_argument("--level", type=int, default=2)
    t.set_defaults(fn=cmd_toc)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    main()
