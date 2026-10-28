"""질의응답 평가 (eval/qa_eval.yaml). 계획서 '기술적 검증 계획' 항목에 맞춘 지표를 한 번에 측정한다.

  python scripts/eval_qa.py --check                         평가셋 정답이 원문(위키)에 실제로 있는지 검사 (0원)
  python scripts/eval_qa.py --route rules                   경로·회사·기간 인식 정확도 (규칙 라우터, 0원)
  python scripts/eval_qa.py --route gemini                  LLM 라우터 (Gemini, 몇 원)
  python scripts/eval_qa.py --retrieval                     검색 Recall@k · MRR, 채널 비교 (dense / dense+bm25)
  python scripts/eval_qa.py --answer auto                   답변 정확도 · 근거 · 형식 · 시간 · 비용
  python scripts/eval_qa.py --answer auto agent rag         모드 비교
  python scripts/eval_qa.py --rescore A.json B.json         저장된 답변을 지금 채점 규칙으로 다시 채점·합산 (0원)
  python scripts/eval_qa.py --answer auto --start 42        42번째 문항부터만 (새로 추가한 문항)
  옵션: --types fact compare (유형 일부만)  --limit 5  -k 5  --workers 3 (답변 평가 동시 실행)
        --rpm 12 (Gemini 라우터 호출을 분당 12회로 제한. 무료 등급 한도가 분당 15회라서)

결과: eval/results/<시각>_<측정>.md (요약표) + .json (문항별 상세)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

TYPES = ["fact", "synonym", "concept", "compare", "trend", "calc", "relative", "cause", "negative"]
CORP_NAMES = {"삼성전자": ["삼성전자", "삼성"], "SK하이닉스": ["SK하이닉스", "하이닉스"], "현대자동차": ["현대자동차", "현대차"]}
BIG = r"(크|큰|커|컸|많|높|앞)"                     # '크다'는 활용하면 글자가 바뀐다 (큰·커·컸)
MAX_CUE = r"(가장|제일)\s*" + BIG + r"|최대|최고|1위"
MIN_CUE = r"(가장|제일)\s*(작|적|낮)|최소|최저"
PAIR_MAX = r"더\s*" + BIG
PAIR_MIN = r"더\s*(작|적|낮)"
UP = r"증가|늘었|늘어|늘며|늘고|상승|확대|개선|커졌"
DOWN = r"감소|줄었|줄어|줄며|줄고|하락|축소|악화|작아졌"
NO_ANSWER_HINTS = ["보유", "없습니다", "없어", "없다", "찾지 못", "확인되지", "확인할 수 없", "제공되지", "존재하지 않",
                   "포함되어 있지 않", "포함돼 있지 않"]


class Pacer:
    """분당 호출 수 제한 (Gemini 무료 등급: 분당 15회). 여러 스레드에서 같이 써도 간격을 지킨다."""

    def __init__(self, rpm: float | None):
        import threading
        self.gap = 60.0 / rpm if rpm else 0.0
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self):
        if not self.gap:
            return
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.gap
        if at > now:
            time.sleep(at - now)


def retry_after(msg: str, default: float = 30.0) -> float:
    m = re.search(r"retry in ([\d.]+)s", msg)
    return float(m.group(1)) + 1 if m else default


# ------------------------------------------------------------------ 채점 도우미
def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def korean_amounts_mil(text: str) -> list[float]:
    """답변 속 '131조 8,950억원', '131.9조원', '6,567억원' 같은 표기를 백만원 단위 숫자로."""
    out = []
    jo_re = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*조(?:\s*(\d[\d,]*)\s*억)?")
    for m in jo_re.finditer(text):
        jo, eok = _num(m.group(1)), _num(m.group(2)) if m.group(2) else 0.0
        if jo is not None:
            out.append(jo * 1_000_000 + (eok or 0) * 100)
    rest = jo_re.sub(" ", text)          # '131조 8,950억' 의 억 부분이 따로 한 번 더 잡히지 않게
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*억", rest):
        v = _num(m.group(1))
        if v is not None:
            out.append(v * 100)
    return out


def value_in_answer(value: str, unit: str | None, text: str) -> bool:
    raw = value.strip()
    if raw in text or raw.replace(",", "") in text.replace(",", ""):
        return True
    if unit == "백만원":
        target = _num(raw)
        if target:
            return any(abs(a - target) / target <= 0.01 for a in korean_amounts_mil(text))
    return False


def _mentions(sent: str) -> list[tuple[int, int, str]]:
    """문장 속 회사 언급 (시작, 끝, 회사). 'X보다' 처럼 비교 기준으로 쓰인 언급은 뺀다."""
    found = []
    for corp, names in CORP_NAMES.items():
        for n in names:
            for m in re.finditer(re.escape(n), sent):
                found.append((m.start(), m.end(), corp))
    found = [f for f in found if not any(g[0] <= f[0] and f[1] <= g[1] and g != f for g in found)]  # 긴 이름 우선
    return sorted(f for f in found if not re.match(r"\s*(의\s*\S+\s*)?보다", sent[f[1]:f[1] + 12]))


def claimed(text: str, cue: str) -> list[str]:
    """'X가 가장 크다' / '가장 큰 곳은 X' 형태에서 X 를 뽑는다."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+|\n", text):
        ms = _mentions(sent)
        for c in re.finditer(cue, sent):
            after = [m for m in ms if 0 <= m[0] - c.end() <= 25 and re.search(r"(은|는)\s*", sent[c.end():m[0]])]
            before = [m for m in ms if m[1] <= c.start()]
            if after:
                out.append(after[0][2])
            elif before:
                out.append(before[-1][2])
    return out


def conclusion_checks(item: dict, text: str) -> dict:
    res = {}
    many = len(item.get("corps") or []) >= 3
    for key, cue, pair in (("expect_max", MAX_CUE, PAIR_MAX), ("expect_min", MIN_CUE, PAIR_MIN)):
        if item.get(key):
            claims = claimed(text, cue) or ([] if many else claimed(text, pair))
            res[key] = bool(claims) and all(c == item[key] for c in claims)
    if item.get("expect_direction"):
        pos, ok = 0, True
        for d in item["expect_direction"]:
            m = re.compile(UP if d == "증가" else DOWN).search(text, pos)
            if not m:
                ok = False
                break
            pos = m.end()
        res["expect_direction"] = ok
    if item.get("expect_calc"):
        c = item["expect_calc"]
        pat = r"(\d+(?:\.\d+)?)\s*%" if c["unit"] == "%" else r"(\d+(?:\.\d+)?)\s*배"
        res["expect_calc"] = any(abs(float(x) - c["value"]) <= c["tol"] for x in re.findall(pat, text))
    return res


def answer_score(item: dict, text: str) -> dict:
    groups = item.get("answer") or []
    kws = item.get("keywords") or []
    v_ok = [any(value_in_answer(v, item.get("unit"), text) for v in alts) for alts in groups]
    k_ok = [any(k.lower() in text.lower() for k in alts) for alts in kws]
    concl = conclusion_checks(item, text)
    checks = v_ok + k_ok + list(concl.values())
    return {"values": v_ok, "keywords": k_ok, "conclusion": concl,
            "conclusion_ok": all(concl.values()) if concl else None,
            "correct": bool(checks) and all(checks)}


def score(item: dict, text: str, status: str) -> dict:
    if item.get("no_answer"):
        return {"correct": status in ("no_evidence", "needs_clarification") or any(h in text for h in NO_ANSWER_HINTS),
                "conclusion_ok": None}
    return answer_score(item, text)


def section_hit(item: dict, sources: list[dict]) -> bool | None:
    exp = item.get("expect") or []
    if not exp:
        return None
    for g in exp:
        if not any(any(s in (src.get("section") or src.get("source") or "") for s in g["section"])
                   and (not g.get("corp") or g["corp"] in (src.get("source") or ""))
                   for src in sources):
            return False
    return True


def pct(n, d):
    return f"{100 * n / d:.0f}%" if d else "-"


def load(a) -> list[dict]:
    qs = yaml.safe_load((ROOT / a.file).read_text("utf-8"))
    if a.start:
        qs = qs[a.start - 1:]
    if a.types:
        qs = [q for q in qs if q["type"] in a.types]
    return qs[: a.limit] if a.limit else qs


# ------------------------------------------------------------------ 1. 평가셋 정답 검사
def run_check(svc, qs) -> tuple[list[str], dict]:
    wiki, routes = svc.wiki, svc.routes
    lines, bad = ["## 평가셋 정답 검사 (원문 존재 여부)", ""], 0
    for it in qs:
        if it.get("no_answer"):
            continue
        texts = []
        for g in it.get("expect") or []:
            periods = [g["period"]] if g.get("period") else [r["period"] for r in wiki.reports_of(g["corp"])]
            for p in periods:
                for n in wiki.find_sections(g["corp"], p, g["section"]):
                    texts.append(wiki.page_text(n))
        hay = "\n".join(texts)
        miss = [alts for alts in it.get("answer") or [] if not any(v in hay for v in alts)]
        miss += [alts for alts in it.get("keywords") or [] if not any(k.lower() in hay.lower() for k in alts)]
        status = "OK" if texts and not miss else ("섹션 없음" if not texts else f"원문에 없음 {miss}")
        bad += status != "OK"
        lines.append(f"- {'✔' if status == 'OK' else '✘'} [{it['type']}] {it['q']} — {status} (페이지 {len(texts)})")
    lines += ["", f"문제 {bad}건"]
    return lines, {"problems": bad}


# ------------------------------------------------------------------ 2. 경로·회사·기간 인식
def run_route(svc, qs, which: str, pacer: Pacer) -> tuple[list[str], dict]:
    from dart_rag.agent.llm import QuotaExhausted
    router = svc.rule_router if which == "rules" else svc.make_llm_router(which)
    rows, agg = [], defaultdict(lambda: defaultdict(int))
    for i, it in enumerate(qs, 1):
        for attempt in range(4):
            if which != "rules":
                pacer.wait()
            t0 = time.perf_counter()
            try:
                d = router.decide(it["q"])
                break
            except QuotaExhausted as e:          # 분당 한도 → 안내된 시간만큼 쉬고 다시 (시간에는 넣지 않음)
                wait = retry_after(str(e))
                print(f"  [{which}] 한도 초과 → {wait:.0f}초 대기 후 재시도", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"{which} 라우터 한도 초과가 계속됩니다")
        sec = time.perf_counter() - t0
        if which != "rules":
            print(f"  [{which} {i}/{len(qs)}] {d.route:6s} {it['q']}", flush=True)
        got_c = sorted(c["corp_name"] for c in d.companies)
        r_ok = d.route in it["route"]
        c_ok = got_c == sorted(it.get("corps") or [])
        p_ok = None
        if it.get("periods") is not None:
            p_ok = all(sorted(d.periods.get(c, [])) == sorted(ps) for c, ps in it["periods"].items())
        a = agg[it["type"]]
        a["n"] += 1
        a["route"] += r_ok
        a["corp"] += c_ok
        if p_ok is not None:
            a["p_n"] += 1
            a["period"] += p_ok
        a["sec"] += sec
        rows.append({"q": it["q"], "type": it["type"], "route": d.route, "route_ok": r_ok, "corps": got_c, "corp_ok": c_ok,
                     "periods": d.periods, "period_ok": p_ok, "sections": d.sections, "sec": round(sec, 3)})
    lines = [f"## 경로·회사·기간 인식 ({which} 라우터)", "",
             "| 유형 | 문항 | 경로 정확도 | 회사 인식 | 기간 인식 | 평균 시간 |", "|---|---:|---:|---:|---:|---:|"]
    tot = defaultdict(float)
    for t in TYPES:
        if t not in agg:
            continue
        a = agg[t]
        for k, v in a.items():
            tot[k] += v
        lines.append(f"| {t} | {a['n']} | {pct(a['route'], a['n'])} | {pct(a['corp'], a['n'])} | "
                     f"{pct(a['period'], a['p_n'])} | {a['sec'] / a['n']:.2f}s |")
    lines.append(f"| **전체** | {int(tot['n'])} | **{pct(tot['route'], tot['n'])}** | **{pct(tot['corp'], tot['n'])}** | "
                 f"**{pct(tot['period'], tot['p_n'])}** | {tot['sec'] / max(tot['n'], 1):.2f}s |")
    wrong = [r for r in rows if not (r["route_ok"] and r["corp_ok"] and r["period_ok"] is not False)]
    if wrong:
        lines += ["", "틀린 문항"] + [f"- [{r['type']}] {r['q']} → 경로 {r['route']}, 회사 {r['corps']}, 기간 {r['periods']}"
                                       for r in wrong]
    return lines, {"rows": rows, "total": dict(tot)}


# ------------------------------------------------------------------ 3. 검색 품질
def run_retrieval(svc, qs, k: int, channel_sets: list[list[str]]) -> tuple[list[str], dict]:
    from scripts.eval_retrieval import match  # 같은 채점 규칙 (섹션·회사·기간)
    lines = [f"## 검색 품질 (Recall@{k}, MRR)", "",
             "| 채널 | 유형 | 문항 | 성공(Hit@k, 모든 그룹) | 그룹 Recall | MRR |", "|---|---|---:|---:|---:|---:|"]
    detail = {}
    for chs in channel_sets:
        agg = defaultdict(lambda: {"n": 0, "ok": 0, "g": 0, "g_ok": 0, "mrr": 0.0})
        rows = []
        for it in qs:
            if not it.get("expect"):
                continue
            r = svc.retrieve(it["q"], k, None, chs)
            hits = [h for h in r.hits if "expanded" not in h.why][:k]
            ranks = [next((i for i, h in enumerate(hits, 1) if match(h.chunk, g)), None) for g in it["expect"]]
            a = agg[it["type"]]
            a["n"] += 1
            a["ok"] += all(ranks)
            a["g"] += len(ranks)
            a["g_ok"] += sum(1 for x in ranks if x)
            a["mrr"] += 1 / ranks[0] if ranks[0] else 0
            rows.append({"q": it["q"], "type": it["type"], "ranks": ranks})
        tot = {"n": 0, "ok": 0, "g": 0, "g_ok": 0, "mrr": 0.0}
        name = "+".join(chs)
        for t in TYPES:
            if t not in agg:
                continue
            a = agg[t]
            for key in tot:
                tot[key] += a[key]
            lines.append(f"| {name} | {t} | {a['n']} | {pct(a['ok'], a['n'])} | {pct(a['g_ok'], a['g'])} | "
                         f"{a['mrr'] / a['n']:.2f} |")
        lines.append(f"| **{name}** | **전체** | {tot['n']} | **{pct(tot['ok'], tot['n'])}** | "
                     f"**{pct(tot['g_ok'], tot['g'])}** | **{tot['mrr'] / max(tot['n'], 1):.2f}** |")
        detail[name] = {"rows": rows, "total": tot}
    return lines, detail


# ------------------------------------------------------------------ 4. 답변
def _ask(svc, it, mode, pacer: Pacer | None = None):
    if pacer is not None and mode == "auto":
        pacer.wait()
    t0 = time.perf_counter()
    try:
        a = svc.ask(it["q"], mode=mode)
        err = None
    except Exception as e:  # noqa: BLE001
        a, err = None, f"{type(e).__name__}: {str(e)[:200]}"
    sec = time.perf_counter() - t0
    if a is None:
        return {"q": it["q"], "type": it["type"], "mode": mode, "error": err, "sec": round(sec, 2), "correct": False}
    text = a.answer
    sc = score(it, text, a.status)
    chk = a.plan.get("evidence_check") or {}
    usage = a.plan.get("usage") or {}
    return {"q": it["q"], "type": it["type"], "mode": mode, "status": a.status, "correct": sc["correct"], "score": sc,
            "conclusion_ok": sc.get("conclusion_ok"),
            "section_hit": None if it.get("no_answer") else section_hit(it, a.sources),
            "n_evidence": chk.get("n_evidence", len(a.sources)), "n_unverified": chk.get("n_unverified", 0),
            "format_ok": bool(text.strip()) and (a.status != "answered" or bool(a.sources)),
            "route": a.plan.get("route") or a.plan.get("mode"), "escalated": a.plan.get("escalated"),
            "rounds": a.plan.get("rounds"), "usd": usage.get("usd"), "sec": round(sec, 2), "llm": a.llm,
            "router": (a.plan.get("decision") or {}).get("router"),
            "router_fallback": any("규칙 라우터" in n for n in a.plan.get("router_notes") or []),
            "answer": text}


def run_answer(svc, qs, modes: list[str], workers: int, pacer: Pacer) -> tuple[list[str], dict]:
    lines, detail = ["## 답변 평가", ""], {}
    for mode in modes:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                rows = list(ex.map(lambda it: _ask(svc, it, mode, pacer), qs))
        else:
            rows = []
            for i, it in enumerate(qs, 1):
                rows.append(_ask(svc, it, mode, pacer))
                r = rows[-1]
                print(f"  [{mode} {i}/{len(qs)}] {'✔' if r['correct'] else '✘'} {r['sec']:5.1f}s "
                      f"{r.get('route') or ''}{' →agent' if r.get('escalated') else ''} {it['q']}", flush=True)
        detail[mode] = rows
        lines += summarize(mode, rows)
    return lines, detail


def summarize(mode: str, rows: list[dict]) -> list[str]:
    lines = [f"### 모드: {mode}", "",
             "| 유형 | 문항 | 답변 정확도 | 결론 정확도 | 근거 섹션 적중 | 미검증 근거 비율 | 에이전트 전환 | 중앙 시간 | 90% 시간 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    by = defaultdict(list)
    for r in rows:
        by[r["type"]].append(r)
    for t in TYPES + ["전체"]:
        rs = rows if t == "전체" else by.get(t)
        if not rs:
            continue
        secs = sorted(r.get("sec") or 0 for r in rs)
        cc = [r["conclusion_ok"] for r in rs if r.get("conclusion_ok") is not None]
        sh = [r["section_hit"] for r in rs if r.get("section_hit") is not None]
        ev = sum(r.get("n_evidence") or 0 for r in rs)
        uv = sum(r.get("n_unverified") or 0 for r in rs)
        esc = sum(1 for r in rs if r.get("escalated") or r.get("route") == "agent")
        p90 = secs[min(len(secs) - 1, int(round(0.9 * (len(secs) - 1))))]
        b = "**" if t == "전체" else ""
        lines.append(f"| {b}{t}{b} | {len(rs)} | {b}{pct(sum(r['correct'] for r in rs), len(rs))}{b} | "
                     f"{pct(sum(cc), len(cc))} | {pct(sum(sh), len(sh))} | {pct(uv, ev)} | {pct(esc, len(rs))} | "
                     f"{statistics.median(secs):.1f}s | {p90:.1f}s |")
    usd = [r["usd"] for r in rows if r.get("usd") is not None]
    fmt = sum(1 for r in rows if r.get("format_ok"))
    errs = [r for r in rows if r.get("error")]
    fb = sum(1 for r in rows if r.get("router_fallback"))
    lines += ["", f"- 형식 준수율 {pct(fmt, len(rows))} · 오류 {len(errs)}건"
              + (f" · LLM 라우터 대신 규칙 라우터로 처리 {fb}건 (한도·시간 초과)" if mode == "auto" else "")
              + (f" · LLM 비용 합계 ${sum(usd):.4f} (질문당 ${sum(usd) / len(usd):.5f})" if usd else ""), ""]
    wrong = [r for r in rows if not r["correct"]]
    if wrong:
        lines += ["틀린 문항"] + [f"- [{r['type']}] {r['q']} — {r.get('error') or r.get('status')}"
                                   f"{' · 채점 ' + json.dumps({k: v for k, v in (r.get('score') or {}).items() if k != 'correct'}, ensure_ascii=False) if r.get('score') else ''}"
                                   for r in wrong]
        lines.append("")
    return lines


def run_rescore(qs, paths: list[str]) -> tuple[list[str], dict]:
    """저장된 답변(.json, 여러 개면 합침)을 현재 평가셋·채점 규칙으로 다시 채점 (LLM 호출 없음).
    같은 질문이 여러 파일에 있으면 뒤 파일 것을 쓴다."""
    by_q = {it["q"]: it for it in qs}
    merged: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in paths:
        f = Path(path) if Path(path).is_absolute() else ROOT / path
        for mode, rows in (json.loads(f.read_text("utf-8")).get("answer") or {}).items():
            for r in rows:
                merged[mode][r["q"]] = r
    lines, detail = [f"## 다시 채점: {', '.join(paths)}", ""], {}
    for mode, by in merged.items():
        new = []
        for r in by.values():
            it = by_q.get(r["q"])
            if it is None or r.get("error"):
                continue
            sc = score(it, r.get("answer") or "", r.get("status"))
            new.append({**r, "type": it["type"], "correct": sc["correct"], "score": sc,
                        "conclusion_ok": sc.get("conclusion_ok")})
        detail[mode] = new
        lines += summarize(mode, new)
    return lines, detail


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="eval/qa_eval.yaml")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--route", nargs="*", choices=["rules", "gemini", "luna"])
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--channels", nargs="*", default=["dense", "dense,bm25"],
                    help="검색 채널 조합 (쉼표 구분). 기본: dense 와 dense,bm25 비교")
    ap.add_argument("--answer", nargs="*", choices=["auto", "agent", "rag"])
    ap.add_argument("--rescore", nargs="+", help="저장된 답변 결과(.json, 여러 개 가능)를 지금 채점 규칙으로 다시 채점·합산")
    ap.add_argument("--start", type=int, help="평가셋의 N번째 문항부터 (예: --start 42 → 새로 추가한 문항만)")
    ap.add_argument("--types", nargs="*", choices=TYPES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rpm", type=float, default=12, help="LLM 라우터 분당 호출 상한 (0 이면 제한 없음)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    if not (a.check or a.route is not None or a.retrieval or a.answer or a.rescore):
        a.check = True

    from dart_rag.service import RAGService
    svc = RAGService()
    pacer = Pacer(a.rpm)
    qs = load(a)
    out, detail, parts = [f"# 질의응답 평가 {datetime.now():%Y-%m-%d %H:%M} {a.label}".strip(), "",
                          f"평가셋 {a.file} · {len(qs)}문항", ""], {}, []
    if a.check:
        ls, d = run_check(svc, qs)
        out += ls + [""]
        detail["check"] = d
        parts.append("check")
    for which in a.route or []:
        ls, d = run_route(svc, qs, which, pacer)
        out += ls + [""]
        detail[f"route_{which}"] = d
        parts.append(f"route-{which}")
    if a.retrieval:
        ls, d = run_retrieval(svc, qs, a.k, [c.split(",") for c in a.channels])
        out += ls + [""]
        detail["retrieval"] = d
        parts.append("retrieval")
    if a.rescore:
        ls, d = run_rescore(qs, a.rescore)
        out += ls
        detail["answer"] = d
        parts.append("rescore")
    if a.answer:
        ls, d = run_answer(svc, qs, a.answer, a.workers, pacer)
        out += ls
        detail["answer"] = d
        parts.append("answer-" + "-".join(a.answer))
    text = "\n".join(out).rstrip() + "\n"
    print("\n" + text)
    res = ROOT / "eval" / "results"
    res.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d_%H%M}_{'_'.join(parts)}{('_' + a.label) if a.label else ''}"
    (res / f"{stem}.md").write_text(text, "utf-8")
    (res / f"{stem}.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(f"저장: eval/results/{stem}.md / .json")


if __name__ == "__main__":
    main()
