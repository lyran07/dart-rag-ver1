"""응답 시간 측정 (서버와 같은 조건: 프로세스 1개, warm-up 후).

  python scripts\\bench_latency.py                       # eval/route_questions.txt, auto 모드
  python scripts\\bench_latency.py --modes auto rag      # 모드 비교
  python scripts\\bench_latency.py --router luna         # 라우터 바꿔서 (gemini | luna | rules)
  python scripts\\bench_latency.py --embed-only          # 질의 임베딩 API 시간만 (병렬 사전 준비 판단용, 거의 0원)

결과: data/bench/latency_<시각>.csv  + 경로별 p50/p95 요약 출력.
비용: auto 모드는 질문당 LLM 2회(라우터+답변), 경로가 agent 면 더 많다. 20문항 기준 수십 원 이하.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(ROOT / "eval" / "route_questions.txt"))
    ap.add_argument("--modes", nargs="*", default=["auto"], choices=["auto", "agent", "rag"])
    ap.add_argument("--router", choices=["gemini", "luna", "rules"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--embed-only", action="store_true")
    a = ap.parse_args()
    if a.router:
        os.environ["ROUTER"] = a.router

    from dart_rag.service import RAGService
    svc = RAGService()
    print("[warmup]", svc.warmup())

    if a.embed_only:
        ts = []
        for i in range(5):
            t = time.perf_counter()
            svc.embedder.embed_query(f"임베딩 지연 측정 질의 {i} {time.time()}")
            ts.append(time.perf_counter() - t)
        print(f"질의 임베딩 API: p50 {statistics.median(ts):.2f}s · 최대 {max(ts):.2f}s · {[round(x, 2) for x in ts]}")
        print("→ p50 이 0.5초를 넘으면 병렬 사전 준비를 넣을 가치가 있습니다.")
        return

    qs = [ln.strip() for ln in Path(a.file).read_text("utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
    if a.limit:
        qs = qs[: a.limit]
    out_dir = ROOT / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"latency_{time.strftime('%Y%m%d_%H%M')}.csv"
    fields = ["mode", "question", "route", "escalated", "status", "total_sec", "t_route", "t_gather", "t_answer",
              "t_agent_total", "t_llm", "t_tools", "input_tokens", "output_tokens", "usd", "error"]
    rows = []
    n_all = len(qs) * len(a.modes)
    print(f"\n{n_all}건 측정 시작 → {out} (한 건씩 바로 기록)", flush=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        i = 0
        for mode in a.modes:
            for q in qs:
                i += 1
                print(f"▶ [{i}/{n_all}] ({mode}) {q}", flush=True)
                t = time.perf_counter()
                try:
                    ans = svc.ask(q, mode=mode)
                    err = ""
                except Exception as e:  # noqa: BLE001
                    ans, err = None, f"{type(e).__name__}: {e}"
                total = time.perf_counter() - t
                p = (ans.plan if ans else {}) or {}
                u = p.get("usage") or {}
                row = {"mode": mode, "question": q, "route": p.get("route", p.get("mode", "")),
                       "escalated": p.get("escalated", ""), "status": ans.status if ans else "error",
                       "total_sec": round(total, 3), **{f"t_{k}": v for k, v in (ans.timings if ans else {}).items()},
                       "input_tokens": u.get("input_tokens", ""), "output_tokens": u.get("output_tokens", ""),
                       "usd": u.get("usd", ""), "error": err}
                rows.append(row)
                w.writerow(row)
                f.flush()
                notes = " / ".join(p.get("router_notes") or [])
                print(f"   {row['route']:7s} {total:5.2f}s {row['status']}" + (f"  ※ {notes}" if notes else "")
                      + (f"  오류: {err[:120]}" if err else ""), flush=True)
    print("\n경로별 응답 시간")
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["mode"], r["route"]), []).append(r["total_sec"])
    for (m, rt), xs in sorted(groups.items()):
        print(f"  {m:5s} {rt:7s} n={len(xs):2d}  p50 {pct(xs, 50):5.2f}s  p95 {pct(xs, 95):5.2f}s")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
