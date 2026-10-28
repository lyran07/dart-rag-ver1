"""검색 평가 (LLM 호출 없음).

질문마다 expect = 조건 그룹 목록. 그룹 = {section: [..하나라도 포함], corp: 회사, period: 기간} (corp/period 생략 가능)
  - 그룹 충족 = 상위 k 개 (small-to-big 확장분 제외) 중 조건을 모두 만족하는 청크가 있음
  - 질문 성공 = 모든 그룹 충족  (비교·추이 질문은 두 회사/두 기간이 모두 나와야 성공)
  - 그룹 recall = 충족 그룹 / 전체 그룹,  MRR = 첫 그룹의 첫 정답 순위 기준

python scripts/eval_retrieval.py --file eval/questions_real.yaml
python scripts/eval_retrieval.py --embed hashing --basic            # 비교 기준선
python scripts/eval_retrieval.py --channels dense                   # 채널별 기여
python scripts/eval_retrieval.py --json out.json                    # 결과 저장 (비교표용)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from dart_rag.config import Settings  # noqa: E402
from dart_rag.service import RAGService  # noqa: E402


def groups_of(item: dict) -> list[dict]:
    if "expect" in item:
        return item["expect"]
    return [{"section": item["expect_section"]}]


def match(chunk, g: dict) -> bool:
    m = chunk.meta
    return (any(s in m.get("section_path", "") for s in g["section"])
            and (not g.get("corp") or m.get("corp_name") == g["corp"])
            and (not g.get("period") or m.get("period") == g["period"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(Path(__file__).resolve().parents[1] / "eval" / "questions.yaml"))
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--embed")
    ap.add_argument("--channels")
    ap.add_argument("--rerank")
    ap.add_argument("--basic", action="store_true", help="질의 분석 기본 규칙 (첫 연도·첫 회사만, 의도 힌트 없음)")
    ap.add_argument("--json")
    ap.add_argument("--label", default="")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    ov = {}
    if a.rerank:
        ov["rerank.mode"] = a.rerank
    if a.embed:
        ov["embedding.provider"] = a.embed
    s = Settings.load(overrides=ov)
    if a.channels:
        s.retrieval.channels = a.channels.split(",")
    s.retrieval.multi_facet = not a.basic
    svc = RAGService(s)
    qs = yaml.safe_load(Path(a.file).read_text("utf-8"))
    by_type = defaultdict(lambda: {"n": 0, "ok": 0, "g": 0, "g_ok": 0, "mrr": 0.0})
    rows, t0 = [], time.perf_counter()
    for item in qs:
        r = svc.retrieve(item["q"], k=a.k)
        hits = [h for h in r.hits if "expanded" not in h.why][:a.k]
        gs = groups_of(item)
        g_ranks = [next((i for i, h in enumerate(hits, 1) if match(h.chunk, g)), None) for g in gs]
        ok = all(g_ranks)
        t = by_type[item.get("type", "all")]
        t["n"] += 1
        t["ok"] += ok
        t["g"] += len(gs)
        t["g_ok"] += sum(1 for x in g_ranks if x)
        t["mrr"] += 1 / g_ranks[0] if g_ranks[0] else 0
        top = [f"{h.chunk.meta.get('corp_name')}·{h.chunk.meta.get('period')}·"
               f"{h.chunk.meta.get('section_path', '').split(' > ')[-1][:18]}" for h in hits]
        rows.append({"q": item["q"], "type": item.get("type"), "ok": ok, "group_ranks": g_ranks, "top": top})
        if not a.quiet:
            print(f"{'✔' if ok else '✘'} {str(g_ranks):14s} [{item.get('type')}] {item['q']}")
            if not ok:
                print(f"      top3: {' | '.join(top[:3])}")
    n = sum(v["n"] for v in by_type.values())
    head = (f"[{a.label}] embed={svc.embedder.key} channels={','.join(s.retrieval.channels)} "
            f"query={'basic' if a.basic else 'multi-facet'} k={a.k} ({time.perf_counter() - t0:.1f}s)")
    print("\n" + head)
    print(f"  {'type':10s} {'성공':>6s} {'그룹recall':>10s} {'MRR':>6s}")
    summary = {}
    for t, v in sorted(by_type.items()):
        summary[t] = {"success": v["ok"] / v["n"], "group_recall": v["g_ok"] / v["g"], "mrr": v["mrr"] / v["n"], "n": v["n"]}
        print(f"  {t:10s} {v['ok']:>2d}/{v['n']:<3d} {v['g_ok'] / v['g']:>10.2f} {v['mrr'] / v['n']:>6.2f}")
    tot = {k: sum(v[k] for v in by_type.values()) for k in ("ok", "g", "g_ok", "mrr")}
    summary["ALL"] = {"success": tot["ok"] / n, "group_recall": tot["g_ok"] / tot["g"], "mrr": tot["mrr"] / n, "n": n}
    print(f"  {'ALL':10s} {tot['ok']:>2d}/{n:<3d} {tot['g_ok'] / tot['g']:>10.2f} {tot['mrr'] / n:>6.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps({"label": a.label, "header": head, "summary": summary, "rows": rows},
                                           ensure_ascii=False, indent=1), "utf-8")


if __name__ == "__main__":
    main()
