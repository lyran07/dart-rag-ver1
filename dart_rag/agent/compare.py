"""비교·추이 계산 검증 (LLM 은 값을 원문에서 옮기고, 대소·순위·증감은 코드가 계산한다).

  답변 JSON 의 comparison = [{subject, item, period, value, unit}]  (여러 회사·기간 값을 비교하는 질문일 때)
  analyze()      같은 항목·단위끼리 묶어 순위(회사 비교) 또는 증감률·배수(기간 추이)를 계산
  find_issues()  답변 문장의 '가장 크다/작다', '증가/감소' 서술이 계산과 다른 곳을 찾는다
  check_and_fix() 다르면 계산 결과를 주고 LLM 에게 1회 다시 쓰게 한다. 그래도 다르면 경고를 단다.
"""
from __future__ import annotations

import re

from .llm import parse_final_json

BIG = r"(크|큰|큽|클|커|컸|많|높|앞)"
SMALL = r"(작|적|낮)"
MAX_CUE = re.compile(r"(가장|제일)\s*" + BIG + r"|최대|최고|1위")
MIN_CUE = re.compile(r"(가장|제일)\s*" + SMALL + r"|최소|최저")
PAIR_MAX = re.compile(r"더\s*" + BIG)
PAIR_MIN = re.compile(r"더\s*" + SMALL)
UP = re.compile(r"증가|늘었|늘어|늘며|늘고|상승|확대|커졌")
DOWN = re.compile(r"감소|줄었|줄어|줄며|줄고|하락|축소|작아졌")

FIX_PROMPT = """[계산 확인] 답변의 comparison 값으로 코드가 계산한 결과입니다.
{lines}

답변의 비교 서술이 계산 결과와 다릅니다: {issues}
계산 결과에 맞게 conclusion·key_points·interpretation 의 비교 서술을 고쳐, 같은 JSON 형식으로 최종 답변을 다시 출력하세요.
값과 근거(evidence)는 그대로 두고, 도구는 호출하지 마세요."""


def parse_value(s) -> float | None:
    """'5,000', '27.0%', '△716', '(4,751,654)', '-12.3' → 숫자. 숫자가 아니면 None."""
    t = str(s or "").strip()
    neg = t.startswith(("△", "-", "−", "(")) and not t.startswith("(+")
    m = re.search(r"\d[\d,]*(?:\.\d+)?", t)
    if not m:
        return None
    v = float(m.group(0).replace(",", ""))
    return -v if neg else v


def _norm_unit(u: str) -> str:
    return re.sub(r"\s+", "", str(u or ""))


def _fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 100 and float(v).is_integer() else f"{v:,.2f}".rstrip("0").rstrip(".")


def analyze(rows: list[dict], seen_texts: list[str] | None = None) -> dict:
    hay = "\n".join(seen_texts or [])
    hay_nc = hay.replace(",", "")
    groups: dict[tuple, list[dict]] = {}
    for r in rows or []:
        v = parse_value(r.get("value"))
        if v is None:
            continue
        raw = re.sub(r"[^\d.,]", "", str(r.get("value")))
        e = {"subject": str(r.get("subject", "")).strip(), "period": str(r.get("period", "")).strip(),
             "value_str": str(r.get("value")).strip(), "value": v,
             "in_source": (not seen_texts) or raw in hay or raw.replace(",", "") in hay_nc}
        groups.setdefault((str(r.get("item", "")).strip(), _norm_unit(r.get("unit"))), []).append(e)
    out = []
    for (item, unit), es in groups.items():
        if len(es) < 2:
            continue
        subjects = {e["subject"] for e in es}
        periods = {e["period"] for e in es}
        g = {"item": item, "unit": unit, "entries": es}
        if len(subjects) > 1 and len(periods) <= 1:
            ranked = sorted(es, key=lambda e: -e["value"])
            g.update(kind="compare", ranked=ranked, max=ranked[0]["subject"], min=ranked[-1]["subject"])
        elif len(subjects) == 1 and len(periods) > 1:
            seq = sorted(es, key=lambda e: e["period"])
            steps = []
            for a, b in zip(seq, seq[1:]):
                pct = (b["value"] / a["value"] - 1) * 100 if a["value"] else None
                steps.append({"from": a["period"], "to": b["period"], "pct": pct,
                              "ratio": (b["value"] / a["value"]) if a["value"] else None,
                              "direction": "증가" if b["value"] > a["value"] else ("감소" if b["value"] < a["value"] else "같음")})
            g.update(kind="trend", seq=seq, steps=steps)
        else:
            g.update(kind="mixed", ranked=sorted(es, key=lambda e: -e["value"]))
        out.append(g)
    return {"groups": out}


def render_lines(info: dict) -> list[str]:
    lines = []
    for g in info.get("groups", []):
        u = f" ({g['unit']})" if g["unit"] else ""
        mark = lambda e: "" if e["in_source"] else " (원문 미확인)"  # noqa: E731
        if g["kind"] == "compare":
            r = g["ranked"]
            s = " > ".join(f"{e['subject']} {e['value_str']}{mark(e)}" for e in r)
            if len(r) == 2 and r[1]["value"] > 0:
                s += f"  · {r[0]['subject']}가 약 {r[0]['value'] / r[1]['value']:.1f}배"
            lines.append(f"- {g['item']}{u}: {s}")
        elif g["kind"] == "trend":
            s = " → ".join(f"{e['period']} {e['value_str']}{mark(e)}" for e in g["seq"])
            ch = ", ".join(f"{st['to']} {st['pct']:+.1f}%" if st["pct"] is not None and abs(st["pct"]) < 1000
                           else f"{st['to']} {st['ratio']:.1f}배" if st["ratio"] else f"{st['to']} {st['direction']}"
                           for st in g["steps"])
            lines.append(f"- {g['item']}{u}: {s}  ({ch})")
        else:
            lines.append(f"- {g['item']}{u}: " + " > ".join(f"{e['subject']} {e['period']} {e['value_str']}"
                                                            for e in g["ranked"]))
    return lines


def _names(subject: str, aliases: dict[str, str]) -> list[str]:
    return [subject] + [a for a, t in (aliases or {}).items() if t == subject and a != subject]


def _claims(sent: str, cue: re.Pattern, subjects: list[str], aliases: dict[str, str]) -> list[str]:
    ms = []
    for subj in subjects:
        for n in _names(subj, aliases):
            for m in re.finditer(re.escape(n), sent):
                ms.append((m.start(), m.end(), subj))
    ms = [f for f in ms if not any(g[0] <= f[0] and f[1] <= g[1] and g != f for g in ms)]
    ms = sorted(f for f in ms if not re.match(r"\s*(의\s*\S+\s*)?보다", sent[f[1]:f[1] + 12]))
    out = []
    for c in cue.finditer(sent):
        after = [m for m in ms if 0 <= m[0] - c.end() <= 25 and re.search(r"(은|는)\s*", sent[c.end():m[0]])]
        before = [m for m in ms if m[1] <= c.start()]
        if after:
            out.append(after[0][2])
        elif before:
            out.append(before[-1][2])
    return out


def find_issues(info: dict, texts: list[str], aliases: dict[str, str] | None = None) -> list[str]:
    issues = []
    sents = [s for t in texts for s in re.split(r"(?<=[.!?])\s+|\n", str(t)) if s.strip()]
    comps = [g for g in info.get("groups", []) if g["kind"] == "compare"]
    for g in comps:
        subjects = [e["subject"] for e in g["entries"]]
        scope = [s for s in sents if len(comps) == 1 or g["item"] in s]
        pair = len(set(subjects)) == 2
        for key, cue, pcue in (("max", MAX_CUE, PAIR_MAX), ("min", MIN_CUE, PAIR_MIN)):
            claimed = [c for s in scope for c in _claims(s, cue, subjects, aliases)]
            if pair:
                claimed += [c for s in scope for c in _claims(s, pcue, subjects, aliases)]
            wrong = sorted({c for c in claimed if c != g[key]})
            if wrong:
                word = "큰" if key == "max" else "작은"
                issues.append(f"'{g['item']}' 값이 가장 {word} 곳은 {g[key]}인데 {', '.join(wrong)}(으)로 서술")
    trends = [g for g in info.get("groups", []) if g["kind"] == "trend" and len(g["steps"]) == 1]
    if len(trends) == 1 and not comps:
        d = trends[0]["steps"][0]["direction"]
        head = str(texts[0]) if texts else ""
        if d == "감소" and UP.search(head) and not DOWN.search(head):
            issues.append(f"'{trends[0]['item']}'은 감소했는데 결론에 증가로 서술")
        if d == "증가" and DOWN.search(head) and not UP.search(head):
            issues.append(f"'{trends[0]['item']}'은 증가했는데 결론에 감소로 서술")
    return issues


def _texts(data: dict) -> list[str]:
    return [str(data.get("conclusion", ""))] + [str(x) for x in data.get("key_points") or []] + \
           [str(x) for x in data.get("interpretation") or []]


def check_and_fix(llm, conv, data: dict, seen: dict[str, dict], aliases: dict[str, str] | None,
                  schema: dict) -> tuple[dict, dict]:
    """comparison 이 있으면 계산·대조하고, 서술이 다르면 1회 다시 쓰게 한다. (data, 정보) 를 돌려준다."""
    texts = [v.get("text", "") for v in seen.values()]
    info = analyze(data.get("comparison") or [], texts)
    if not info["groups"]:
        return data, {}
    meta = {"lines": render_lines(info), "issues": find_issues(info, _texts(data), aliases), "retried": False,
            "usage": {}, "seconds": 0.0}
    if meta["issues"] and conv is not None:
        llm.add_user(conv, FIX_PROMPT.format(lines="\n".join(meta["lines"]), issues="; ".join(meta["issues"])))
        st = llm.step(conv, None, schema, allow_tools=False)
        meta.update(retried=True, usage=st.usage or {}, seconds=st.seconds)
        try:
            fixed = parse_final_json(st.final_text)
            fixed.setdefault("comparison", data.get("comparison"))
            data = fixed
        except Exception:  # noqa: BLE001  다시 쓰기에 실패하면 원래 답 + 경고
            pass
        info = analyze(data.get("comparison") or [], texts)
        meta["lines"] = render_lines(info)
        meta["issues_after"] = find_issues(info, _texts(data), aliases)
    else:
        meta["issues_after"] = meta["issues"]
    data["_calc_lines"] = meta["lines"]
    data["_calc_issues"] = meta["issues_after"]
    return data, meta
