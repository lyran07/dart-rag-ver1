"""질문 문자열에서 회사·기간 찾기 (규칙 · LLM 없음). 규칙 라우터와 LLM 라우터 결과 검증에 함께 쓴다."""
from __future__ import annotations

import re
from datetime import date

from ..agent.tools import _norm, match_company

TYPE_MONTH = {"사업보고서": "12", "반기보고서": "06"}


def find_companies(question: str, corps: dict[str, str], aliases: dict[str, str]) -> list[str]:
    """질문에 등장하는 회사 (등록명·별칭, 부분 일치). 등장 순서대로."""
    q = _norm(question)
    names: list[tuple[int, str]] = []
    cands = {**{c: c for c in corps}, **{a: t for a, t in (aliases or {}).items() if t in corps}}
    for key in sorted(cands, key=len, reverse=True):
        i = q.find(_norm(key))
        if i >= 0 and cands[key] not in [n for _, n in names]:
            names.append((i, cands[key]))
    return [n for _, n in sorted(names)]


def resolve_company_names(names: list[str], corps: dict[str, str], aliases: dict[str, str]) -> list[dict]:
    out: list[dict] = []
    for n in names:
        m = match_company(n, corps, aliases)
        if len(m) == 1 and m[0] not in out:
            out.append(m[0])
    return out


def _years(question: str, today: date) -> list[int]:
    ys = [int(y) for y in re.findall(r"(20\d{2})\s*년?", question)]
    rel = [("재작년", -2), ("작년", -1), ("지난해", -1), ("전년", -1), ("올해", 0), ("금년", 0), ("이번\s*해", 0)]
    for pat, d in rel:
        if re.search(pat, question):
            ys.append(today.year + d)
    return list(dict.fromkeys(ys))


def _report_type(question: str) -> tuple[str | None, str | None]:
    """(보고서 종류, 기준월) — 질문에서 읽히는 만큼."""
    qm = re.search(r"([1-4])\s*분기", question)
    if "사업보고서" in question or re.search(r"연간|연말|년\s*말|기말|1년\s*치", question):
        return "사업보고서", "12"
    if re.search(r"반기|상반기", question) or (qm and qm.group(1) == "2"):
        return "반기보고서", "06"
    if qm:
        return "분기보고서", {"1": "03", "3": "09", "4": "12"}[qm.group(1)]
    if "분기" in question:
        return "분기보고서", None
    return None, None


def resolve_periods(question: str, filings: list, today: date | None = None) -> list[str]:
    """회사의 보유 공시(filings: Filing 목록) 중 질문에 맞는 기간들. 아무 단서가 없으면 최신 1건."""
    today = today or date.today()
    have = sorted({f.period for f in filings}, reverse=True)
    if not have:
        return []
    years = _years(question, today)
    rtype, month = _report_type(question)
    out: list[str] = []
    for y in years or [None]:
        cands = [p for p in have if (y is None or p.startswith(str(y))) and (month is None or p.endswith("." + month))]
        if rtype == "분기보고서" and month is None:
            cands = [p for p in cands if p[-2:] in ("03", "09")]
        if cands:
            out.append(cands[0])                       # 그 해의 가장 최근 공시
    if re.search(r"최근\s*([2-8])\s*(개\s*)?(분기|기간|번)", question):
        n = int(re.search(r"최근\s*([2-8])", question).group(1))
        out = have[:n]
    return list(dict.fromkeys(out)) or ([] if years else have[:1])
