"""질의 분석: 질문 → QueryPlan (필터, 확장 질의, 섹션 힌트, 검색 분할).

LLM 호출 없음. 규칙 + config/lexicon.yaml.

multi_facet=False (기본 규칙, v0.2 초기 동작)
  첫 연도 하나, 회사 하나만 필터로 건다.
multi_facet=True
  - 연도가 여럿이면 연도별로, 회사가 여럿이면 회사별로 검색을 나눠서(facet) 각각 결과를 확보한다
    → "2023년 대비 2024년", "삼성전자와 SK하이닉스 비교" 가 한쪽 문서만 보는 문제 해결
  - 의도(전망·원인·추이) 키워드 → 서술형 섹션 힌트 + 같은 섹션 본문 확장
(LLM 기반 HyDE / 섹션 라우팅은 이 클래스를 상속해 analyze() 를 덮어쓰면 된다.)
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROJECT_ROOT
from ..domain import SearchFilter

LEXICON_PATH = PROJECT_ROOT / "config" / "lexicon.yaml"


@dataclass
class QueryPlan:
    question: str
    dense_text: str                 # 임베딩용 (원 질문)
    lexical_text: str               # BM25 용 (동의어 확장)
    filters: SearchFilter
    section_hints: list[str] = field(default_factory=list)
    relax_order: list[tuple[str, ...]] = field(default_factory=list)
    facets: list[dict] = field(default_factory=list)   # 나눠서 검색할 조건들 (비어 있으면 한 번에)
    intents: list[str] = field(default_factory=list)   # outlook | cause | trend | compare
    strict: bool = False            # True 면 사용자가 지정한 조건을 완화하지 않는다

    def filter_ladder(self, base: SearchFilter | None = None) -> list[SearchFilter]:
        """결과가 없으면 조건을 단계적으로 푼다: 전체 → 기간 해제 → 보고서종류 해제 → 연도 해제."""
        cur = base or self.filters
        out = [cur]
        if self.strict:
            return out
        for keys in self.relax_order:
            nxt = cur.without(*keys)
            if nxt.must != cur.must:
                out.append(nxt)
                cur = nxt
        return out

    def to_dict(self) -> dict:
        return {"filters": self.filters.must, "lexical_text": self.lexical_text,
                "section_hints": self.section_hints, "facets": self.facets, "intents": self.intents}


class QueryAnalyzer:
    def __init__(self, lexicon_path: Path | None = None, corp_names: list[str] | None = None,
                 latest_only: bool = True, multi_facet: bool = True, aliases: dict[str, str] | None = None):
        lex = self._load_lexicon(lexicon_path or LEXICON_PATH)
        self.synonyms = lex.get("synonyms", {}) or {}
        self.section_hints = lex.get("section_hints", {}) or {}
        self.narrative = lex.get("narrative_hints", {}) or {}
        self.corp_names = sorted(set(corp_names or []), key=len, reverse=True)
        self.latest_only = latest_only
        self.multi_facet = multi_facet
        self.aliases = dict(aliases or {})        # 별칭 → DART 등록명 (config/agent.yaml, 예: 현대차 → 현대자동차)

    @staticmethod
    def _load_lexicon(path: Path) -> dict:
        if not path.exists():
            return {}
        import yaml
        return yaml.safe_load(path.read_text("utf-8")) or {}

    def set_corp_names(self, names: list[str]) -> None:
        self.corp_names = sorted(set(names), key=len, reverse=True)

    # ------------------------------------------------------------------
    def find_corps(self, q: str) -> list[str]:
        """긴 이름부터, 겹치지 않게 (예: '삼성전자우' 가 있으면 '삼성전자' 로 중복 매칭 안 함)."""
        ql, used, out = q.lower(), [], []
        names = set(self.corp_names)
        cands = [(c, c) for c in self.corp_names] + [(a, t) for a, t in self.aliases.items() if t in names]
        for key, corp in sorted(cands, key=lambda x: len(x[0]), reverse=True):
            i = ql.find(key.lower())
            if i >= 0 and not any(a <= i < b for a, b in used) and corp not in [c for _, c in out]:
                used.append((i, i + len(key)))
                out.append((i, corp))
        return [c for _, c in sorted(out)]

    def analyze(self, q: str, extra: dict | None = None) -> QueryPlan:
        years = list(dict.fromkeys(re.findall(r"(20\d{2})\s*년?", q)))
        corps = self.find_corps(q)
        intents = self.intents(q) if self.multi_facet else []
        if self.multi_facet and len(corps) > 1:
            intents.append("compare")

        must = self.infer_filters(q)
        facets: list[dict] = []
        if self.multi_facet:
            if len(years) > 1:           # 여러 기간: 보고서 종류·기간 고정 해제, 연도별 facet
                for k in ("period", "report_type", "year"):
                    must.pop(k, None)
                must["year"] = years
            if len(corps) == 1:
                must["corp_name"] = corps[0]
            elif corps:
                must["corp_name"] = corps
            axes = []
            if len(corps) > 1:
                axes.append([("corp_name", c) for c in corps])
            if len(years) > 1:
                axes.append([("year", y) for y in years])
            facets = [dict(combo) for combo in itertools.product(*axes)] if axes else []
        elif corps:
            must["corp_name"] = corps[0]   # 기본 규칙: 첫 회사만
        if self.latest_only:
            must["is_latest"] = True
        extra = dict(extra or {})
        strict = bool(extra.pop("strict", False)) or bool(extra)   # 명시 조건이 있으면 완화하지 않음
        must.update(extra)

        hints = self.hints(q)
        for it in intents:
            hints += [str(s) for s in (self.narrative.get(it, {}) or {}).get("sections", [])]
        return QueryPlan(
            question=q,
            dense_text=q,
            lexical_text=self.expand(q),
            filters=SearchFilter(must),
            section_hints=list(dict.fromkeys(hints))[:8],
            relax_order=[("period",), ("report_type",), ("year",)],
            facets=facets,
            intents=intents,
            strict=strict,
        )

    @staticmethod
    def infer_filters(q: str) -> dict:
        """'2023년 사업보고서' → {report_type: 사업보고서, period: 2023.12}
        '2024년 3분기' → {report_type: 분기보고서, period: 2024.09}"""
        f: dict = {}
        ym = re.search(r"(20\d{2})\s*년?", q)
        y = ym.group(1) if ym else None
        quarter = re.search(r"([1-4])\s*분기", q)
        if "사업보고서" in q or re.search(r"연간|년\s*말|연말|기말", q):
            f["report_type"] = "사업보고서"
            if y:
                f["period"] = f"{y}.12"
        elif "반기" in q or "상반기" in q or (quarter and quarter.group(1) == "2"):
            f["report_type"] = "반기보고서"
            if y:
                f["period"] = f"{y}.06"
        elif "분기보고서" in q or quarter:
            f["report_type"] = "분기보고서"
            if y and quarter:
                f["period"] = f"{y}.{'03' if quarter.group(1) == '1' else '09'}"
        if y:
            f["year"] = y
        return f

    def intents(self, q: str) -> list[str]:
        return [name for name, spec in self.narrative.items()
                if any(str(k) in q for k in (spec or {}).get("keywords", []))]

    def expand(self, q: str) -> str:
        add: list[str] = []
        ql = q.lower()
        for key, vals in self.synonyms.items():
            if str(key).lower() in ql:
                add += [str(v) for v in vals]
        return q if not add else q + " " + " ".join(dict.fromkeys(add))

    def hints(self, q: str) -> list[str]:
        out: list[str] = []
        ql = q.lower()
        for key, vals in self.section_hints.items():
            if str(key).lower() in ql:
                out += [str(v) for v in vals]
        return list(dict.fromkeys(out))[:6]
