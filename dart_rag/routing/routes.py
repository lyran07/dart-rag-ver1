"""config/wiki_routes.yaml 로딩."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROJECT_ROOT

ROUTES_PATH = PROJECT_ROOT / "config" / "wiki_routes.yaml"


@dataclass
class WikiRoute:
    id: str
    desc: str
    keywords: list[str] = field(default_factory=list)
    match: list[str] = field(default_factory=list)


@dataclass
class RouteTable:
    routes: list[WikiRoute]

    @property
    def by_id(self) -> dict[str, WikiRoute]:
        return {r.id: r for r in self.routes}

    def match_keywords(self, question: str) -> list[str]:
        """질문에 키워드가 들어 있는 섹션 id (긴 키워드가 먼저 맞은 순)."""
        q = re.sub(r"\s+", "", question.lower())
        hits: list[tuple[int, str]] = []
        for r in self.routes:
            best = max((len(k) for k in r.keywords if re.sub(r"\s+", "", k.lower()) in q), default=0)
            if best:
                hits.append((best, r.id))
        return [i for _, i in sorted(hits, key=lambda x: -x[0])]

    def patterns(self, ids: list[str]) -> list[str]:
        out: list[str] = []
        for i in ids:
            r = self.by_id.get(i)
            if r:
                out += r.match
        return out

    def describe(self) -> str:
        return "\n".join(f"- {r.id}: {r.desc}" for r in self.routes)


def load_routes(path: str | Path | None = None) -> RouteTable:
    import yaml
    p = Path(path or ROUTES_PATH)
    d = yaml.safe_load(p.read_text("utf-8")) if p.exists() else {}
    return RouteTable([WikiRoute(**{k: v for k, v in r.items() if k in WikiRoute.__dataclass_fields__})
                       for r in (d or {}).get("routes", [])])
