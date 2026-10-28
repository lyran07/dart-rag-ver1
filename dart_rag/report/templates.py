"""config/report_templates.yaml 로딩."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT

TEMPLATES_PATH = PROJECT_ROOT / "config" / "report_templates.yaml"

DEFAULTS = {"page_chars": 7000, "page_chars_compare": 4500, "max_pages": 4, "context_chars": 22000,
            "parallel": 7, "max_output_tokens": 1000, "reasoning": "none"}


@dataclass
class SectionSpec:
    id: str
    title: str
    instruction: str = ""
    routes: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    table: bool = False
    summary: bool = False
    report_types: list[str] = field(default_factory=list)   # 비어 있으면 모든 공시 유형


@dataclass
class ReportTemplate:
    id: str
    title: str
    sections: list[SectionSpec]
    options: dict = field(default_factory=lambda: dict(DEFAULTS))

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title,
                "sections": [{"id": s.id, "title": s.title, "summary": s.summary} for s in self.sections]}


def load_templates(path: str | Path | None = None) -> dict[str, ReportTemplate]:
    raw = yaml.safe_load(Path(path or TEMPLATES_PATH).read_text("utf-8")) or {}
    opts = {**DEFAULTS, **(raw.get("defaults") or {})}
    out: dict[str, ReportTemplate] = {}
    for tid, t in (raw.get("templates") or {}).items():
        secs = [SectionSpec(id=s["id"], title=s["title"], instruction=str(s.get("instruction", "")).strip(),
                            routes=list(s.get("routes") or []), keywords=list(s.get("keywords") or []),
                            table=bool(s.get("table")), summary=bool(s.get("summary")),
                            report_types=list(s.get("report_types") or []))
                for s in t.get("sections") or []]
        ids = [s.id for s in secs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"템플릿 {tid}: 섹션 id 가 중복됩니다 {ids}")
        out[tid] = ReportTemplate(tid, t.get("title", tid), secs, {**opts, **(t.get("options") or {})})
    return out
