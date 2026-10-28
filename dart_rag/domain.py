"""도메인 모델: 모든 계층이 주고받는 데이터 타입.

계층 간에는 이 타입만 오간다. 다른 계층의 구현 클래스를 직접 들고 다니지 않는다.

  Filing          공시 1건의 식별 정보 (catalog / ingestion)
  Section/Block/Table  파싱된 보고서 트리 (parsing → chunking, wiki)
  Chunk           검색 단위 (chunking → embedding → index → retrieval)
  EmbeddingBatch  임베더 출력 (dense + 선택적 sparse)
  SearchFilter    메타데이터 필터 (retrieval → index)
  ScoredChunk     검색 결과 1개
  Answer          최종 응답
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


# ------------------------------------------------------------------ filing
@dataclass
class Filing:
    rcept_no: str
    corp_code: str
    corp_name: str
    report_nm: str
    rcept_dt: str
    is_latest: bool = True

    @property
    def report_type(self) -> str:
        for t in ("사업보고서", "반기보고서", "분기보고서"):
            if t in self.report_nm:
                return t
        return self.report_nm

    @property
    def period(self) -> str:
        """'사업보고서 (2023.12)' → '2023.12'"""
        m = re.search(r"\((\d{4}\.\d{2})\)", self.report_nm)
        return m.group(1) if m else self.rcept_dt[:4] + "." + self.rcept_dt[4:6]

    @property
    def is_amendment(self) -> bool:
        return "정정" in self.report_nm

    def doc_meta(self) -> dict[str, Any]:
        """청크 메타데이터의 공통 부분."""
        return {
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "report_type": self.report_type,
            "period": self.period,
            "year": self.period[:4],
            "rcept_no": self.rcept_no,
            "rcept_dt": self.rcept_dt,
            "report_nm": self.report_nm,
        }


# ------------------------------------------------------------------ parsed document
@dataclass
class Table:
    header: list[str]
    rows: list[list[str]]
    caption: str = ""
    unit: str = ""

    def to_markdown(self, max_rows: int | None = None) -> str:
        rows = self.rows if max_rows is None else self.rows[:max_rows]
        w = max([len(self.header)] + [len(r) for r in rows]) if (rows or self.header) else 0
        pad = lambda r: (r + [""] * w)[:w]  # noqa: E731
        lines = []
        if self.caption or self.unit:
            cap = f"**{self.caption}** " if self.caption else ""
            lines.append(f"{cap}{('(단위: ' + self.unit + ')') if self.unit else ''}".strip())
        hdr = pad(self.header) if self.header else [f"col{i + 1}" for i in range(w)]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * w)
        for r in rows:
            lines.append("| " + " | ".join(pad(r)) + " |")
        return "\n".join(lines)


@dataclass
class Block:
    """섹션 안의 순서 있는 콘텐츠 블록 (문단 또는 표)."""
    kind: str  # "text" | "table"
    text: str = ""
    table: Table | None = None


@dataclass
class Section:
    title: str
    level: int
    blocks: list[Block] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    parent: "Section | None" = field(default=None, repr=False)

    @property
    def path(self) -> list[str]:
        node, out = self, []
        while node is not None and node.level > 0:
            out.append(node.title)
            node = node.parent
        return list(reversed(out))

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def plain_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.kind == "text")

    # 직렬화 (parsed 아티팩트) -------------------------------------------
    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "level": self.level,
            "blocks": [
                {"kind": b.kind, "text": b.text, "table": asdict(b.table) if b.table else None}
                for b in self.blocks
            ],
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict, parent: "Section | None" = None) -> "Section":
        sec = cls(title=d["title"], level=d["level"], parent=parent)
        sec.blocks = [Block(b["kind"], b.get("text", ""), Table(**b["table"]) if b.get("table") else None)
                      for b in d.get("blocks", [])]
        sec.children = [cls.from_dict(c, sec) for c in d.get("children", [])]
        return sec


# ------------------------------------------------------------------ chunk
@dataclass
class Chunk:
    id: str
    text: str            # 임베딩 대상 (breadcrumb 포함)
    kind: str            # "text" | "table" | "table_kv"
    meta: dict = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """임베딩 캐시 키. 청크 ID가 아니라 '내용' 기준이라, 청킹 파라미터가 바뀌어도
        내용이 같으면 GPU 로 다시 계산하지 않는다."""
        return hashlib.md5(self.text.encode("utf-8")).hexdigest()

    @property
    def raw(self) -> str:
        return self.meta.get("raw", self.text)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(id=d["id"], text=d["text"], kind=d["kind"], meta=d.get("meta", {}))


# ------------------------------------------------------------------ embedding
SparseVec = dict[int, float]


@dataclass
class EmbeddingBatch:
    dense: np.ndarray                     # (n, dim) float32, L2 정규화
    sparse: list[SparseVec] | None = None  # bge-m3 lexical weights (없으면 None)

    def __len__(self) -> int:
        return int(self.dense.shape[0])


# ------------------------------------------------------------------ retrieval
@dataclass
class SearchFilter:
    """인덱스 백엔드에 독립적인 메타 필터. 값이 list 면 any-of."""
    must: dict[str, Any] = field(default_factory=dict)

    def without(self, *keys: str) -> "SearchFilter":
        return SearchFilter({k: v for k, v in self.must.items() if k not in keys})

    def matches(self, payload: dict) -> bool:
        for k, v in self.must.items():
            pv = payload.get(k)
            if isinstance(v, (list, tuple, set)):
                if isinstance(pv, list):
                    if not set(pv) & set(v):
                        return False
                elif pv not in v:
                    return False
            else:
                if isinstance(pv, list):
                    if v not in pv:
                        return False
                elif pv != v:
                    return False
        return True

    def __bool__(self) -> bool:
        return bool(self.must)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    why: dict[str, float] = field(default_factory=dict)  # 점수 구성 (디버깅/평가용)


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[dict]
    plan: dict
    llm: str = "none"
    timings: dict[str, float] = field(default_factory=dict)
    status: str = "answered"   # answered | no_evidence | extractive_fallback
