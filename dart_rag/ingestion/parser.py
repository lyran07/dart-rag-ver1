"""DART 공시 원문 XML → 섹션 트리(Section) / 표(Table).

DART XML 골격:
<DOCUMENT><BODY>
  <SECTION-1><TITLE>I. 회사의 개요</TITLE>
     <SECTION-2><TITLE>1. 회사의 개요</TITLE><P>...</P><TABLE>...</TABLE></SECTION-2>
  </SECTION-1>
</BODY></DOCUMENT>

파싱 규칙을 바꾸면 DartXmlParser.version 을 올린다 → parse 이후 단계만 다시 실행된다.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from ..domain import Block, Section, Table

CELL_TAGS = {"td", "th", "te", "tu"}  # TE/TU = DART 전용 데이터 셀(입력형/선택형)
CONTAINER_TAGS = {"body", "document", "part", "div", "table-group", "library", "html"}
UNIT_RE = re.compile(r"단위\s*[:：]\s*([^)\]）]+)")


def _clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def _has_section(tag: Tag) -> bool:
    return tag.find(lambda t: t.name and t.name.lower().startswith("section")) is not None


class TableParser:
    """<TABLE> → Table. rowspan/colspan 그리드 전개 + 다층 헤더 병합."""

    def parse(self, tbl: Tag) -> Table:
        unit = ""
        prev = tbl.find_previous_sibling()
        if prev is not None and prev.name in {"p", "table-group", "div"}:
            m = UNIT_RE.search(prev.get_text(" "))
            if m:
                unit = m.group(1).strip()
        grid, is_head = self._grid(tbl)
        header, rows = self._split_header(grid, is_head)
        return Table(header=header, rows=rows, caption="", unit=unit)

    @staticmethod
    def _grid(tbl: Tag) -> tuple[list[list[str]], list[bool]]:
        trs = [tr for tr in tbl.find_all("tr") if tr.find_parent("table") is tbl]
        grid: list[list[str]] = []
        is_head: list[bool] = []
        pending: dict[int, tuple[str, int]] = {}  # col -> (value, remaining rows)
        for tr in trs:
            cells = [c for c in tr.find_all(list(CELL_TAGS)) if c.find_parent("tr") is tr]
            row: list[str] = []
            col = ci = 0
            while ci < len(cells) or (col in pending and pending[col][1] > 0):
                if col in pending and pending[col][1] > 0:
                    v, r = pending[col]
                    row.append(v)
                    pending[col] = (v, r - 1)
                    col += 1
                    continue
                c = cells[ci]
                ci += 1
                txt = _clean(c.get_text(" "))
                cs = int(c.get("colspan", 1) or 1)
                rs = int(c.get("rowspan", 1) or 1)
                for k in range(cs):
                    row.append(txt)
                    if rs > 1:
                        pending[col + k] = (txt, rs - 1)
                col += cs
            if not any(row):
                continue
            grid.append(row)
            is_head.append(tr.find_parent("thead") is not None
                           or (bool(cells) and all(c.name == "th" for c in cells)))
        return grid, is_head

    @staticmethod
    def _split_header(grid: list[list[str]], is_head: list[bool]) -> tuple[list[str], list[list[str]]]:
        n_head = 0
        while n_head < len(grid) and is_head[n_head]:
            n_head += 1
        if n_head == 0 and len(grid) > 1:
            n_head = 1  # THEAD 없으면 첫 행을 헤더로
        header: list[str] = []
        if n_head:
            width = max(len(r) for r in grid[:n_head])
            for j in range(width):
                parts: list[str] = []
                for r in grid[:n_head]:
                    v = r[j] if j < len(r) else ""
                    if v and (not parts or parts[-1] != v):
                        parts.append(v)
                header.append(" / ".join(parts))
        return header, grid[n_head:]

    @staticmethod
    def is_textual(t: Table) -> bool:
        """설명문/단위표기를 1열 표로 감싼 경우 → 본문 텍스트로 처리."""
        allrows = ([t.header] if t.header else []) + t.rows
        if all(len([v for v in set(r) if v]) <= 1 for r in allrows):
            return True
        if len(t.rows) <= 1:  # '(기준일 : 2023년 12월 31일)' 식 1행 표: 수치가 없으면 본문
            joined = " ".join(v for r in allrows for v in r)
            return not re.search(r"\d{1,3}(?:,\d{3})+|\d+\.\d+|\b\d{4,}\b(?!\s*년)", joined)
        return False


class DartXmlParser:
    version = "p1"

    def __init__(self, table_parser: TableParser | None = None):
        self.tables = table_parser or TableParser()
        self._pending_unit: dict[int, str] = {}

    def parse(self, xml: str) -> Section:
        self._pending_unit = {}
        xml = re.sub(r"&(cr|crlf|nbsp);", " ", xml)   # DART 커스텀 엔티티
        xml = re.sub(r"<\?xml[^>]*\?>", "", xml)
        soup = BeautifulSoup(xml, "lxml")
        root = Section(title="ROOT", level=0)
        body = soup.find("body") or soup
        self._build(body, root)
        if not root.children:  # 섹션 밖에 떠 있는 표지/개요 텍스트도 보존
            self._collect_blocks(body, root)
        return root

    # ------------------------------------------------------------------
    def _build(self, node: Tag, parent: Section) -> None:
        for ch in node.children:
            if not isinstance(ch, Tag):
                continue
            name = ch.name.lower()
            if name.startswith("section"):
                m = re.match(r"section-?(\d+)", name)
                level = int(m.group(1)) if m else parent.level + 1
                title_tag = ch.find("title", recursive=False) or ch.find("title")
                title = _clean(re.sub(r"\s+", " ", title_tag.get_text(" "))) if title_tag else "(제목 없음)"
                sec = Section(title=title, level=level, parent=parent)
                parent.children.append(sec)
                self._collect_blocks(ch, sec)
                self._build(ch, sec)
            elif name in CONTAINER_TAGS or _has_section(ch):
                self._build(ch, parent)

    def _collect_blocks(self, node: Tag, sec: Section) -> None:
        """섹션 직속 자식 중 SECTION 을 제외한 콘텐츠를 순서대로 블록화."""
        key = id(sec)
        for ch in node.children:
            if not isinstance(ch, Tag):
                txt = _clean(str(ch))
                if txt:
                    sec.blocks.append(Block("text", txt))
                continue
            name = ch.name.lower()
            if name.startswith("section") or name == "title":
                continue
            if name == "table":
                t = self.tables.parse(ch)
                if not (t.rows or t.header):
                    continue
                if self.tables.is_textual(t):
                    txt = "\n".join(next((v for v in r if v), "")
                                    for r in ([t.header] if t.header else []) + t.rows).strip()
                    m = UNIT_RE.search(txt)
                    if m:
                        self._pending_unit[key] = m.group(1).strip()
                    if txt:
                        sec.blocks.append(Block("text", txt))
                    continue
                if not t.unit and self._pending_unit.get(key):
                    t.unit = self._pending_unit[key]
                self._pending_unit[key] = ""
                sec.blocks.append(Block("table", table=t))
            elif name in CONTAINER_TAGS or _has_section(ch) or ch.find("table") is not None:
                self._collect_blocks(ch, sec)
            else:
                txt = _clean(ch.get_text(" "))
                if txt:
                    m = UNIT_RE.search(txt)
                    if m:
                        self._pending_unit[key] = m.group(1).strip()
                    sec.blocks.append(Block("text", txt))

    # ------------------------------------------------------------------
    @staticmethod
    def toc(root: Section, max_level: int = 2) -> str:
        lines = []
        for s in root.walk():
            if 0 < s.level <= max_level:
                n_tbl = sum(1 for b in s.blocks if b.kind == "table")
                lines.append("  " * (s.level - 1) + f"- {s.title}  [text {len(s.plain_text())}자, 표 {n_tbl}개]")
        return "\n".join(lines)
