"""공시 특화 청킹. (로직은 기존 chunker.py 와 동일, 입출력 타입만 domain 으로 정리)

설계 원칙
1. breadcrumb 주입: `[삼성전자 | 사업보고서 2023.12 | II. 사업의 내용 > 4. 매출 및 수주상황]`
2. 섹션 = 청크 경계 (섹션을 섞지 않음), 문단 누적 + overlap
3. 표는 별도 청킹: 헤더 반복 + 글자 예산 row-group, 넓은 표는 열 그룹 분할
4. 표 행 선형화(table_kv): "매출액(제55기)=258,935,494 백만원" → 수치 질의용
5. parent_id(섹션 id)로 small-to-big

파라미터가 바뀌면 version 이 자동으로 바뀐다 → chunk 이후 단계만 재실행.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict

from ..config import ChunkingCfg
from ..domain import Chunk, Filing, Section, Table

SECTION_TAGS = [
    (r"회사의 개요|연혁|자본금|주식의 총수|정관", "company"),
    (r"사업의 개요|주요 제품|원재료|생산|매출|수주|위험관리|파생|연구개발|기타 참고", "business"),
    (r"재무에 관한|요약재무|연결재무|재무제표|주석|배당|증권의 발행", "financial"),
    (r"이사회|감사제도|주주총회|의결권", "governance"),
    (r"임원|직원|보수", "people"),
    (r"주주에 관한|최대주주|소액주주", "shareholder"),
    (r"계열회사|타법인출자", "affiliates"),
    (r"이해관계자|대주주.*거래|제재|소송", "related"),
]


def tag_section(path: list[str]) -> list[str]:
    joined = " ".join(path)
    return [t for pat, t in SECTION_TAGS if re.search(pat, joined)] or ["etc"]


def make_id(*parts: str) -> str:
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def section_id(rcept_no: str, path: list[str]) -> str:
    return make_id(rcept_no, *path)


def split_paragraphs(text: str, max_para: int = 800) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    out = []
    for p in paras:
        if len(p) <= max_para:
            out.append(p)
            continue
        for s in re.split(r"(?<=[.다\.]\s)", p):
            s = s.strip()
            if s:
                out.extend(s[i:i + max_para] for i in range(0, len(s), max_para))
    return out


class DartChunker:
    def __init__(self, cfg: ChunkingCfg | None = None):
        self.cfg = cfg or ChunkingCfg()

    @property
    def version(self) -> str:
        h = hashlib.md5(json.dumps(asdict(self.cfg), sort_keys=True).encode()).hexdigest()[:6]
        return f"c1-{h}"

    # ------------------------------------------------------------------
    def chunk(self, root: Section, filing: Filing) -> list[Chunk]:
        doc = filing.doc_meta()
        chunks: list[Chunk] = []
        for sec in root.walk():
            if sec.level == 0 and not sec.blocks:
                continue
            path = sec.path or ["표지"]
            crumb = f"[{doc['corp_name']} | {doc['report_type']} {doc['period']} | {' > '.join(path)}]"
            base = {
                **doc,
                "section_path": " > ".join(path),
                "section_title": path[-1],
                "section_level": sec.level,
                "tags": tag_section(path),
                "parent_id": section_id(doc["rcept_no"], path),
            }
            chunks += self._text_chunks(sec, crumb, base)
            t_idx = 0
            for b in sec.blocks:
                if b.kind == "table" and b.table:
                    chunks += self._table_chunks(b.table, t_idx, crumb, base)
                    t_idx += 1
        return chunks

    # ------------------------------------------------------------------
    def _text_chunks(self, sec: Section, crumb: str, meta: dict) -> list[Chunk]:
        paras = split_paragraphs(sec.plain_text())
        if not paras:
            return []
        cfg = self.cfg
        out: list[Chunk] = []
        buf: list[str] = []
        buf_len = idx = 0

        def flush():
            nonlocal buf, buf_len, idx
            if not buf:
                return
            body = "\n".join(buf)
            cid = make_id(meta["rcept_no"], meta["section_path"], "text", str(idx))
            out.append(Chunk(cid, f"{crumb}\n{body}", "text", {**meta, "chunk_index": idx, "raw": body}))
            idx += 1
            keep = buf[-cfg.overlap_paras:] if cfg.overlap_paras else []
            buf, buf_len = list(keep), sum(len(p) for p in keep)

        for p in paras:
            if buf and buf_len + len(p) > cfg.target_chars:
                flush()
            buf.append(p)
            buf_len += len(p)
            if buf_len > cfg.max_chars:
                flush()
        flush()
        return out

    def _table_chunks(self, t: Table, t_idx: int, crumb: str, meta: dict) -> list[Chunk]:
        mc = self.cfg.max_cols
        width = max([len(t.header)] + [len(r) for r in t.rows]) if (t.header or t.rows) else 0
        if width <= mc:
            return self._row_groups(t, t_idx, crumb, meta)
        out = []
        for ci, cs in enumerate(range(1, width, mc - 1)):  # 첫 열(라벨) 고정 + 열 그룹 분할
            cols = [0] + list(range(cs, min(cs + mc - 1, width)))
            pick = lambda r: [r[j] if j < len(r) else "" for j in cols]  # noqa: E731
            sub = Table(header=pick(t.header) if t.header else [], rows=[pick(r) for r in t.rows],
                        caption=t.caption, unit=t.unit)
            out += self._row_groups(sub, t_idx, crumb, meta, col_group=ci)
        return out

    def _row_groups(self, t: Table, t_idx: int, crumb: str, meta: dict, col_group: int = 0) -> list[Chunk]:
        cfg = self.cfg
        hdr = t.header
        groups: list[list[list[str]]] = []
        cur: list[list[str]] = []
        cur_len = 0
        for r in t.rows:
            rl = sum(len(v) + len(h) + 4 for h, v in zip(hdr, r)) + 20
            if cur and (cur_len + rl > cfg.table_chars or len(cur) >= cfg.table_rows_per_chunk):
                groups.append(cur)
                cur, cur_len = [], 0
            cur.append(r)
            cur_len += rl
        groups.append(cur)
        out: list[Chunk] = []
        start = 0
        for g, rows in enumerate(groups):
            sub = Table(header=hdr, rows=rows, caption=t.caption, unit=t.unit)
            md = sub.to_markdown()
            tmeta = {**meta, "table_index": t_idx, "row_start": start, "col_group": col_group,
                     "unit": t.unit, "table_header": " / ".join(hdr)[:300]}
            cid = make_id(meta["rcept_no"], meta["section_path"], f"table{t_idx}", f"{col_group}-{g}")
            out.append(Chunk(cid, f"{crumb}\n(표 {t_idx + 1}, 행 {start + 1}-{start + len(rows)})\n{md}",
                             "table", {**tmeta, "raw": md[:cfg.hard_cap]}))
            kv = self.linearize(sub)
            if kv and re.search(r"\d", kv):
                out.append(Chunk(make_id(cid, "kv"), f"{crumb}\n{kv[:cfg.hard_cap]}", "table_kv",
                                 {**tmeta, "raw": kv[:cfg.hard_cap]}))
            start += len(rows)
        return out

    @staticmethod
    def linearize(t: Table) -> str:
        if not t.header or not t.rows:
            return ""
        unit = f" {t.unit}" if t.unit else ""
        lines = []
        for r in t.rows:
            label = r[0] if r else ""
            pairs = [f"{label}({h})={v}{unit if re.search(r'[0-9]', v) else ''}"
                     for h, v in zip(t.header[1:], r[1:]) if v and h]
            if pairs:
                lines.append("; ".join(pairs))
        return "\n".join(lines)
