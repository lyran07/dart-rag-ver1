"""위키 형태 지식 구조 (첫 데모 dart-rag-demo 의 wiki.py 를 현재 도메인 타입에 맞게 이식).

wiki/
  index.json                         전체 페이지 인덱스(그래프: 노드+엣지)
  삼성전자/
    _company.md                      회사 허브 페이지 (보고서 목록·주제별 링크)
    사업보고서_2023.12/
      _report.md                     보고서 허브 (목차 → 섹션 페이지 링크)
      II.-사업의-내용__4.-매출-및-수주상황.md   섹션 페이지 (본문 + 표 + 관련 청크 id)
  _topics/
    financial.md                     주제(태그)별 크로스 리포트 페이지 (시계열 비교용)

설계 원칙
- 상위 구조(기업 → 보고서 → 섹션 → 주제)와 기업·기간·접수번호·청크 id 는 코드로 확정한다.
- 섹션 페이지는 청크의 parent(small-to-big) 와 같은 id 를 쓰므로 검색 인덱스와 분류 정보를 공유한다.
- LLM 요약은 선택 단계(summaries) 로만 덧붙이며 원문 페이지를 대체하지 않는다 (근거 chunk_ids 유지).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .domain import Chunk, Filing, Section
from .ingestion.chunker import section_id, tag_section


def _slug(s: str) -> str:
    s = re.sub(r"[^\w\-가-힣.]+", "-", s).strip("-")
    return s[:80]


class WikiBuilder:
    version = "w2"

    def __init__(self, root: str | Path = "wiki"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index = (json.loads(self.index_path.read_text("utf-8")) if self.index_path.exists()
                      else {"nodes": {}, "edges": []})
        self._dirty = False

    def _node(self, nid: str, **attrs):
        if "path" in attrs:                      # 위키 폴더 기준 상대경로로 저장 (폴더 이동·배포에 안전)
            attrs["path"] = self.rel(attrs["path"])
        self.index["nodes"][nid] = {**self.index["nodes"].get(nid, {}), **attrs}

    # ------------------------------------------------------------------ 경로 (상대 ↔ 절대)
    def rel(self, p: str | Path) -> str:
        """절대경로(예전 index 의 C:\\...\\wiki\\...)든 상대경로든 위키 폴더 기준 상대경로(posix)로."""
        s = str(p).replace("\\", "/")
        root = str(self.root.resolve()).replace("\\", "/").rstrip("/") + "/"
        if s.startswith(root):
            return s[len(root):]
        i = s.rfind("/wiki/")
        if i >= 0 and (":" in s[:3] or s.startswith("/")):
            return s[i + len("/wiki/"):]
        return s

    def abs(self, p: str | Path) -> Path:
        return self.root / self.rel(p)

    def _edge(self, a: str, b: str, rel: str):
        e = {"from": a, "to": b, "rel": rel}
        if e not in self.index["edges"]:
            self.index["edges"].append(e)

    # ------------------------------------------------------------------
    def add_report(self, root_sec: Section, filing: Filing, chunks: list[Chunk]) -> int:
        meta = filing.doc_meta()
        corp, rtype, period, rcept = meta["corp_name"], meta["report_type"], meta["period"], meta["rcept_no"]
        corp_dir = self.root / _slug(corp)
        rep_dir = corp_dir / _slug(f"{rtype}_{period}")
        rep_dir.mkdir(parents=True, exist_ok=True)

        corp_id, rep_id = f"corp:{corp}", f"report:{rcept}"
        self._node(corp_id, type="company", title=corp, corp_code=meta["corp_code"],
                   path=str(corp_dir / "_company.md"))
        self._node(rep_id, type="report", title=f"{corp} {rtype} ({period})", corp=corp, report_type=rtype,
                   period=period, rcept_no=rcept, rcept_dt=meta["rcept_dt"], is_latest=filing.is_latest,
                   path=str(rep_dir / "_report.md"))
        self._edge(corp_id, rep_id, "has_report")

        by_parent: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_parent.setdefault(c.meta.get("parent_id", ""), []).append(c)

        toc, n_pages = [], 0
        for sec in root_sec.walk():
            if sec.level == 0:
                continue
            path = sec.path
            page_id = section_id(rcept, path)
            fname = _slug("__".join(path)) + ".md"
            fpath = rep_dir / fname
            tags = tag_section(path)
            sec_chunks = by_parent.get(page_id, [])
            fpath.write_text(self._render_section(sec, meta, path, tags, sec_chunks), "utf-8")
            n_pages += 1
            self._node(f"section:{page_id}", type="section", title=" > ".join(path), corp=corp, report_type=rtype,
                       period=period, rcept_no=rcept, tags=tags, path=str(fpath),
                       n_chunks=len(sec_chunks), chunk_ids=[c.id for c in sec_chunks])
            self._edge(rep_id, f"section:{page_id}", "has_section")
            if sec.parent is not None and sec.parent.level > 0:
                self._edge(f"section:{section_id(rcept, sec.parent.path)}", f"section:{page_id}", "child")
            for t in tags:
                self._node(f"topic:{t}", type="topic", title=t, path=str(self.root / "_topics" / f"{t}.md"))
                self._edge(f"topic:{t}", f"section:{page_id}", "topic_of")
            toc.append("  " * (sec.level - 1) + f"- [{sec.title}]({fname})  `{','.join(tags)}`")

        (rep_dir / "_report.md").write_text(
            f"---\ntype: report\ncorp: {corp}\nreport_type: {rtype}\nperiod: {period}\nrcept_no: {rcept}\n"
            f"rcept_dt: {meta['rcept_dt']}\nis_latest: {str(filing.is_latest).lower()}\n---\n"
            f"# {corp} {rtype} ({period})\n\n- 회사 페이지: [{corp}](../_company.md)\n- 청크 수: {len(chunks)}\n"
            f"- DART 원문: https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept}\n\n## 목차\n" + "\n".join(toc) + "\n",
            "utf-8")
        self._write_company(corp, corp_dir)
        self._dirty = True
        return n_pages

    # ------------------------------------------------------------------
    def _render_section(self, sec: Section, meta: dict, path: list[str], tags: list[str], chunks: list[Chunk]) -> str:
        fm = (f"---\ntype: section\ncorp: {meta['corp_name']}\nreport_type: {meta['report_type']}\n"
              f"period: {meta['period']}\nrcept_no: {meta['rcept_no']}\nsection_path: {' > '.join(path)}\n"
              f"tags: [{', '.join(tags)}]\nchunk_ids: [{', '.join(c.id for c in chunks)}]\n---\n")
        out = [fm, f"# {' > '.join(path)}", "",
               f"> 출처: {meta['corp_name']} {meta['report_type']} ({meta['period']}) · 접수번호 {meta['rcept_no']}"
               f" · [보고서 목차](_report.md)", ""]
        if sec.parent is not None and sec.parent.level > 0:
            out.append(f"⬆ 상위: [{sec.parent.title}]({_slug('__'.join(sec.parent.path))}.md)")
        if sec.children:
            out.append("⬇ 하위: " + " · ".join(f"[{c.title}]({_slug('__'.join(c.path))}.md)" for c in sec.children))
        out.append("")
        t_idx = 0
        for b in sec.blocks:
            if b.kind == "text" and b.text:
                out += [b.text, ""]
            elif b.table is not None:
                t_idx += 1
                out += [f"### 표 {t_idx}", b.table.to_markdown(), ""]
        return "\n".join(out)

    def _write_company(self, corp: str, corp_dir: Path):
        reps = [n for n in self.index["nodes"].values() if n.get("type") == "report" and n.get("corp") == corp]
        reps.sort(key=lambda n: (n["period"], n.get("rcept_dt", "")), reverse=True)
        lines = [f"---\ntype: company\ncorp: {corp}\n---", f"# {corp}", "", "## 보고서"]
        for r in reps:
            flag = "" if r.get("is_latest", True) else "  (정정 이전본)"
            lines.append(f"- [{r['title']}]({Path(r['path']).parent.name}/_report.md)  접수번호 {r['rcept_no']}{flag}")
        topics = sorted({t for n in self.index["nodes"].values()
                         if n.get("type") == "section" and n.get("corp") == corp for t in n.get("tags", [])})
        lines += ["", "## 주제별", ""] + [f"- [{t}](../_topics/{t}.md)" for t in topics]
        (corp_dir / "_company.md").write_text("\n".join(lines) + "\n", "utf-8")

    def _write_topics(self):
        tdir = self.root / "_topics"
        tdir.mkdir(exist_ok=True)
        secs = [n for n in self.index["nodes"].values() if n.get("type") == "section"]
        for t in {t for n in secs for t in n.get("tags", [])}:
            lines = [f"# 주제: {t}", "", "보고서 간 동일 섹션을 시계열로 비교할 때 사용.", ""]
            for n in sorted((s for s in secs if t in s.get("tags", [])),
                            key=lambda s: (s["title"], s["period"]), reverse=True):
                rel = self.rel(n["path"])
                lines.append(f"- [{n['corp']} {n['report_type']} {n['period']} · {n['title']}](../{rel})")
            (tdir / f"{t}.md").write_text("\n".join(lines) + "\n", "utf-8")

    def _link_periods(self) -> None:
        """같은 회사·같은 섹션 제목을 기간순으로 잇는다 (prev_period 엣지) → 추이 질문용."""
        self.index["edges"] = [e for e in self.index["edges"] if e.get("rel") != "prev_period"]
        groups: dict[tuple, list] = {}
        for nid, n in self.index["nodes"].items():
            if n.get("type") == "section" and n.get("is_latest", True) is not False:
                groups.setdefault((n.get("corp"), n.get("title")), []).append((n.get("period", ""), nid))
        for items in groups.values():
            items.sort()
            for (_, a), (_, b) in zip(items, items[1:]):
                self.index["edges"].append({"from": b, "to": a, "rel": "prev_period"})

    def flush(self) -> None:
        """주제 페이지·인덱스를 기록 (여러 보고서 처리 후 1회)."""
        if not self._dirty:
            return
        for n in self.index["nodes"].values():   # 예전 index 의 절대경로도 상대경로로 정리
            if "path" in n:
                n["path"] = self.rel(n["path"])
        self._link_periods()
        self._write_topics()
        self.index_path.write_text(json.dumps(self.index, ensure_ascii=False, indent=1), "utf-8")
        self._dirty = False

    # ------------------------------------------------------------------ 조회
    def get_section_text(self, parent_id: str) -> str | None:
        n = self.index["nodes"].get(f"section:{parent_id}")
        return self.abs(n["path"]).read_text("utf-8") if n else None

    # ------------------------------------------------------------------ 메타데이터 조회 (위키 경로)
    def reports_of(self, corp: str, report_type: str | None = None) -> list[dict]:
        rs = [n for n in self.index["nodes"].values() if n.get("type") == "report" and n.get("corp") == corp
              and n.get("is_latest", True) and (report_type is None or n.get("report_type") == report_type)]
        return sorted(rs, key=lambda r: (r["period"], r.get("rcept_dt", "")), reverse=True)

    def find_sections(self, corp: str, period: str, patterns: list[str]) -> list[dict]:
        """회사·기간·섹션 제목 패턴(부분 문자열)으로 섹션 노드를 찾는다. 검색·임베딩 없음."""
        cp = getattr(self, "_by_cp", None)
        if cp is None or getattr(self, "_by_cp_n", -1) != len(self.index["nodes"]):
            cp = {}
            for nid, n in self.index["nodes"].items():
                if n.get("type") == "section":
                    cp.setdefault((n.get("corp"), n.get("period")), []).append((nid, n))
            self._by_cp, self._by_cp_n = cp, len(self.index["nodes"])
        out = [{**n, "id": nid} for nid, n in cp.get((corp, period), [])
               if any(p in n.get("title", "") for p in patterns)]
        return sorted(out, key=lambda n: n["title"])

    def page_text(self, node: dict) -> str:
        """섹션 페이지 본문 (front matter 제외)."""
        t = self.abs(node["path"]).read_text("utf-8")
        if t.startswith("---"):
            j = t.find("\n---", 3)
            if j >= 0:
                t = t[j + 4:].lstrip("\n")
        return t

    def company(self, corp: str) -> dict | None:
        n = self.index["nodes"].get(f"corp:{corp}")
        if not n:
            return None
        reports = sorted((r for r in self.index["nodes"].values() if r.get("type") == "report" and r.get("corp") == corp),
                         key=lambda r: (r["period"], r.get("rcept_dt", "")), reverse=True)
        return {"corp": corp, "corp_code": n.get("corp_code"), "reports": reports}

    def sections(self, rcept_no: str) -> list[dict]:
        return [n for n in self.index["nodes"].values() if n.get("type") == "section" and n.get("rcept_no") == rcept_no]
