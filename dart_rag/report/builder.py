"""보고서 생성기 (공시 1건 = 보고서 1건, 선택적으로 비교 공시 1건).

  start(corp, period, compare_period, template) → job id
     저장된 보고서(data/reports/)가 있고 버전이 같으면 즉시 완료 상태로 돌려준다 (LLM 호출 0회).
     없으면 백그라운드 스레드에서 생성:
       섹션마다 위키 섹션 페이지 수집(검색·임베딩 없음) → LLM 1회(JSON) → 인용 검증 · 표 숫자 대조
       일반 섹션은 병렬로, 종합 의견(summary) 섹션은 나머지가 끝난 뒤 그 결과만 읽고 쓴다.
     오류 없이 끝나면 data/reports/ 에 저장 → 다음 요청부터 즉시.
  get(job id) → 진행 상황과 완성된 섹션 (웹은 이걸 주기적으로 조회해 완성된 섹션부터 보여준다)
  run(...)    → 같은 작업을 동기로 (파이프라인 report 단계 · CLI · 테스트)

버전: 프롬프트·템플릿·모델이 바뀌면 version 이 바뀌어 저장본을 쓰지 않고 다시 만든다.
프로토타입 범위: 진행 중인 작업 상태는 서버 메모리에만 둔다. 재무 수치는 요약재무정보 위키 페이지를
LLM 이 읽어 정리한다 (DART 재무제표 API 로 숫자를 구조화하는 것은 다음 단계).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

from ..agent.llm import parse_final_json
from ..agent.loop import verify_evidence
from ..routing.auto import select_relevant
from .templates import ReportTemplate, SectionSpec, load_templates

SECTION_SCHEMA = {
    "type": "json_schema",
    "name": "report_section",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["conclusion", "key_points", "interpretation", "table_markdown", "evidence"],
        "properties": {
            "conclusion": {"type": "string", "description": "이 섹션의 핵심 1~2문장"},
            "key_points": {"type": "array", "items": {"type": "string"}, "description": "공시에서 확인되는 사실·수치"},
            "interpretation": {"type": "array", "items": {"type": "string"}, "description": "사실로부터의 해석"},
            "table_markdown": {"type": "string", "description": "요구된 경우만 마크다운 표, 아니면 빈 문자열"},
            "evidence": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["chunk_id", "quote"],
                "properties": {"chunk_id": {"type": "string"}, "quote": {"type": "string"}}}},
        },
    },
}

SECTION_SYSTEM = """당신은 한국 상장회사 DART 공시로 기업분석 보고서의 한 섹션을 쓰는 애널리스트입니다.
아래 [근거] 로 주어진 공시 원문만 사용합니다. 기억이나 추측으로 쓰지 않습니다.
- 수치는 원문 값을 그대로 쓰고 연결/별도, 기간(제N기, 누적·3개월), 단위를 함께 적습니다.
  표 바로 위의 소제목(예: "가. 요약연결재무정보")으로 연결/별도를 구분합니다. 단위가 보이지 않으면 "단위 미표시"라고 적습니다.
- 비교 기간이 주어지면 기준 기간과 비교 기간을 구분해 적고, 같은 기준(연결/별도, 누적/3개월)끼리만 비교합니다.
- key_points 에는 원문에서 확인되는 사실만, interpretation 에는 그로부터의 해석만 적습니다.
- table_markdown 은 [섹션 지시] 에 표가 요구될 때만 마크다운 표로 쓰고, 표 안의 숫자는 원문 값을 그대로 옮깁니다
  (직접 계산한 값을 넣지 않습니다). 표가 필요 없으면 빈 문자열로 둡니다.
- conclusion·key_points·interpretation 문장 안에서 근거를 evidence 배열 순서 번호 [1], [2] 로 인용합니다.
  evidence 의 chunk_id 는 [근거] 의 id 를 그대로, quote 에는 근거 원문 일부를 숫자까지 그대로 200자 이내로 옮깁니다.
- 근거로 쓸 내용이 없으면 conclusion 에 무엇이 없는지 적고 key_points·evidence 를 비웁니다.
- 보고서 문체(~다)로, 한국어로 간결하게 씁니다."""

SUMMARY_SYSTEM = """당신은 기업분석 보고서의 마지막 섹션(종합 의견)을 쓰는 애널리스트입니다.
아래 [섹션 요약] 은 같은 보고서의 앞 섹션들이 공시 원문을 근거로 정리한 내용입니다.
- [섹션 요약] 에 있는 내용만 사용하고, 새 사실이나 수치를 만들지 않습니다.
- key_points 에는 보고서 전체에서 중요한 사실 3~5개를 쓰고 끝에 출처 섹션 이름을 괄호로 붙입니다.
- interpretation 에는 투자자가 점검할 포인트와 해석을 씁니다. 매수·매도 같은 투자 권유는 쓰지 않습니다.
- 이 섹션은 해석이므로 evidence 는 빈 배열, table_markdown 은 빈 문자열로 둡니다.
- 보고서 문체(~다)로, 한국어로 간결하게 씁니다."""

_NUM = re.compile(r"\d[\d,]*\.?\d*")


def unverified_table_numbers(table_md: str, texts: list[str]) -> list[str]:
    """표 안의 숫자 중 근거 원문 어디에도 없는 것 (콤마 유무 차이는 같은 숫자로 본다).
    연도·기수처럼 짧은 숫자(3자리 미만)는 건너뛴다."""
    if not table_md.strip():
        return []
    hay = "\n".join(texts)
    hay_nc = hay.replace(",", "")
    out = []
    for raw in _NUM.findall(table_md):
        n = raw.rstrip(".,")
        digits = n.replace(",", "").replace(".", "")
        if len(digits) < 3:
            continue
        if n in hay or n.replace(",", "") in hay_nc:
            continue
        out.append(n)
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ 결과 타입
@dataclass
class SectionResult:
    id: str
    title: str
    kind: str = "section"           # section | summary
    status: str = "pending"         # pending | running | done | no_evidence | error
    conclusion: str = ""
    key_points: list[str] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)
    table_markdown: str = ""
    sources: list[dict] = field(default_factory=list)
    check: dict = field(default_factory=dict)      # 인용 검증 · 표 숫자 대조 결과
    diag: dict = field(default_factory=dict)       # 근거 수·글자 수·잘림·시간·토큰 (이슈 파악용)
    error: str | None = None


@dataclass
class ReportJob:
    id: str
    template: str
    title: str
    corp: str
    period: str
    compare_period: str | None
    base_report: dict
    compare_report: dict | None
    sections: list[SectionResult]
    status: str = "running"         # running | done | error
    created: float = field(default_factory=time.time)
    finished: float | None = None
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "usd": 0.0})
    error: str | None = None
    version: str = ""
    from_cache: bool = False
    generated_sec: float | None = None   # 처음 생성할 때 걸린 시간 (저장본을 열어도 유지)

    @classmethod
    def from_dict(cls, d: dict) -> "ReportJob":
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["sections"] = [SectionResult(**{k: v for k, v in s.items() if k in SectionResult.__dataclass_fields__})
                         for s in d.get("sections", [])]
        return cls(**d)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["progress"] = {"done": sum(s.status not in ("pending", "running") for s in self.sections),
                         "total": len(self.sections)}
        d["elapsed"] = round((self.finished or time.time()) - self.created, 1)
        return d


# ------------------------------------------------------------------ 생성기
class ReportBuilder:
    MAX_JOBS = 50

    def __init__(self, service, llm=None, templates: dict[str, ReportTemplate] | None = None,
                 today: date | None = None, root: str | Path | None = None):
        self.svc = service
        self.templates = templates if templates is not None else load_templates()
        self._llm = llm
        self.today = today
        self.root = Path(root) if root else Path(service.settings.storage.root_path) / "reports"
        self.jobs: dict[str, ReportJob] = {}
        self._lock = threading.Lock()
        self._runner = ThreadPoolExecutor(max_workers=2, thread_name_prefix="report")

    # ------------------------------------------------------------------ LLM
    def llm(self, tpl: ReportTemplate):
        if self._llm is None:
            from ..agent.llm import make_llm
            o = tpl.options
            self._llm = make_llm(self.svc.agent_config, backend="responses", reasoning=o.get("reasoning", "none"),
                                 max_output_tokens=int(o.get("max_output_tokens", 1800)), timeout=60)
        return self._llm

    # ------------------------------------------------------------------ 버전·저장
    def model_name(self) -> str:
        if self._llm is not None:
            return str(getattr(self._llm, "model", "?"))
        cfg = self.svc.agent_config
        return str((cfg.backends.get("responses") or {}).get("model", "?"))

    def version(self, template: str = "company_analysis") -> str:
        """프롬프트·템플릿·모델 → 해시. 하나라도 바뀌면 저장본을 다시 만든다."""
        tpl = self.templates[template]
        key = json.dumps([SECTION_SYSTEM, SUMMARY_SYSTEM, SECTION_SCHEMA, tpl.options, self.model_name(),
                          [vars(sp) for sp in tpl.sections]], ensure_ascii=False, sort_keys=True, default=str)
        return "r1." + hashlib.md5(key.encode("utf-8")).hexdigest()[:8]

    def cache_path(self, corp_code: str, base_rcept: str, compare_rcept: str | None, template: str) -> Path:
        name = base_rcept + (f"__vs_{compare_rcept}" if compare_rcept else "") + f".{template}.json"
        return self.root / (corp_code or "_") / name

    def _job_cache_path(self, job: ReportJob) -> Path:
        return self.cache_path(self._corp_code(job.corp), job.base_report["rcept_no"],
                               (job.compare_report or {}).get("rcept_no"), job.template)

    def _corp_code(self, corp: str) -> str:
        n = self.svc.wiki.index["nodes"].get(f"corp:{corp}") or {}
        return n.get("corp_code") or ""

    def load_cached(self, job: ReportJob) -> ReportJob | None:
        p = self._job_cache_path(job)
        if not p.exists():
            return None
        try:
            d = json.loads(p.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        if d.get("version") != job.version:
            return None
        cached = ReportJob.from_dict(d)
        cached.id, cached.from_cache = job.id, True
        return cached

    def save(self, job: ReportJob) -> Path:
        p = self._job_cache_path(job)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = job.to_dict()
        d.pop("progress", None)
        d.pop("elapsed", None)
        d["from_cache"] = False
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), "utf-8")
        os.replace(tmp, p)
        return p

    # ------------------------------------------------------------------ 시작·조회
    def reports_of(self, corp: str) -> list[dict]:
        return self.svc.wiki.reports_of(corp)

    def default_compare(self, corp: str, period: str) -> str | None:
        """전년 동기 보고서 (예: 2026.06 → 2025.06). 없으면 None."""
        want = f"{int(period[:4]) - 1}{period[4:]}"
        return want if any(r["period"] == want for r in self.reports_of(corp)) else None

    def has_cached(self, corp: str, period: str, compare_period: str | None,
                   template: str = "company_analysis") -> bool:
        reps = {r["period"]: r for r in self.reports_of(corp)}
        if period not in reps or (compare_period and compare_period not in reps):
            return False
        p = self.cache_path(self._corp_code(corp), reps[period]["rcept_no"],
                            reps[compare_period]["rcept_no"] if compare_period else None, template)
        if not p.exists():
            return False
        try:
            return json.loads(p.read_text("utf-8")).get("version") == self.version(template)
        except (OSError, ValueError):
            return False

    def _validate(self, corp: str, period: str, compare_period: str | None, template: str):
        if template not in self.templates:
            raise ValueError(f"알 수 없는 템플릿: {template} (가능: {sorted(self.templates)})")
        reps = {r["period"]: r for r in self.reports_of(corp)}
        if not reps:
            raise ValueError(f"위키에 보고서가 없는 기업입니다: {corp}")
        if period not in reps:
            raise ValueError(f"{corp} 의 {period} 보고서가 없습니다 (보유: {', '.join(sorted(reps, reverse=True))})")
        if compare_period:
            if compare_period == period:
                raise ValueError("비교 기간이 기준 기간과 같습니다.")
            if compare_period not in reps:
                raise ValueError(f"{corp} 의 {compare_period} 보고서가 없습니다")
        return reps[period], (reps[compare_period] if compare_period else None)

    def _new_job(self, corp, period, compare_period, template, use_cache: bool = True) -> ReportJob:
        base, comp = self._validate(corp, period, compare_period, template)
        tpl = self.templates[template]
        job = ReportJob(id=uuid.uuid4().hex[:12], template=template, title=tpl.title, corp=corp, period=period,
                        compare_period=compare_period or None,
                        base_report={k: base.get(k) for k in ("title", "report_type", "period", "rcept_no")},
                        compare_report=({k: comp.get(k) for k in ("title", "report_type", "period", "rcept_no")}
                                        if comp else None),
                        sections=[SectionResult(s.id, s.title, "summary" if s.summary else "section")
                                  for s in tpl.sections],
                        version=self.version(template))
        cached = self.load_cached(job) if use_cache else None
        if cached is not None:
            job = cached
        with self._lock:
            self.jobs[job.id] = job
            for old in sorted(self.jobs.values(), key=lambda j: j.created)[:-self.MAX_JOBS]:
                self.jobs.pop(old.id, None)
        return job

    def start(self, corp: str, period: str, compare_period: str | None = None,
              template: str = "company_analysis") -> str:
        job = self._new_job(corp, period, compare_period, template)
        if not job.from_cache:
            self._runner.submit(self._execute, job)
        return job.id

    def run(self, corp: str, period: str, compare_period: str | None = None,
            template: str = "company_analysis", use_cache: bool = True) -> ReportJob:
        job = self._new_job(corp, period, compare_period, template, use_cache)
        if not job.from_cache:
            self._execute(job)
        return job

    def get(self, job_id: str) -> ReportJob | None:
        return self.jobs.get(job_id)

    # ------------------------------------------------------------------ 실행
    def _execute(self, job: ReportJob) -> None:
        tpl = self.templates[job.template]
        specs = {s.id: s for s in tpl.sections}
        normal = [r for r in job.sections if r.kind == "section"]
        summaries = [r for r in job.sections if r.kind == "summary"]
        try:
            with ThreadPoolExecutor(max_workers=max(1, int(tpl.options.get("parallel", 4)))) as ex:
                list(ex.map(lambda r: self._section(job, tpl, specs[r.id], r), normal))
            for r in summaries:
                self._summary(job, tpl, specs[r.id], r)
            job.status = "done"
        except Exception as e:  # noqa: BLE001  섹션 오류는 섹션 안에서 처리, 여기는 예상 밖 오류만
            job.status, job.error = "error", f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            job.finished = time.time()
            job.generated_sec = round(job.finished - job.created, 1)
        if job.status == "done" and not any(r.status == "error" for r in job.sections):
            try:
                self.save(job)     # 섹션 오류(API 실패 등)가 있으면 저장하지 않는다 → 다음 요청에서 다시 시도
            except OSError as e:
                job.error = f"저장 실패: {e}"

    def _add_usage(self, job: ReportJob, u: dict) -> None:
        with self._lock:
            job.usage["input_tokens"] += u.get("input_tokens", 0) or 0
            job.usage["output_tokens"] += u.get("output_tokens", 0) or 0
            if u.get("usd") is None or job.usage["usd"] is None:
                job.usage["usd"] = None
            else:
                job.usage["usd"] = round(job.usage["usd"] + u["usd"], 6)

    # ------------------------------------------------------------------ 근거 수집
    def gather(self, job: ReportJob, tpl: ReportTemplate, spec: SectionSpec) -> tuple[list[dict], dict]:
        wiki, routes = self.svc.wiki, self.svc.routes
        by_id = routes.by_id
        pats = routes.patterns(spec.routes)
        kws = spec.keywords or [k for rid in spec.routes for k in (by_id[rid].keywords if rid in by_id else [])]
        periods = [job.period] + ([job.compare_period] if job.compare_period else [])
        o = tpl.options
        page_chars = int(o["page_chars_compare"] if job.compare_period else o["page_chars"])
        items, diag = [], {"pages_found": 0, "pages_used": 0, "pages_trimmed": 0, "missing_routes": [],
                           "periods_without_pages": []}
        diag["missing_routes"] = [rid for rid in spec.routes if rid not in by_id]
        for period in periods:
            nodes = wiki.find_sections(job.corp, period, pats) if pats else []
            diag["pages_found"] += len(nodes)
            if not nodes:
                diag["periods_without_pages"].append(period)
            for n in nodes[: int(o["max_pages"])]:
                page = wiki.page_text(n)
                text = select_relevant(page, " ".join(kws) or spec.title, kws, page_chars)
                diag["pages_trimmed"] += len(text) < len(page)
                items.append({"id": f"wiki:{n['id'].split(':', 1)[1]}", "text": text,
                              "source": f"{n['corp']} {n['report_type']}({n['period']}) · {n['title']}",
                              "rcept_no": n.get("rcept_no"), "period": n.get("period"),
                              "section": n.get("title"), "kind": "wiki"})
        return items, diag

    # ------------------------------------------------------------------ 섹션 작성
    def _period_line(self, job: ReportJob) -> str:
        line = f"기준: {job.base_report['title']}"
        if job.compare_report:
            line += f" / 비교: {job.compare_report['title']}"
        return line

    def _section(self, job: ReportJob, tpl: ReportTemplate, spec: SectionSpec, r: SectionResult) -> None:
        r.status = "running"
        t0 = time.perf_counter()
        try:
            items, diag = self.gather(job, tpl, spec)
            limit = int(tpl.options["context_chars"])
            parts, used, dropped = [], 0, 0
            for it in items:
                block = f"<근거 id={it['id']}>\n출처: {it['source']}\n{it['text']}\n</근거>"
                if used + len(block) > limit:
                    dropped += 1
                    continue
                parts.append(block)
                used += len(block)
            diag.update(n_context=len(parts), context_chars=used, dropped_for_budget=dropped,
                        gather_sec=round(time.perf_counter() - t0, 3))
            r.diag = diag
            if not parts:
                r.status = "no_evidence"
                r.conclusion = "이 보고서에서 해당 섹션의 공시 내용을 찾지 못했다."
                return
            user = (f"오늘 날짜: {(self.today or date.today()).isoformat()}\n"
                    f"회사: {job.corp}\n{self._period_line(job)}\n"
                    f"보고서 섹션: {spec.title}\n[섹션 지시]\n{spec.instruction}"
                    + ("\n표를 반드시 작성한다 (table_markdown)." if spec.table else "\n표는 쓰지 않는다 (table_markdown 은 빈 문자열).")
                    + "\n\n[근거]\n" + "\n\n".join(parts))
            llm = self.llm(tpl)
            conv = llm.start(SECTION_SYSTEM, user)
            st = llm.step(conv, None, SECTION_SCHEMA, allow_tools=False)
            self._add_usage(job, st.usage or {})
            data = parse_final_json(st.final_text)
            seen = {it["id"]: it for it in items}
            sources, check = verify_evidence(data.get("evidence") or [], seen)
            table = str(data.get("table_markdown") or "").strip()
            bad_nums = unverified_table_numbers(table, [it["text"] for it in items]) if table else []
            check["table_unverified_numbers"] = bad_nums
            r.conclusion = str(data.get("conclusion", "")).strip()
            r.key_points = [str(x) for x in data.get("key_points") or []]
            r.interpretation = [str(x) for x in data.get("interpretation") or []]
            r.table_markdown = table
            r.sources, r.check = sources, check
            r.diag.update(llm_sec=round(st.seconds, 2), input_tokens=(st.usage or {}).get("input_tokens"),
                          output_tokens=(st.usage or {}).get("output_tokens"))
            r.status = "done" if sources else "no_evidence"
        except Exception as e:  # noqa: BLE001  한 섹션 실패가 보고서 전체를 막지 않게
            r.status, r.error = "error", f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            r.diag["total_sec"] = round(time.perf_counter() - t0, 2)

    def _summary(self, job: ReportJob, tpl: ReportTemplate, spec: SectionSpec, r: SectionResult) -> None:
        r.status = "running"
        t0 = time.perf_counter()
        try:
            done = [s for s in job.sections if s.kind == "section" and s.status == "done"]
            if not done:
                r.status, r.conclusion = "no_evidence", "앞 섹션에서 정리된 내용이 없어 종합 의견을 쓰지 않았다."
                return
            blocks = []
            for s in done:
                lines = [f"## {s.title}", s.conclusion] + [f"- {x}" for x in s.key_points]
                if s.interpretation:
                    lines += ["(해석)"] + [f"- {x}" for x in s.interpretation]
                blocks.append(re.sub(r"\[\d+\]", "", "\n".join(lines)))
            user = (f"회사: {job.corp}\n{self._period_line(job)}\n[섹션 지시]\n{spec.instruction}\n\n"
                    "[섹션 요약]\n" + "\n\n".join(blocks))
            llm = self.llm(tpl)
            conv = llm.start(SUMMARY_SYSTEM, user)
            st = llm.step(conv, None, SECTION_SCHEMA, allow_tools=False)
            self._add_usage(job, st.usage or {})
            data = parse_final_json(st.final_text)
            r.conclusion = str(data.get("conclusion", "")).strip()
            r.key_points = [str(x) for x in data.get("key_points") or []]
            r.interpretation = [str(x) for x in data.get("interpretation") or []]
            r.diag = {"n_sections_used": len(done), "context_chars": len(user), "llm_sec": round(st.seconds, 2),
                      "input_tokens": (st.usage or {}).get("input_tokens"),
                      "output_tokens": (st.usage or {}).get("output_tokens")}
            r.status = "done"
        except Exception as e:  # noqa: BLE001
            r.status, r.error = "error", f"{type(e).__name__}: {str(e)[:300]}"
        finally:
            r.diag["total_sec"] = round(time.perf_counter() - t0, 2)


# ------------------------------------------------------------------ 마크다운
def to_markdown(job: ReportJob) -> str:
    out = [f"# {job.corp} {job.title}", ""]
    meta = f"기준 보고서: {job.base_report['title']} (접수번호 {job.base_report['rcept_no']})"
    if job.compare_report:
        meta += f"  \n비교 보고서: {job.compare_report['title']} (접수번호 {job.compare_report['rcept_no']})"
    out += [meta, "", "> DART 공시 원문만을 근거로 자동 생성한 초안입니다. 수치는 반드시 원문으로 확인하세요. "
                      "'해석' 항목과 종합 의견은 공시 내용에서 추론한 것이며 투자 권유가 아닙니다.", ""]
    for s in job.sections:
        out += [f"## {s.title}", ""]
        if s.status == "error":
            out += [f"(작성 실패: {s.error})", ""]
            continue
        if s.conclusion:
            out += [s.conclusion, ""]
        if s.table_markdown:
            out += [s.table_markdown, ""]
        if s.key_points:
            out += ["**주요 내용**", ""] + [f"- {x}" for x in s.key_points] + [""]
        if s.interpretation:
            out += ["**해석**", ""] + [f"- {x}" for x in s.interpretation] + [""]
        warn = []
        if s.check.get("n_unverified"):
            warn.append(f"원문과 대조되지 않은 근거 {s.check['n_unverified']}건")
        if s.check.get("table_unverified_numbers"):
            warn.append("표에 원문에서 찾지 못한 숫자: " + ", ".join(s.check["table_unverified_numbers"][:8]))
        if warn:
            out += ["> ⚠ " + " · ".join(warn), ""]
        if s.sources:
            out += ["<details><summary>근거</summary>", ""]
            for src in s.sources:
                mark = "" if src.get("verified") else " (미검증)"
                url = f" — [DART 원문]({src['url']})" if src.get("url") else ""
                out.append(f"- [{src['n']}] {src.get('source')}{mark}{url}  \n  \"{src.get('quote', '')}\"")
            out += ["", "</details>", ""]
    return "\n".join(out).rstrip() + "\n"
