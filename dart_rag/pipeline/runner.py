"""파이프라인 실행기: 이어서 하기 + 마감 시간 + 안전 종료.

- 공시 1건씩 선택된 단계를 차례로 실행 (완료된 단계는 catalog 보고 건너뜀)
- deadline − stop_margin 이 지나면 새 공시를 시작하지 않고 정리(finalize) 후 종료
- SIGTERM/SIGINT(Ctrl+C) 를 받으면 현재 공시만 마치고 종료 → 대여 종료 직전에도 산출물이 깨지지 않음
"""
from __future__ import annotations

import signal
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..domain import Filing
from .stages import DEFAULT_ORDER, STAGES, Stage, StageContext


@dataclass
class RunReport:
    processed: dict[str, int] = field(default_factory=dict)   # stage → 실행한 공시 수
    skipped: dict[str, int] = field(default_factory=dict)
    failed: list[tuple[str, str, str]] = field(default_factory=list)
    items: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    stopped_early: str | None = None

    def lines(self) -> list[str]:
        out = []
        from .stages import DEFAULT_ORDER
        for s in [x for x in DEFAULT_ORDER if x in self.processed or x in self.skipped]:
            sec = self.seconds.get(s, 0.0)
            ips = f", {self.items.get(s, 0) / sec:.1f} items/s" if sec > 0 else ""
            out.append(f"  {s:6s} 실행 {self.processed.get(s, 0)} · 건너뜀 {self.skipped.get(s, 0)} · "
                       f"items {self.items.get(s, 0)} · {sec:.1f}s{ips}")
        if self.failed:
            out.append(f"  실패 {len(self.failed)}건 (status 명령으로 확인)")
        if self.stopped_early:
            out.append(f"  ⏹ 조기 종료: {self.stopped_early}")
        return out


def parse_deadline(until: str | None, max_minutes: float | None) -> datetime | None:
    """'06:30' → 다음 06:30 / '2026-10-01T06:30' → 그 시각 / max_minutes → 지금부터 N분."""
    now = datetime.now()
    cands = []
    if max_minutes:
        cands.append(now + timedelta(minutes=max_minutes))
    if until:
        if "T" in until or "-" in until:
            cands.append(datetime.fromisoformat(until))
        else:
            h, m = map(int, until.split(":"))
            d = now.replace(hour=h, minute=m, second=0, microsecond=0)
            cands.append(d if d > now else d + timedelta(days=1))
    return min(cands) if cands else None


class PipelineRunner:
    def __init__(self, ctx: StageContext, stages: list[str] | None = None, deadline: datetime | None = None,
                 stop_margin_min: float = 20, force: bool = False, verbose: bool = True):
        names = stages or DEFAULT_ORDER
        unknown = [n for n in names if n not in STAGES]
        if unknown:
            raise ValueError(f"unknown stage: {unknown} (가능: {list(STAGES)})")
        self.ctx = ctx
        self.stages: list[Stage] = [STAGES[n](ctx) for n in DEFAULT_ORDER if n in names]
        self.deadline, self.margin = deadline, timedelta(minutes=stop_margin_min)
        self.force, self.verbose = force, verbose
        self._stop: str | None = None

    def _on_signal(self, signum, _frame):
        self._stop = f"signal {signal.Signals(signum).name}"
        print(f"\n[runner] {self._stop} 수신 → 현재 공시만 마치고 종료합니다.")

    def _time_up(self) -> bool:
        return bool(self.deadline and datetime.now() + self.margin >= self.deadline)

    def run(self, filings: list[Filing]) -> RunReport:
        rep = RunReport()
        old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            for s in old:
                signal.signal(s, self._on_signal)
        except ValueError:  # 메인 스레드가 아닐 때 (API 서버 등)
            old = {}
        cat, versions = self.ctx.catalog, {st.name: st.version() for st in self.stages}
        try:
            for i, f in enumerate(filings, 1):
                if self._stop:
                    rep.stopped_early = self._stop
                    break
                if self._time_up():
                    rep.stopped_early = f"마감 {self.deadline:%m-%d %H:%M} − {self.margin} 도달 ({i - 1}/{len(filings)}건 처리)"
                    break
                for st in self.stages:
                    ver = versions[st.name]
                    if not self.force and cat.is_done(f.rcept_no, st.name, ver):
                        rep.skipped[st.name] = rep.skipped.get(st.name, 0) + 1
                        continue
                    t = time.perf_counter()
                    cat.mark(f.rcept_no, st.name, ver, "running", host=self.ctx.host)
                    try:
                        n = st.run(f)
                    except Exception as e:  # noqa: BLE001  한 건 실패가 전체를 멈추지 않게
                        cat.mark(f.rcept_no, st.name, ver, "failed", error=f"{type(e).__name__}: {e}"[:500],
                                 host=self.ctx.host)
                        rep.failed.append((f.rcept_no, st.name, str(e)))
                        if self.verbose:
                            print(f"  ✘ {f.corp_name} {f.report_nm} [{st.name}] {e}")
                            traceback.print_exc(limit=2)
                        break  # 이 공시의 뒤 단계는 건너뜀
                    dt = time.perf_counter() - t
                    cat.mark(f.rcept_no, st.name, ver, "done", n_items=n, seconds=dt, host=self.ctx.host)
                    rep.processed[st.name] = rep.processed.get(st.name, 0) + 1
                    rep.items[st.name] = rep.items.get(st.name, 0) + n
                    rep.seconds[st.name] = rep.seconds.get(st.name, 0.0) + dt
                if self.verbose and i % 10 == 0:
                    print(f"[runner] {i}/{len(filings)}")
        finally:
            for st in self.stages:
                try:
                    st.finalize()
                except Exception as e:  # noqa: BLE001
                    print(f"[runner] finalize({st.name}) 실패: {e}")
            for s, h in old.items():
                signal.signal(s, h)
        return rep
