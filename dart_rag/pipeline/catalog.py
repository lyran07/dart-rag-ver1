"""카탈로그 DB (SQLite): 어떤 공시가 있고, 각 단계가 어느 버전까지 끝났는지.

대여 GPU 서버가 꺼졌다 켜져도 catalog.db 만 영구 저장소에 있으면 이어서 처리한다.
단계 완료 조건 = status='done' AND version=현재 버전  → 파서/청커/모델을 바꾸면 해당 단계부터 자동 재실행.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..domain import Filing

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
  rcept_no    TEXT PRIMARY KEY,
  corp_code   TEXT NOT NULL,
  corp_name   TEXT NOT NULL,
  report_nm   TEXT NOT NULL,
  report_type TEXT,
  period      TEXT,
  rcept_dt    TEXT,
  is_latest   INTEGER DEFAULT 1,
  source      TEXT,
  added_at    REAL
);
CREATE INDEX IF NOT EXISTS ix_filings_corp ON filings(corp_code, report_type, period);

CREATE TABLE IF NOT EXISTS stage_runs (
  rcept_no   TEXT NOT NULL,
  stage      TEXT NOT NULL,
  version    TEXT NOT NULL,
  status     TEXT NOT NULL,          -- running | done | failed
  n_items    INTEGER DEFAULT 0,
  seconds    REAL DEFAULT 0,
  host       TEXT,
  error      TEXT,
  updated_at REAL,
  PRIMARY KEY (rcept_no, stage)
);
"""


class Catalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------------ filings
    def upsert_filings(self, filings: list[Filing], source: str) -> int:
        now = time.time()
        with self.db:
            self.db.executemany(
                "INSERT INTO filings (rcept_no, corp_code, corp_name, report_nm, report_type, period, rcept_dt, source, added_at) "
                "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(rcept_no) DO UPDATE SET report_nm=excluded.report_nm",
                [(f.rcept_no, f.corp_code, f.corp_name, f.report_nm, f.report_type, f.period, f.rcept_dt, source, now)
                 for f in filings])
        self.recompute_latest()
        return len(filings)

    def recompute_latest(self) -> None:
        """같은 (회사, 보고서종류, 기간) 중 가장 늦게 접수된 것만 is_latest=1 (기재정정 반영)."""
        with self.db:
            self.db.execute("""
              UPDATE filings SET is_latest = CASE WHEN rcept_no = (
                SELECT f2.rcept_no FROM filings f2
                WHERE f2.corp_code=filings.corp_code AND f2.report_type=filings.report_type AND f2.period=filings.period
                ORDER BY f2.rcept_dt DESC, f2.rcept_no DESC LIMIT 1) THEN 1 ELSE 0 END""")

    def filings(self, corps: list[str] | None = None, latest_only: bool = False, limit: int | None = None) -> list[Filing]:
        q = "SELECT rcept_no, corp_code, corp_name, report_nm, rcept_dt, is_latest FROM filings WHERE 1=1"
        args: list = []
        if corps:
            q += f" AND (corp_name IN ({','.join('?' * len(corps))}) OR corp_code IN ({','.join('?' * len(corps))}))"
            args += corps + corps
        if latest_only:
            q += " AND is_latest=1"
        q += " ORDER BY corp_code, rcept_dt DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        return [Filing(r[0], r[1], r[2], r[3], r[4], bool(r[5])) for r in self.db.execute(q, args)]

    # ------------------------------------------------------------------ stage runs
    def is_done(self, rcept_no: str, stage: str, version: str) -> bool:
        r = self.db.execute("SELECT status, version FROM stage_runs WHERE rcept_no=? AND stage=?",
                            (rcept_no, stage)).fetchone()
        return bool(r and r[0] == "done" and r[1] == version)

    def mark(self, rcept_no: str, stage: str, version: str, status: str, n_items: int = 0,
             seconds: float = 0.0, error: str | None = None, host: str | None = None) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO stage_runs VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(rcept_no, stage) DO UPDATE SET "
                "version=excluded.version, status=excluded.status, n_items=excluded.n_items, seconds=excluded.seconds, "
                "host=excluded.host, error=excluded.error, updated_at=excluded.updated_at",
                (rcept_no, stage, version, status, n_items, seconds, host, error, time.time()))

    def summary(self) -> list[dict]:
        rows = self.db.execute("""
          SELECT stage, version, status, COUNT(*), SUM(n_items), SUM(seconds)
          FROM stage_runs GROUP BY stage, version, status ORDER BY stage, version, status""").fetchall()
        return [{"stage": s, "version": v, "status": st, "filings": n, "items": it or 0, "seconds": round(sec or 0, 1),
                 "items_per_sec": round((it or 0) / sec, 1) if sec else None} for s, v, st, n, it, sec in rows]

    def counts(self) -> dict:
        n, latest, corps = self.db.execute(
            "SELECT COUNT(*), SUM(is_latest), COUNT(DISTINCT corp_code) FROM filings").fetchone()
        return {"filings": n or 0, "latest": latest or 0, "corps": corps or 0}

    def failures(self, limit: int = 20) -> list[tuple]:
        return self.db.execute("SELECT rcept_no, stage, error FROM stage_runs WHERE status='failed' "
                               "ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
