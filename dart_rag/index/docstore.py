"""DocStore: 청크 본문·메타 저장소 (관계형 DB).

벡터 인덱스와 분리한 이유
- Qdrant payload 에 본문을 넣으면 RAM/디스크가 커지고, 모델을 바꿀 때마다 본문까지 복제된다.
- small-to-big 확장(같은 섹션의 표 KV, 원본 표, 섹션 본문)은 "parent_id 로 조회" 라서 RDB 가 적합.
- `table` 마크다운 청크처럼 임베딩하지 않는 청크도 여기엔 저장한다.

구현
- PostgresDocStore (서빙 기본): 여러 사용자·API 워커 동시 접속, 서비스 중 증분 적재, meta 는 JSONB
- SQLiteDocStore (노트북 단독·테스트): 설치 없이 파일 하나

DocStore 는 언제든 아티팩트(chunks/)에서 다시 만들 수 있다:  python -m dart_rag run --stages index --force
"""
from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from ..domain import Chunk

COLUMNS = ("chunk_id", "rcept_no", "corp_code", "corp_name", "parent_id", "kind", "table_index", "row_start",
           "col_group", "section_path", "text", "meta")


def _row_values(rcept_no: str, ch: Chunk) -> tuple:
    m = ch.meta
    return (ch.id, rcept_no, m.get("corp_code"), m.get("corp_name"), m.get("parent_id"), ch.kind,
            m.get("table_index"), m.get("row_start"), m.get("col_group"), m.get("section_path"), ch.text)


class DocStore(ABC):
    backend = "base"

    @abstractmethod
    def replace_filing(self, rcept_no: str, chunks: list[Chunk]) -> None:
        """공시 1건의 청크를 통째로 교체 (한 트랜잭션)."""

    @abstractmethod
    def get_many(self, ids: list[str]) -> dict[str, Chunk]: ...

    @abstractmethod
    def by_parent(self, parent_id: str, kind: str | None = None, limit: int = 200) -> list[Chunk]: ...

    @abstractmethod
    def table_group(self, parent_id: str, table_index, row_start, col_group=0) -> Chunk | None: ...

    @abstractmethod
    def corp_names(self) -> list[str]: ...

    @abstractmethod
    def count(self) -> int: ...

    def describe(self) -> dict:
        return {"backend": self.backend, "chunks": self.count()}


# ============================================================== SQLite
SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     TEXT PRIMARY KEY,
  rcept_no     TEXT NOT NULL,
  corp_code    TEXT,
  corp_name    TEXT,
  parent_id    TEXT,
  kind         TEXT,
  table_index  INTEGER,
  row_start    INTEGER,
  col_group    INTEGER,
  section_path TEXT,
  text         TEXT NOT NULL,
  meta         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id, kind);
CREATE INDEX IF NOT EXISTS ix_chunks_rcept  ON chunks(rcept_no);
CREATE INDEX IF NOT EXISTS ix_chunks_corp   ON chunks(corp_name);
"""


class SQLiteDocStore(DocStore):
    backend = "sqlite"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conn().executescript(SQLITE_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, check_same_thread=False)
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return c

    @staticmethod
    def _chunk(row) -> Chunk:
        cid, text, kind, meta = row
        return Chunk(cid, text, kind, json.loads(meta))

    def replace_filing(self, rcept_no, chunks):
        c = self._conn()
        with c:
            c.execute("DELETE FROM chunks WHERE rcept_no=?", (rcept_no,))
            c.executemany(f"INSERT OR REPLACE INTO chunks VALUES ({','.join('?' * len(COLUMNS))})",
                          [(*_row_values(rcept_no, ch), json.dumps(ch.meta, ensure_ascii=False)) for ch in chunks])

    def get_many(self, ids):
        out: dict[str, Chunk] = {}
        for s in range(0, len(ids), 500):
            part = ids[s:s + 500]
            q = f"SELECT chunk_id, text, kind, meta FROM chunks WHERE chunk_id IN ({','.join('?' * len(part))})"
            for row in self._conn().execute(q, part):
                out[row[0]] = self._chunk(row)
        return out

    def by_parent(self, parent_id, kind=None, limit=200):
        q, args = "SELECT chunk_id, text, kind, meta FROM chunks WHERE parent_id=?", [parent_id]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        q += " ORDER BY table_index, col_group, row_start, chunk_id LIMIT ?"
        return [self._chunk(r) for r in self._conn().execute(q, [*args, limit])]

    def table_group(self, parent_id, table_index, row_start, col_group=0):
        row = self._conn().execute(
            "SELECT chunk_id, text, kind, meta FROM chunks WHERE parent_id=? AND kind='table' "
            "AND table_index=? AND row_start=? AND IFNULL(col_group,0)=? LIMIT 1",
            (parent_id, table_index, row_start, col_group or 0)).fetchone()
        return self._chunk(row) if row else None

    def corp_names(self):
        return [r[0] for r in self._conn().execute("SELECT DISTINCT corp_name FROM chunks WHERE corp_name IS NOT NULL")]

    def count(self):
        return self._conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]


# ============================================================== PostgreSQL
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     TEXT PRIMARY KEY,
  rcept_no     TEXT NOT NULL,
  corp_code    TEXT,
  corp_name    TEXT,
  parent_id    TEXT,
  kind         TEXT,
  table_index  INTEGER,
  row_start    INTEGER,
  col_group    INTEGER,
  section_path TEXT,
  text         TEXT NOT NULL,
  meta         JSONB NOT NULL,
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id, kind);
CREATE INDEX IF NOT EXISTS ix_chunks_rcept  ON chunks(rcept_no);
CREATE INDEX IF NOT EXISTS ix_chunks_corp   ON chunks(corp_name);
CREATE INDEX IF NOT EXISTS ix_chunks_meta   ON chunks USING GIN (meta jsonb_path_ops);
"""


class PostgresDocStore(DocStore):
    """psycopg 3. 스레드별 연결 (FastAPI 동기 엔드포인트는 스레드풀에서 실행됨)."""
    backend = "postgres"

    def __init__(self, dsn: str, schema: str = "public"):
        try:
            import psycopg  # noqa: F401
        except ImportError as e:
            raise RuntimeError("PostgreSQL 사용에는 psycopg 가 필요합니다: pip install 'psycopg[binary]'") from e
        self.dsn, self.schema = dsn, schema
        self._local = threading.local()
        with self._conn().transaction():
            cur = self._conn().cursor()
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
            cur.execute(f'SET search_path TO "{schema}"')
            cur.execute(PG_SCHEMA)

    def _conn(self):
        import psycopg
        c = getattr(self._local, "conn", None)
        if c is None or c.closed:
            try:
                c = psycopg.connect(self.dsn, autocommit=True, options=f"-c search_path={self.schema}")
            except psycopg.OperationalError as e:
                raise RuntimeError(
                    f"PostgreSQL 연결 실패 ({_mask(self.dsn)}). `docker compose up -d` 로 DB 를 띄웠는지, "
                    "DATABASE_URL 이 맞는지 확인하세요. (DB 없이 쓰려면 DOCSTORE_BACKEND=sqlite)") from e
            self._local.conn = c
        return c

    @staticmethod
    def _chunk(row) -> Chunk:
        cid, text, kind, meta = row
        return Chunk(cid, text, kind, meta if isinstance(meta, dict) else json.loads(meta))

    def replace_filing(self, rcept_no, chunks):
        from psycopg.types.json import Jsonb
        c = self._conn()
        with c.transaction():
            cur = c.cursor()
            cur.execute("DELETE FROM chunks WHERE rcept_no=%s", (rcept_no,))
            cur.executemany(
                f"INSERT INTO chunks ({','.join(COLUMNS)}) VALUES ({','.join(['%s'] * len(COLUMNS))}) "
                "ON CONFLICT (chunk_id) DO UPDATE SET rcept_no=EXCLUDED.rcept_no, text=EXCLUDED.text, "
                "meta=EXCLUDED.meta, kind=EXCLUDED.kind, parent_id=EXCLUDED.parent_id, updated_at=now()",
                [(*_row_values(rcept_no, ch), Jsonb(ch.meta)) for ch in chunks])

    def get_many(self, ids):
        if not ids:
            return {}
        rows = self._conn().execute(
            "SELECT chunk_id, text, kind, meta FROM chunks WHERE chunk_id = ANY(%s)", (list(ids),)).fetchall()
        return {r[0]: self._chunk(r) for r in rows}

    def by_parent(self, parent_id, kind=None, limit=200):
        q, args = "SELECT chunk_id, text, kind, meta FROM chunks WHERE parent_id=%s", [parent_id]
        if kind:
            q += " AND kind=%s"
            args.append(kind)
        q += " ORDER BY table_index NULLS FIRST, col_group NULLS FIRST, row_start NULLS FIRST, chunk_id LIMIT %s"
        return [self._chunk(r) for r in self._conn().execute(q, [*args, limit]).fetchall()]

    def table_group(self, parent_id, table_index, row_start, col_group=0):
        row = self._conn().execute(
            "SELECT chunk_id, text, kind, meta FROM chunks WHERE parent_id=%s AND kind='table' "
            "AND table_index=%s AND row_start=%s AND COALESCE(col_group,0)=%s LIMIT 1",
            (parent_id, table_index, row_start, col_group or 0)).fetchone()
        return self._chunk(row) if row else None

    def corp_names(self):
        return [r[0] for r in self._conn().execute(
            "SELECT DISTINCT corp_name FROM chunks WHERE corp_name IS NOT NULL").fetchall()]

    def count(self):
        return self._conn().execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def describe(self):
        return {"backend": self.backend, "dsn": _mask(self.dsn), "schema": self.schema, "chunks": self.count()}


def _mask(dsn: str) -> str:
    """로그에 비밀번호가 찍히지 않게."""
    import re
    return re.sub(r"(://[^:/@]+):[^@]*@", r"\1:***@", dsn)
