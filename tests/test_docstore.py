"""DocStore 두 구현(SQLite / PostgreSQL)이 같은 동작을 하는지.
PostgreSQL 테스트는 TEST_DATABASE_URL 이 있을 때만:  docker compose up -d postgres
  TEST_DATABASE_URL=postgresql://dart:dart@localhost:5432/dart pytest -q
"""
import pytest

from dart_rag.domain import Chunk
from dart_rag.index import make_docstore
from dart_rag.service import RAGService

from .conftest import build_sample, make_settings, requires_pg

BACKENDS = ["sqlite", pytest.param("postgres", marks=requires_pg)]


def _chunks(rc="R1"):
    base = {"corp_name": "가나", "corp_code": "01", "parent_id": "P", "section_path": "III > 배당"}
    return [
        Chunk(f"{rc}-t", "본문", "text", {**base}),
        Chunk(f"{rc}-kv", "배당금=361원", "table_kv", {**base, "table_index": 0, "row_start": 0, "col_group": 0}),
        Chunk(f"{rc}-tb", "| 배당금 |", "table", {**base, "table_index": 0, "row_start": 0, "col_group": 0}),
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_docstore_contract(tmp_path, backend):
    ds = make_docstore(make_settings(tmp_path, docstore=backend))
    ds.replace_filing("R1", _chunks())
    assert ds.count() == 3
    got = ds.get_many(["R1-kv", "없음"])
    assert list(got) == ["R1-kv"] and got["R1-kv"].meta["table_index"] == 0     # meta 왕복 (JSON/JSONB)
    assert [c.id for c in ds.by_parent("P", "table_kv")] == ["R1-kv"]
    assert ds.table_group("P", 0, 0, 0).id == "R1-tb"
    assert ds.corp_names() == ["가나"]
    ds.replace_filing("R1", _chunks()[:1])         # 교체 = 이전 청크 삭제 후 삽입
    assert ds.count() == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_end_to_end_with_docstore(tmp_path, backend):
    s = build_sample(tmp_path, "qdrant", backend)
    svc = RAGService(s)
    r = svc.retrieve("2023년 보통주 1주당 배당금은?", k=3)
    assert "배당" in r.hits[0].chunk.meta["section_path"]
    assert svc.docstore.describe()["backend"] == backend


def test_bad_dsn_gives_clear_error(tmp_path):
    pytest.importorskip("psycopg")
    s = make_settings(tmp_path, docstore="sqlite")
    s.docstore.backend, s.docstore.dsn = "postgres", "postgresql://u:secret@127.0.0.1:1/x?connect_timeout=1"
    with pytest.raises(RuntimeError) as e:
        make_docstore(s)
    assert "docker compose" in str(e.value) and "secret" not in str(e.value)
