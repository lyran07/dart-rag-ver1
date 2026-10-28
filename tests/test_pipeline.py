from datetime import datetime, timedelta

import pytest

from dart_rag.domain import Filing
from dart_rag.pipeline import Catalog, PipelineRunner, StageContext
from dart_rag.pipeline.storage import ArtifactStore

from .conftest import build_sample, make_settings


def test_hashing_embedder_regression():
    """GPT 수정본 버그: store.add 가 embed(..., is_query=False) 를 호출해 HashingEmbedder 에서 TypeError."""
    from dart_rag.embedding.providers import HashingEmbedder
    e = HashingEmbedder()
    assert e.embed_documents(["a", "b"]).dense.shape == (2, 1024)
    assert e.embed_query("q").dense.shape == (1, 1024)


def test_full_pipeline_and_resume(tmp_path):
    s = build_sample(tmp_path)
    ctx = StageContext(s)
    summary = {(r["stage"], r["status"]): r["filings"] for r in ctx.catalog.summary()}
    for st in ("fetch", "parse", "chunk", "embed", "index", "wiki"):
        assert summary[(st, "done")] == 2
    # 두 번째 실행은 전부 건너뜀
    rep = PipelineRunner(ctx, verbose=False).run(ctx.catalog.filings())
    assert not rep.processed and rep.skipped["embed"] == 2


def test_version_bump_reruns_only_downstream(tmp_path):
    s = build_sample(tmp_path)
    s.chunking.target_chars = 500   # 청킹 파라미터 변경 → chunk 이후만 재실행
    ctx = StageContext(s)
    rep = PipelineRunner(ctx, ["fetch", "parse", "chunk", "embed"], verbose=False).run(ctx.catalog.filings())
    assert rep.skipped.get("fetch") == 2 and rep.skipped.get("parse") == 2
    assert rep.processed.get("chunk") == 2 and rep.processed.get("embed") == 2


def test_deadline_stops_before_starting(tmp_path):
    s = make_settings(tmp_path)
    from dart_rag.ingestion.sources import SampleSource
    src = SampleSource()
    ctx = StageContext(s, source=src)
    ctx.catalog.upsert_filings(src.discover([], "", ""), "sample")
    rep = PipelineRunner(ctx, deadline=datetime.now() + timedelta(minutes=5), stop_margin_min=10,
                         verbose=False).run(ctx.catalog.filings())
    assert rep.stopped_early and not rep.processed


def test_failure_is_recorded_and_isolated(tmp_path):
    s = make_settings(tmp_path)
    ctx = StageContext(s, source=None)          # source 없음 → fetch 실패
    ctx.catalog.upsert_filings([Filing("X1", "0001", "테스트", "사업보고서 (2023.12)", "20240301")], "t")
    rep = PipelineRunner(ctx, ["fetch", "parse"], verbose=False).run(ctx.catalog.filings())
    assert rep.failed and ctx.catalog.failures()[0][1] == "fetch"


def test_embedding_cache_reuses_vectors(tmp_path):
    s = build_sample(tmp_path)
    ctx = StageContext(s)
    f = ctx.catalog.filings()[0]
    from dart_rag.pipeline.stages import EmbedStage
    assert EmbedStage(ctx).run(f) == 0          # 내용이 같으면 새로 임베딩한 청크 0개


def test_latest_flag_for_amendment(tmp_path):
    cat = Catalog(tmp_path / "c.db")
    cat.upsert_filings([Filing("A", "01", "가", "사업보고서 (2023.12)", "20240301"),
                        Filing("B", "01", "가", "[기재정정]사업보고서 (2023.12)", "20240501")], "t")
    latest = {f.rcept_no: f.is_latest for f in cat.filings()}
    assert latest == {"A": False, "B": True}


def test_embeddings_roundtrip_with_sparse(tmp_path):
    import numpy as np
    from dart_rag.domain import EmbeddingBatch
    st = ArtifactStore(tmp_path)
    f = Filing("R", "C", "n", "사업보고서 (2023.12)", "20240101")
    b = EmbeddingBatch(np.eye(2, 4, dtype=np.float32), [{1: 0.5, 7: 0.25}, {}])
    st.save_embeddings(f, "m", ["h1", "h2"], b)
    got = st.load_embeddings(f, "m")
    assert got["h1"][1] == {1: 0.5, 7: 0.25} and got["h2"][1] == {}
    assert np.allclose(got["h1"][0], [1, 0, 0, 0])


@pytest.mark.parametrize("backend", ["memory", "qdrant"])
def test_index_backends_agree(tmp_path, backend):
    if backend == "qdrant":
        pytest.importorskip("qdrant_client")      # requirements-optional.txt
    from dart_rag.service import RAGService
    s = build_sample(tmp_path, backend)
    r = RAGService(s).retrieve("2023년 보통주 1주당 배당금은?", k=3)
    assert "배당" in r.hits[0].chunk.meta["section_path"]
