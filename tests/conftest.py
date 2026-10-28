import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dart_rag.config import Settings  # noqa: E402


@pytest.fixture(autouse=True)
def _no_llm_keys(monkeypatch):
    for k in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "UPSTAGE_API_KEY", "VLLM_BASE_URL", "USE_OLLAMA",
              "LLM_ORDER", "EMBED_PROVIDER", "INDEX_BACKEND", "DART_DATA_ROOT", "EMBED_DEVICE",
              "DOCSTORE_BACKEND", "DATABASE_URL", "QDRANT_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("dart_rag.config.load_dotenv", lambda: None)
    # 기존 테스트는 rag 모드로 (에이전트는 tests/test_agent.py 에서 가짜 LLM 으로 따로 검증)
    for k in ("AGENT_BACKEND", "AGENT_MODEL", "AGENT_REASONING"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ASK_MODE", "rag")
    monkeypatch.setenv("WARMUP", "0")
    monkeypatch.delenv("ROUTER", raising=False)


requires_pg = pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"),
                                 reason="TEST_DATABASE_URL 미설정 (예: postgresql://dart:dart@localhost:5432/dart)")


def make_settings(tmp_path, backend="memory", docstore="sqlite") -> Settings:
    ov = {}
    if docstore == "postgres":   # 테스트마다 새 스키마 → 서로 간섭 없음
        import uuid
        ov = {"docstore.dsn": os.environ["TEST_DATABASE_URL"], "docstore.schema": f"t_{uuid.uuid4().hex[:10]}"}
    return Settings.load(ROOT / "config" / "settings.yaml", {**ov,
        "docstore.backend": docstore,
        "storage.root": str(tmp_path / "data"),
        "embedding.provider": "hashing",
        "index.backend": backend,
        "index.quantization": "none",
        "rerank.mode": "off",
    })


def build_sample(tmp_path, backend="memory", docstore="sqlite") -> Settings:
    from dart_rag.ingestion.sources import SampleSource
    from dart_rag.pipeline import PipelineRunner, StageContext
    s = make_settings(tmp_path, backend, docstore)
    src = SampleSource()
    ctx = StageContext(s, source=src)
    ctx.catalog.upsert_filings(src.discover([], "", ""), "sample")
    PipelineRunner(ctx, verbose=False).run(ctx.catalog.filings())
    return s
