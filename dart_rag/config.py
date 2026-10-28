"""설정 계층.

우선순위: 코드 기본값 < config/settings.yaml < 환경변수(.env 포함).
다른 모듈은 os.environ 을 직접 읽지 않고 Settings 만 받는다 (LLM 키는 예외: providers.yaml 이 env 이름을 지정).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StorageCfg:
    root: str = "data"            # 영구 저장소. GPU 서버에선 2TB 스토리지 마운트 경로
    scratch: str = "data/_scratch"  # 임시 작업 공간. GPU 서버에선 300GiB 로컬 디스크

    @property
    def root_path(self) -> Path:
        return _abs(self.root)

    @property
    def scratch_path(self) -> Path:
        return _abs(self.scratch)


@dataclass
class ChunkingCfg:
    target_chars: int = 900
    max_chars: int = 1400
    overlap_paras: int = 1
    table_rows_per_chunk: int = 12
    max_cols: int = 10
    table_chars: int = 1400
    hard_cap: int = 6000


@dataclass
class EmbeddingCfg:
    provider: str = "bge-m3"        # bge-m3 | e5 | openai | hashing
    model: str | None = None        # None → provider 기본 모델
    device: str = "auto"            # auto | cuda | cpu | mps
    batch_size: int | None = None   # None → GPU 64 / CPU 16
    doc_max_len: int = 1024
    query_max_len: int = 128
    embed_kinds: list[str] = field(default_factory=lambda: ["text", "table_kv"])
    use_sparse: bool = True         # bge-m3 lexical weights 저장 여부
    # 섹션 경로에 이 문자열이 있으면 임베딩하지 않음 (DocStore 에는 저장). 예: ["주석"] → 재무제표 주석 표 제외
    skip_sections: list[str] = field(default_factory=list)


@dataclass
class SparseCfg:
    tokenizer: str = "regex"        # regex | kiwi
    k1: float = 1.2
    b: float = 0.75
    avgdl: float = 200.0            # 전역 평균 문서 길이(토큰). 재계산하지 않고 상수로 둔다


@dataclass
class DocStoreCfg:
    backend: str = "postgres"     # postgres | sqlite
    dsn: str | None = None          # None → 환경변수 DATABASE_URL → docker-compose 기본값
    schema: str = "public"
    default_dsn: str = "postgresql://dart:dart@localhost:5432/dart"

    def resolved_dsn(self) -> str:
        return self.dsn or os.environ.get("DATABASE_URL") or self.default_dsn


@dataclass
class IndexCfg:
    backend: str = "memory"         # memory | qdrant
    qdrant_url: str | None = None   # None → 로컬 모드 (storage.root/index/qdrant)
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    collection_prefix: str = "dart_chunks"
    quantization: str = "int8"      # int8 | none
    on_disk: bool = True
    upsert_batch: int = 256


@dataclass
class RetrievalCfg:
    candidates: int = 40
    top_k: int = 8
    latest_only: bool = True
    channels: list[str] = field(default_factory=lambda: ["dense", "m3", "bm25"])
    prefetch_limit: int = 50
    boosts: dict[str, float] = field(default_factory=lambda: {
        "section_hint": 0.5, "title_overlap": 1.5, "table_kv": 0.1,
        "short_penalty": 0.5, "footnote_penalty": 0.2,
    })
    boosts_after_rerank: dict[str, float] = field(default_factory=lambda: {
        "section_hint": 0.05, "title_overlap": 0.1, "footnote_penalty": 0.05,
    })
    sibling_kv_parents: int = 3
    sibling_kv_per_parent: int = 2
    table_for_kv: bool = True
    section_text_chunks: int = 0
    multi_facet: bool = True        # 여러 연도·회사 질문을 나눠서 검색 + 의도(전망·원인·추이) 힌트
    narrative_text_chunks: int = 2  # 전망·원인 질문이면 같은 섹션 본문 N개 더 붙임


@dataclass
class RerankCfg:
    mode: str = "auto"              # auto | on | off   (auto: GPU 면 켬, CPU 면 cpu_enabled 따름)
    model: str = "BAAI/bge-reranker-v2-m3"
    gpu_candidates: int = 40
    cpu_candidates: int = 15
    cpu_enabled: bool = False
    max_length: int = 1024


@dataclass
class GenerationCfg:
    per_chunk_chars: int = 1800
    max_ctx_chars: int = 9000
    providers_file: str = "config/providers.yaml"


@dataclass
class Settings:
    storage: StorageCfg = field(default_factory=StorageCfg)
    chunking: ChunkingCfg = field(default_factory=ChunkingCfg)
    embedding: EmbeddingCfg = field(default_factory=EmbeddingCfg)
    sparse: SparseCfg = field(default_factory=SparseCfg)
    docstore: DocStoreCfg = field(default_factory=DocStoreCfg)
    index: IndexCfg = field(default_factory=IndexCfg)
    retrieval: RetrievalCfg = field(default_factory=RetrievalCfg)
    rerank: RerankCfg = field(default_factory=RerankCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)

    # 환경변수 → 설정 경로
    ENV_OVERRIDES = {
        "DART_DATA_ROOT": "storage.root",
        "DART_SCRATCH": "storage.scratch",
        "EMBED_PROVIDER": "embedding.provider",
        "EMBED_MODEL": "embedding.model",
        "EMBED_DEVICE": "embedding.device",
        "EMBED_BATCH_SIZE": "embedding.batch_size",
        "INDEX_BACKEND": "index.backend",
        "DOCSTORE_BACKEND": "docstore.backend",
        "DOCSTORE_SCHEMA": "docstore.schema",
        "QDRANT_URL": "index.qdrant_url",
        "RERANK_MODE": "rerank.mode",
        "SPARSE_TOKENIZER": "sparse.tokenizer",
        "PROVIDERS_FILE": "generation.providers_file",
    }

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, overrides: dict[str, Any] | None = None) -> "Settings":
        load_dotenv()
        data: dict[str, Any] = {}
        path = Path(path) if path else Path(os.environ.get("DART_SETTINGS", PROJECT_ROOT / "config" / "settings.yaml"))
        if path.exists():
            import yaml
            data = yaml.safe_load(path.read_text("utf-8")) or {}
        s = _from_dict(cls, data)
        for env, dotted in cls.ENV_OVERRIDES.items():
            if os.environ.get(env):
                s.set(dotted, os.environ[env])
        for dotted, v in (overrides or {}).items():
            s.set(dotted, v)
        return s

    def set(self, dotted: str, value: Any) -> None:
        sec_name, key = dotted.split(".", 1)
        sec = getattr(self, sec_name)
        cur = getattr(sec, key)
        if isinstance(cur, bool):
            value = value if isinstance(value, bool) else str(value).lower() in {"1", "true", "yes", "on"}
        elif isinstance(cur, int) and not isinstance(value, int):
            value = int(value)
        elif isinstance(cur, float) and not isinstance(value, float):
            value = float(value)
        elif key == "batch_size" and value is not None:
            value = int(value)
        setattr(sec, key, value)


def _from_dict(tp, data: dict):
    kwargs = {}
    for f in fields(tp):
        if f.name not in data:
            continue
        v = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(v, dict):
            kwargs[f.name] = _from_dict(type(default), v)
        elif isinstance(default, dict) and isinstance(v, dict):
            kwargs[f.name] = {**default, **v}
        else:
            kwargs[f.name] = v
    return tp(**kwargs)


def _abs(p: str) -> Path:
    path = Path(os.path.expanduser(p))
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_dotenv() -> None:
    try:
        from dotenv import load_dotenv as _ld
        _ld(PROJECT_ROOT / ".env")
    except ImportError:
        pass
