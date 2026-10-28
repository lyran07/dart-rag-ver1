"""임베딩 계층. make_embedder(settings) 로만 생성한다."""
from __future__ import annotations

from ..config import EmbeddingCfg
from .base import Embedder
from .device import gpu_info, resolve_device

__all__ = ["Embedder", "make_embedder", "resolve_device", "gpu_info"]


def make_embedder(cfg: EmbeddingCfg) -> Embedder:
    try:
        return _make(cfg)
    except ImportError as e:
        # 조용히 hashing 으로 폴백하지 않는다: 다른 모델로 만든 인덱스를 엉뚱하게 쓰게 되는 걸 막기 위함
        raise RuntimeError(
            f"임베딩 provider '{cfg.provider}' 에 필요한 패키지가 없습니다 ({e.name}). "
            "requirements-ml.txt 를 설치하거나 EMBED_PROVIDER=hashing 으로 테스트하세요.") from e


def _make(cfg: EmbeddingCfg) -> Embedder:
    from . import providers as p
    prov = cfg.provider.lower()
    if prov in {"bge-m3", "bgem3", "m3"}:
        return p.BGEM3Embedder(cfg.model, cfg.device, cfg.batch_size, cfg.doc_max_len, cfg.query_max_len,
                               use_sparse=cfg.use_sparse)
    if prov in {"bge-m3-onnx", "m3-onnx", "onnx"}:
        return p.BGEM3OnnxEmbedder(cfg.model, cfg.batch_size, cfg.doc_max_len, cfg.query_max_len,
                                   use_sparse=cfg.use_sparse)
    if prov in {"e5", "st", "sentence-transformers"}:
        return p.SentenceTransformerEmbedder(cfg.model, cfg.device, cfg.batch_size, max_len=min(cfg.doc_max_len, 512))
    if prov == "openai":
        return p.OpenAICompatEmbedder(cfg.model)
    if prov in {"hash", "hashing"}:
        return p.HashingEmbedder()
    raise ValueError(f"unknown embedding provider: {cfg.provider}")
