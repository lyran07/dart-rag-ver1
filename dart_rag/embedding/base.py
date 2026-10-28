"""임베더 인터페이스.

모든 임베더는 문서용/질문용 두 메서드를 분리해 제공한다 (e5 처럼 접두어가 다른 모델 대응).
`key` 는 인덱스·아티팩트 경로에 들어가는 식별자로, **모델이 같으면 장치(GPU/CPU)가 달라도 같다.**
→ GPU 로 만든 문서 벡터와 CPU 로 만든 질문 벡터를 섞어 써도 된다.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

from ..domain import EmbeddingBatch


DEFAULT_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-m3-onnx": "gpahal/bge-m3-onnx-int8",
    "e5": "intfloat/multilingual-e5-small",
    "openai": "text-embedding-3-small",
    "hashing": "hashing-ngram-1024",
}


def model_key(model_name: str) -> str:
    """'BAAI/bge-m3' → 'bge-m3'. 인덱스 컴렉션·임베딩 파일 경로에 쓰이는 식별자."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name.split("/")[-1]).lower()


def configured_key(provider: str, model: str | None) -> str:
    """모델을 로드하지 않고 설정만으로 key 계산 (index 단계는 임베더가 필요 없음)."""
    p = provider.lower()
    p = {"bgem3": "bge-m3", "m3": "bge-m3", "hash": "hashing", "st": "e5",
         "m3-onnx": "bge-m3-onnx", "onnx": "bge-m3-onnx"}.get(p, p)
    return model_key(model or DEFAULT_MODELS.get(p, provider))


class Embedder(ABC):
    model_name: str = "base"
    dim: int = 0
    supports_sparse: bool = False
    device: str = "cpu"

    @property
    def key(self) -> str:
        return model_key(self.model_name)

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        ...

    @abstractmethod
    def embed_query(self, text: str) -> EmbeddingBatch:
        ...

    def describe(self) -> dict:
        return {"model": self.model_name, "key": self.key, "dim": self.dim,
                "sparse": self.supports_sparse, "device": self.device}
