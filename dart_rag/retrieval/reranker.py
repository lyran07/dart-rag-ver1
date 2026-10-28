"""크로스인코더 리랭커.

mode=auto: GPU 가 있으면 켬(후보 40), CPU 면 cpu_enabled 설정을 따른다(켜면 후보 15, 질문당 ~1초대).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import RerankCfg
from ..embedding.device import resolve_device


class Reranker(ABC):
    enabled = True
    candidates = 40
    name = "base"

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class NoopReranker(Reranker):
    enabled = False
    name = "none"

    def __init__(self, candidates: int = 40):
        self.candidates = candidates

    def score(self, query, passages):
        return [0.0] * len(passages)


class BGEReranker(Reranker):
    def __init__(self, model: str, device: str, candidates: int, max_length: int = 1024):
        self.name, self.device, self.candidates, self.max_length = model, device, candidates, max_length
        try:
            from FlagEmbedding import FlagReranker
            self._flag = FlagReranker(model, use_fp16=device == "cuda", devices=device) \
                if _accepts(FlagReranker, "devices") else FlagReranker(model, use_fp16=device == "cuda")
            self._ce = None
        except ImportError:
            from sentence_transformers import CrossEncoder
            self._flag = None
            self._ce = CrossEncoder(model, device=device, max_length=max_length)

    def score(self, query, passages):
        pairs = [(query, p) for p in passages]
        if not pairs:
            return []
        if self._flag is not None:
            s = self._flag.compute_score(pairs, normalize=True, max_length=self.max_length)
            return [float(x) for x in (s if isinstance(s, list) else [s])]
        import numpy as np
        return [float(x) for x in 1 / (1 + np.exp(-np.asarray(self._ce.predict(pairs))))]


def _accepts(cls, arg: str) -> bool:
    import inspect
    try:
        return arg in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def make_reranker(cfg: RerankCfg, fallback_candidates: int = 40) -> Reranker:
    if cfg.mode == "off":
        return NoopReranker(fallback_candidates)
    device = resolve_device("auto")
    on_gpu = device == "cuda"
    if cfg.mode == "auto" and not on_gpu and not cfg.cpu_enabled:
        return NoopReranker(fallback_candidates)
    try:
        return BGEReranker(cfg.model, device, cfg.gpu_candidates if on_gpu else cfg.cpu_candidates, cfg.max_length)
    except ImportError:
        print("[rerank] FlagEmbedding/sentence-transformers 미설치 → 리랭커 비활성")
        return NoopReranker(fallback_candidates)
