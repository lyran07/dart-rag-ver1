"""BM25 를 sparse 벡터로 표현 (Qdrant sparse vector + IDF modifier 방식).

문서 벡터 가중치 = TF 포화항  tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))
IDF 는 인덱스가 계산한다 (Qdrant: Modifier.IDF / 메모리 인덱스: 직접 계산).
→ 기존 rank_bm25 처럼 전 코퍼스를 메모리에 올리고 매번 다시 만들 필요가 없고, 문서를 증분 추가할 수 있다.
"""
from __future__ import annotations

import hashlib
from collections import Counter

from ..domain import SparseVec
from .tokenizer import Tokenizer


def token_id(tok: str) -> int:
    return int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(), "little") & 0x7FFFFFFF


class BM25Encoder:
    def __init__(self, tokenizer: Tokenizer, k1: float = 1.2, b: float = 0.75, avgdl: float = 200.0):
        self.tok, self.k1, self.b, self.avgdl = tokenizer, k1, b, avgdl

    @property
    def key(self) -> str:
        return f"bm25-{self.tok.name}"

    def encode_document(self, text: str) -> SparseVec:
        toks = self.tok.tokenize(text)
        if not toks:
            return {}
        dl = len(toks)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        out: SparseVec = {}
        for t, tf in Counter(toks).items():
            i = token_id(t)
            out[i] = out.get(i, 0.0) + tf * (self.k1 + 1) / (tf + norm)
        return out

    def encode_query(self, text: str) -> SparseVec:
        return {token_id(t): 1.0 for t in set(self.tok.tokenize(text))}
