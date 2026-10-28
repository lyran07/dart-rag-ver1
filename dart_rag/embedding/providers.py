"""임베더 구현체.

| provider | 모델                               | dense | sparse | 장치        | 용도 |
|----------|------------------------------------|-------|--------|-------------|------|
| bge-m3   | BAAI/bge-m3                        | 1024  | ✅     | GPU/CPU     | 기본 (한국어, 8192 토큰) |
| e5       | intfloat/multilingual-e5-small     | 384   | ❌     | GPU/CPU     | 가벼운 대안 (512 토큰 제한) |
| openai   | text-embedding-3-small             | 1536  | ❌     | API         | GPU 없이 대량 처리 |
| hashing  | char n-gram 해싱                   | 1024  | ❌     | CPU         | 오프라인 테스트 전용 |
"""
from __future__ import annotations

import hashlib
import os
import time

import numpy as np

from ..domain import EmbeddingBatch
from .base import Embedder
from .device import default_batch_size, resolve_device


def _normalize(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)


def _length_sorted_batches(texts: list[str], batch_size: int):
    """길이순 정렬 후 배치 → padding 낭비 감소. (원래 순서 인덱스와 함께 반환)"""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        yield idx, [texts[i] for i in idx]


# ---------------------------------------------------------------- hashing
class HashingEmbedder(Embedder):
    """외부 의존성 없는 폴백 (char n-gram 해싱). 의미 벡터가 아니므로 테스트용."""
    model_name = "hashing-ngram-1024"
    dim = 1024

    def __init__(self, tokenizer=None):
        from ..index.tokenizer import RegexTokenizer
        self.tok = tokenizer or RegexTokenizer()

    def _embed(self, texts: list[str]) -> np.ndarray:
        M = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self.tok.tokenize(t):
                h = int(hashlib.blake2b(tok.encode(), digest_size=4).hexdigest(), 16) % self.dim
                M[i, h] += 1.0
        return _normalize(np.log1p(M))

    def embed_documents(self, texts):
        return EmbeddingBatch(self._embed(texts))

    def embed_query(self, text):
        return EmbeddingBatch(self._embed([text]))


# ---------------------------------------------------------------- sentence-transformers (e5 등)
class SentenceTransformerEmbedder(Embedder):
    DEFAULT = "intfloat/multilingual-e5-small"

    def __init__(self, model: str | None = None, device: str = "auto", batch_size: int | None = None,
                 max_len: int = 512, query_prefix: str | None = None, doc_prefix: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model or self.DEFAULT
        self.device = resolve_device(device)
        self.m = SentenceTransformer(self.model_name, device=self.device)
        self.m.max_seq_length = min(max_len, self.m.max_seq_length or max_len)
        if self.device == "cuda":
            self.m.half()
        self.dim = self.m.get_sentence_embedding_dimension()
        self.batch_size = default_batch_size(self.device, batch_size)
        is_e5 = "e5" in self.model_name.lower()
        self.qp = query_prefix if query_prefix is not None else ("query: " if is_e5 else "")
        self.dp = doc_prefix if doc_prefix is not None else ("passage: " if is_e5 else "")

    def _encode(self, texts: list[str]) -> np.ndarray:
        return _normalize(self.m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                        show_progress_bar=False))

    def embed_documents(self, texts):
        return EmbeddingBatch(self._encode([self.dp + t for t in texts]))

    def embed_query(self, text):
        return EmbeddingBatch(self._encode([self.qp + text]))


# ---------------------------------------------------------------- bge-m3 (dense + sparse)
class BGEM3Embedder(Embedder):
    """BAAI/bge-m3. 한 번의 forward 로 dense(1024) + sparse(lexical weights) 를 얻는다.

    - GPU: fp16, batch 64 (4090 기준). CPU: fp32, batch 16 — 질문 1개 인코딩은 CPU 로도 충분.
    - FlagEmbedding 이 없으면 sentence-transformers 로 dense 만 사용 (sparse 채널 비활성).
    - ColBERT 벡터는 저장 비용(토큰당 1024차원)이 커서 사용하지 않는다.
    """
    DEFAULT = "BAAI/bge-m3"
    dim = 1024

    def __init__(self, model: str | None = None, device: str = "auto", batch_size: int | None = None,
                 doc_max_len: int = 1024, query_max_len: int = 128, use_sparse: bool = True,
                 min_sparse_weight: float = 0.01):
        self.model_name = model or self.DEFAULT
        self.device = resolve_device(device)
        self.batch_size = default_batch_size(self.device, batch_size)
        self.doc_max_len, self.query_max_len = doc_max_len, query_max_len
        self.min_w = min_sparse_weight
        self._flag = None
        self._st = None
        try:
            from FlagEmbedding import BGEM3FlagModel
            try:   # FlagEmbedding >= 1.3
                self._flag = BGEM3FlagModel(self.model_name, use_fp16=self.device == "cuda", devices=self.device)
            except TypeError:   # 구버전
                self._flag = BGEM3FlagModel(self.model_name, use_fp16=self.device == "cuda", device=self.device)
            self.supports_sparse = use_sparse
        except ImportError:
            from sentence_transformers import SentenceTransformer
            print("[embed] FlagEmbedding 미설치 → bge-m3 dense 만 사용 (pip install FlagEmbedding 권장)")
            self._st = SentenceTransformer(self.model_name, device=self.device)
            if self.device == "cuda":
                self._st.half()
            self.supports_sparse = False

    def _encode(self, texts: list[str], max_len: int) -> EmbeddingBatch:
        if self._st is not None:
            self._st.max_seq_length = max_len
            d = self._st.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                show_progress_bar=False)
            return EmbeddingBatch(_normalize(d))
        dense = np.zeros((len(texts), self.dim), dtype=np.float32)
        sparse: list[dict[int, float]] = [{} for _ in texts]
        for idx, batch in _length_sorted_batches(texts, self.batch_size):
            o = self._flag.encode(batch, batch_size=len(batch), max_length=max_len,
                                  return_dense=True, return_sparse=self.supports_sparse,
                                  return_colbert_vecs=False)
            dense[idx] = np.asarray(o["dense_vecs"], dtype=np.float32)
            if self.supports_sparse:
                for j, lw in zip(idx, o["lexical_weights"]):
                    sparse[j] = {int(k): float(v) for k, v in lw.items() if float(v) >= self.min_w}
        return EmbeddingBatch(_normalize(dense), sparse if self.supports_sparse else None)

    def embed_documents(self, texts):
        return self._encode(texts, self.doc_max_len)

    def embed_query(self, text):
        return self._encode([text], self.query_max_len)


# ---------------------------------------------------------------- bge-m3 ONNX (torch 없이 CPU)
class BGEM3OnnxEmbedder(Embedder):
    """bge-m3 를 onnxruntime 으로 실행 (torch 불필요 → 노트북·저사양 CPU 서버용).

    기본 모델 gpahal/bge-m3-onnx-int8: BAAI/bge-m3 를 ONNX 로 변환 + int8 동적 양자화 (570MB).
    출력 dense_vecs(정규화) + sparse_vecs(토큰별 가중치) → FlagEmbedding 과 같은 방식으로 lexical weights 계산.
    주의: int8 은 fp16/fp32 원본과 벡터가 미세하게 달라 **별도 인덱스 key**(bge-m3-onnx-int8)를 쓴다.
          GPU 에서 원본 bge-m3 로 만든 인덱스를 CPU 에서 조회하려면 fp32 ONNX(BAAI/bge-m3 의 onnx/)를 쓸 것.
    """
    DEFAULT = "gpahal/bge-m3-onnx-int8"
    FILES = ("model_quantized.onnx", "tokenizer.json")
    dim = 1024
    supports_sparse = True

    def __init__(self, model: str | None = None, batch_size: int | None = None, doc_max_len: int = 1024,
                 query_max_len: int = 128, threads: int | None = None, min_sparse_weight: float = 0.01,
                 use_sparse: bool = True):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.model_name = model or self.DEFAULT
        self.device = "cpu-onnx"
        path = self._resolve(self.model_name)
        onnx_file = next(p for p in (path / "model_quantized.onnx", path / "model.onnx") if p.exists())
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or (os.cpu_count() or 2)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_file), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.pad_id = self.tok.token_to_id("<pad>")
        self.special = {self.tok.token_to_id(t) for t in ("<s>", "</s>", "<pad>", "<unk>")}
        self.batch_size = batch_size or 8
        self.doc_max_len, self.query_max_len = doc_max_len, query_max_len
        self.min_w, self.supports_sparse = min_sparse_weight, use_sparse

    def _resolve(self, name: str):
        from pathlib import Path
        p = Path(name).expanduser()
        if p.is_dir():
            return p
        cache = Path(os.environ.get("DART_MODEL_CACHE", Path.home() / ".cache" / "dart_rag")) / name.replace("/", "__")
        cache.mkdir(parents=True, exist_ok=True)
        import requests
        for f in self.FILES:
            dst = cache / f
            if dst.exists():
                continue
            print(f"[embed] 다운로드 {name}/{f}")
            with requests.get(f"https://huggingface.co/{name}/resolve/main/{f}", stream=True, timeout=60) as r:
                r.raise_for_status()
                tmp = dst.with_suffix(".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(dst)
        return cache

    def _encode(self, texts: list[str], max_len: int) -> EmbeddingBatch:
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=self.pad_id, pad_token="<pad>")
        dense = np.zeros((len(texts), self.dim), dtype=np.float32)
        sparse: list[dict[int, float]] = [{} for _ in texts]
        for idx, batch in _length_sorted_batches(texts, self.batch_size):
            enc = self.tok.encode_batch(batch)
            ids = np.asarray([e.ids for e in enc], dtype=np.int64)
            mask = np.asarray([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            out = dict(zip([o.name for o in self.sess.get_outputs()],
                           self.sess.run(None, {k: v for k, v in feed.items() if k in self.inputs})))
            dense[idx] = out["dense_vecs"]
            if self.supports_sparse:
                w = out["sparse_vecs"].reshape(ids.shape[0], ids.shape[1])
                for row, j in enumerate(idx):
                    d: dict[int, float] = {}
                    for tid, wt in zip(ids[row], w[row]):   # 같은 토큰은 최댓값 (FlagEmbedding 과 동일)
                        tid = int(tid)
                        if tid in self.special or wt < self.min_w:
                            continue
                        if wt > d.get(tid, 0.0):
                            d[tid] = float(wt)
                    sparse[j] = d
        return EmbeddingBatch(_normalize(dense), sparse if self.supports_sparse else None)

    def embed_documents(self, texts):
        return self._encode(texts, self.doc_max_len)

    def embed_query(self, text):
        return self._encode([text], self.query_max_len)


# ---------------------------------------------------------------- OpenAI 호환 API
def is_quota_exhausted(e: Exception) -> bool:
    """429 중에서도 '한도·크레딧 소진'(insufficient_quota 등)은 재시도해도 소용없다. 일시적 속도 제한과 구분."""
    msg = str(getattr(e, "message", "") or e).lower()
    return getattr(e, "status_code", None) == 429 and any(
        k in msg for k in ("insufficient_quota", "exceeded your current quota", "quota exceeded",
                           "resource_exhausted", "billing"))


class OpenAICompatEmbedder(Embedder):
    """OpenAI 및 OpenAI 호환 임베딩 API (Upstage 등 base_url 로 교체)."""
    DEFAULT = "text-embedding-3-small"

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 dim: int | None = None):
        from openai import OpenAI
        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"임베딩 API 키({api_key_env})가 없습니다.")
        self.client = (OpenAI(api_key=key, base_url=base_url, max_retries=0) if base_url
                       else OpenAI(api_key=key, max_retries=0))   # 재시도는 _create_with_retry 에서
        self.model_name = model or self.DEFAULT
        self.device = "api"
        self.dim = dim or {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}.get(self.model_name, 0)
        self.tokens_used = 0      # 이 프로세스에서 쓴 누적 토큰 (문서+질문)

    # 재시도할 오류: 속도 제한(429), 서버 오류(5xx), 연결·타임아웃
    MAX_RETRIES = 6
    BATCH = 100
    TIMEOUT_QUERY = 10.0      # 초. 질문 1개 임베딩 (응답 없으면 끊고 재시도)
    TIMEOUT_DOCS = 60.0       # 초. 문서 100개 배치
    PRICE_PER_1M = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}  # USD, 2026-10 확인

    def _create_with_retry(self, batch: list[str], timeout: float | None = None):
        import openai
        retryable = (openai.RateLimitError, openai.APIConnectionError,
                     openai.APITimeoutError, openai.InternalServerError)
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self.client.embeddings.create(model=self.model_name, input=batch,
                                                     timeout=timeout or self.TIMEOUT_DOCS)
            except retryable as e:
                if is_quota_exhausted(e) or attempt == self.MAX_RETRIES:
                    raise                                   # 한도·크레딧 소진은 기다려도 풀리지 않음
                wait = min(60, 2 ** attempt * 2)        # 2, 4, 8, 16, 32, 60초
                print(f"  [embed] {type(e).__name__} → {wait}초 후 재시도 ({attempt + 1}/{self.MAX_RETRIES})",
                      flush=True)
                time.sleep(wait)

    def _embed(self, texts, log_usage: bool = False, timeout: float | None = None):
        out: list = []
        tokens = 0
        n_batches = (len(texts) + self.BATCH - 1) // self.BATCH
        t0 = time.time()
        for bi, i in enumerate(range(0, len(texts), self.BATCH), 1):
            r = self._create_with_retry([t[:8000] for t in texts[i:i + self.BATCH]], timeout)
            out += [d.embedding for d in r.data]
            tokens += getattr(getattr(r, "usage", None), "total_tokens", 0) or 0
            if log_usage and n_batches > 1 and (bi % 10 == 0 or bi == n_batches):
                print(f"  [embed] {bi}/{n_batches} 배치 · {tokens:,} 토큰", flush=True)
        self.tokens_used += tokens
        if log_usage:
            self._log_usage(len(texts), tokens, time.time() - t0)
        M = _normalize(np.asarray(out))
        self.dim = M.shape[1]
        return M

    def _log_usage(self, n_texts: int, tokens: int, seconds: float) -> None:
        """실측 토큰·비용을 JSONL 로 누적 (경로: EMBED_USAGE_LOG, 기본 data/embed_usage.jsonl)."""
        import json
        price = self.PRICE_PER_1M.get(self.model_name)
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "model": self.model_name, "n_texts": n_texts,
               "tokens": tokens, "seconds": round(seconds, 2),
               "usd": round(tokens / 1e6 * price, 6) if price is not None else None}
        path = os.environ.get("EMBED_USAGE_LOG", os.path.join("data", "embed_usage.jsonl"))
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError as e:                              # 기록 실패가 임베딩을 막지 않게
            print(f"  [embed] 사용량 기록 실패: {e}", flush=True)
        usd = f" · ${rec['usd']:.4f}" if rec["usd"] is not None else ""
        print(f"  [embed] {n_texts:,}개 · {tokens:,} 토큰{usd} · {seconds:.1f}초", flush=True)

    def embed_documents(self, texts):
        return EmbeddingBatch(self._embed(texts, log_usage=True))

    def embed_query(self, text):
        return EmbeddingBatch(self._embed([text], timeout=self.TIMEOUT_QUERY))
