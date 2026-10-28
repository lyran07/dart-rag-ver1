"""임베딩 처리량 측정 (GPU 대여 첫 시간에 1회 실행 → 한 번 대여로 몇 개사를 처리할지 계산).

python scripts/bench_embed.py --n 1000              # 카탈로그의 청크 아티팩트에서 샘플
python scripts/bench_embed.py --n 1000 --device cpu # 노트북 CPU 와 비교
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_rag.config import Settings  # noqa: E402
from dart_rag.embedding import gpu_info, make_embedder  # noqa: E402
from dart_rag.pipeline import StageContext  # noqa: E402
from dart_rag.pipeline.stages import ChunkStage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int)
    a = ap.parse_args()
    ov = {k: v for k, v in {"embedding.device": a.device, "embedding.batch_size": a.batch_size}.items() if v}
    s = Settings.load(overrides=ov)
    ctx = StageContext(s)
    kinds = set(s.embedding.embed_kinds)
    texts: list[str] = []
    for f in ctx.catalog.filings():
        try:
            texts += [c.text for c in ctx.store.load_chunks(f, ChunkStage(ctx).version()) if c.kind in kinds]
        except FileNotFoundError:
            continue
        if len(texts) >= a.n:
            break
    if not texts:
        sys.exit("청크 아티팩트가 없습니다. 먼저 run --stages fetch,parse,chunk")
    texts = texts[:a.n]
    e = make_embedder(s.embedding)
    e.embed_documents(texts[:8])  # warm-up
    t = time.perf_counter()
    b = e.embed_documents(texts)
    dt = time.perf_counter() - t
    rate = len(texts) / dt
    print(f"embedder={e.describe()} gpu={gpu_info()}")
    print(f"{len(texts)} chunks, 평균 {sum(map(len, texts)) / len(texts):.0f}자 → {dt:.1f}s = {rate:.1f} chunks/s")
    for n, label in [(16_000, "삼성전자 8건"), (300_000, "30개사×3년"), (1_000_000, "KOSPI200×3년 (임베딩 대상)")]:
        print(f"  {label:28s} {n:>9,} chunks ≈ {n / rate / 3600:6.2f} h")
    print(f"  dense {b.dense.shape}, sparse={'yes' if b.sparse else 'no'}")


if __name__ == "__main__":
    main()
