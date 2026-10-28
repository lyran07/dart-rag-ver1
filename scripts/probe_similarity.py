"""임베더의 의미 유사도 감 잡기: 질문 표현 ↔ 공시 원문 표현 코사인.
python scripts/probe_similarity.py --embed bge-m3-onnx
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dart_rag.config import EmbeddingCfg  # noqa: E402
from dart_rag.embedding import make_embedder  # noqa: E402

PAIRS = [
    ("영업손익 규모", "영업이익(제55기)=6,566,976 백만원"),
    ("인건비 부담", "연간급여총액=8,927,760 백만원; 1인평균급여액=121 백만원"),
    ("현금 창출력이 좋아졌어?", "영업활동으로 인한 현금흐름은 44조 1,374억원으로 전년 대비 감소하였습니다."),
    ("반도체 업황 전망", "메모리 시장은 AI 서버 수요 확대로 HBM 등 고부가 제품 중심의 수요 회복이 예상됩니다."),
    ("실적이 왜 나빠졌어?", "IT 수요 부진에 따른 메모리 가격 하락으로 DS 부문 영업손실이 확대되었습니다."),
    ("빚이 얼마야", "부채총계(제55기)=92,228,115 백만원"),
    # 대조군 (관련 없음)
    ("인건비 부담", "당사의 본점은 경기도 수원시 영통구 삼성로 129에 소재하고 있습니다."),
    ("반도체 업황 전망", "발행할 주식의 총수=25,000,000,000 주"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", default="bge-m3-onnx")
    ap.add_argument("--model")
    a = ap.parse_args()
    e = make_embedder(EmbeddingCfg(provider=a.embed, model=a.model))
    t = time.perf_counter()
    for q, d in PAIRS:
        qv, dv = e.embed_query(q).dense[0], e.embed_documents([d]).dense[0]
        print(f"{float(qv @ dv):.3f}  {q:18s} ↔ {d[:48]}")
    print(f"({e.key}, {time.perf_counter() - t:.1f}s)")


if __name__ == "__main__":
    main()
