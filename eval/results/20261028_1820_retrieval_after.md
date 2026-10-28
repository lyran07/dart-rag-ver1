# 질의응답 평가 2026-10-28 18:20 after

평가셋 eval/qa_eval.yaml · 56문항

## 검색 품질 (Recall@5, MRR)

| 채널 | 유형 | 문항 | 성공(Hit@k, 모든 그룹) | 그룹 Recall | MRR |
|---|---|---:|---:|---:|---:|
| dense | fact | 10 | 90% | 90% | 0.82 |
| dense | synonym | 6 | 50% | 50% | 0.37 |
| dense | concept | 5 | 40% | 40% | 0.40 |
| dense | compare | 10 | 50% | 62% | 0.68 |
| dense | trend | 5 | 40% | 55% | 0.52 |
| dense | calc | 6 | 100% | 100% | 0.92 |
| dense | relative | 3 | 100% | 100% | 0.50 |
| dense | cause | 6 | 83% | 83% | 0.75 |
| **dense** | **전체** | 51 | **69%** | **69%** | **0.65** |
| dense+bm25 | fact | 10 | 100% | 100% | 1.00 |
| dense+bm25 | synonym | 6 | 67% | 67% | 0.58 |
| dense+bm25 | concept | 5 | 100% | 100% | 0.67 |
| dense+bm25 | compare | 10 | 60% | 76% | 0.75 |
| dense+bm25 | trend | 5 | 60% | 73% | 0.67 |
| dense+bm25 | calc | 6 | 100% | 100% | 1.00 |
| dense+bm25 | relative | 3 | 100% | 100% | 0.83 |
| dense+bm25 | cause | 6 | 83% | 83% | 0.83 |
| **dense+bm25** | **전체** | 51 | **82%** | **84%** | **0.81** |
