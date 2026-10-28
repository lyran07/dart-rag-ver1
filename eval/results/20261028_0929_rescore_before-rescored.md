# 질의응답 평가 2026-10-28 09:29 before-rescored

평가셋 eval/qa_eval.yaml · 56문항

## 다시 채점: eval/results/20261028_1752_answer-auto.json, eval/results/20261028_1811_answer-auto_before-new.json

### 모드: auto

| 유형 | 문항 | 답변 정확도 | 결론 정확도 | 근거 섹션 적중 | 미검증 근거 비율 | 에이전트 전환 | 중앙 시간 | 90% 시간 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fact | 10 | 100% | - | 100% | 0% | 0% | 4.1s | 5.1s |
| synonym | 6 | 100% | - | 100% | 0% | 17% | 5.1s | 5.3s |
| concept | 5 | 100% | - | 100% | 0% | 0% | 8.5s | 9.5s |
| compare | 10 | 80% | 90% | 100% | 7% | 50% | 10.0s | 23.6s |
| trend | 5 | 100% | 100% | 100% | 0% | 0% | 6.1s | 7.4s |
| calc | 6 | 83% | 83% | 100% | 0% | 0% | 4.8s | 5.3s |
| relative | 3 | 100% | 100% | 100% | 0% | 0% | 4.0s | 4.2s |
| cause | 6 | 100% | - | 83% | 0% | 17% | 8.3s | 9.1s |
| negative | 5 | 100% | - | - | - | 100% | 6.3s | 13.3s |
| **전체** | 56 | **95%** | 91% | 98% | 2% | 21% | 5.7s | 14.4s |

- 형식 준수율 100% · 오류 0건 · 규칙 확정으로 LLM 라우터 생략 0건 · LLM 라우터 실패(한도·시간 초과) 0건 · LLM 비용 합계 $0.0370 (질문당 $0.00066)

틀린 문항
- [compare] 삼성전자, SK하이닉스, 현대자동차의 2026년 반기 보통주 주당 배당금을 비교해줘 — answered · 채점 {"values": [true, true, true], "keywords": [], "conclusion": {"expect_max": false, "expect_min": true}, "conclusion_ok": false}
- [compare] 삼성전자와 현대자동차 2026년 반기 연구개발비 비교해줘 — needs_clarification · 채점 {"values": [true, true], "keywords": [], "conclusion": {"expect_max": true}, "conclusion_ok": true, "clarification": true}
- [calc] 현대차 2026년 반기 매출은 전년 반기보다 몇 % 늘었어? — answered · 채점 {"values": [], "keywords": [], "conclusion": {"expect_calc": false}, "conclusion_ok": false}
