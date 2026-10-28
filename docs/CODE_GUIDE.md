# 코드 가이드 (v0.3, 2026-10-28)

DART 정기보고서를 미리 가공해 두고, 질문에는 공시 원문 근거를 붙여 답하며, 공시 1건을 기업분석 보고서로 정리하는 백엔드다.
설계 원칙은 하나다. **근거는 공시 구조(위키)와 검색이 정하고, LLM은 읽고 쓰기만 하며, 숫자 판단은 코드가 검증한다.**

> 처음 실행하는 방법은 [README](../README.md), 측정 결과는 `eval/results/`. 이 문서는 "어디에 무엇이 있고 왜 그렇게 만들었나"만 다룬다.
> `docs/ARCHITECTURE.md` 는 v0.2 기준이라 auto 모드·에이전트·보고서 내용이 없다.

---

## 1. 전체 흐름

```
[오프라인 · 미리 한 번]  python -m dart_rag run
  fetch → parse → chunk → embed → index → wiki → report
  DART XML   섹션·표   검색 단위   OpenAI    SQLite +   기업/보고서/   공시 1건당
  다운로드    트리      청크        임베딩    메모리 인덱스 섹션 페이지    보고서 저장
                                                  └──────────── data/ ────────────┘

[온라인 · 질문마다]  POST /ask (기본 auto 모드)
  질문 → 라우터 ─┬─ wiki   : 위키 섹션 페이지 직행 (검색·임베딩 없음)      ─┐
  (규칙 우선,    ├─ hybrid : dense + BM25 검색                              ├→ Luna 답변 1회 → 근거 숫자 검증
   애매하면      ├─ vector : dense 검색 (원인·전망)                         ─┘   → 비교 계산 검증 → 응답
   Gemini)       └─ agent  : tool calling 에이전트 (복합·답 없음)            ← 근거를 못 찾으면 여기로 전환

[보고서]  POST /report  또는 웹 '보고서' 탭
  저장본(data/reports/) 있으면 즉시 반환 · 없으면 섹션별 위키 근거 → Luna 병렬 작성 → 검증 → 저장
```

---

## 2. 핵심 설계 포인트 (발표용)

| # | 포인트 | 어디 | 근거 수치 |
|---|---|---|---|
| 1 | **공시 구조를 살린 파싱·청킹**: 목차·표를 보존하고 표 행을 `매출액(제55기)=258,935,494` 같은 검색용 텍스트로 풀어 둔다 | `ingestion/parser.py`, `chunker.py` | 위키·검색·보고서가 모두 이 위에서 동작 |
| 2 | **위키 직행 경로**: 답이 한 섹션에 있는 질문은 검색 없이 해당 섹션 페이지를 바로 읽는다 | `wiki.py`, `routing/auto.py` | 근거 수집 수 ms |
| 3 | **하이브리드 검색**: dense(의미) + BM25(정확한 단어·숫자)를 RRF 로 결합 | `retrieval/retriever.py`, `index/` | 검색 성공률 dense 58% → dense+BM25 78% |
| 4 | **규칙 우선 라우팅**: 규칙이 회사·기간·섹션을 확신하면 LLM 라우터를 건너뛴다 | `routing/auto.py`, `router.py` | 56문항 중 40문항 생략, 응답 중앙값 5.7 → 4.4초 |
| 5 | **근거·비교 검증**: 인용 숫자는 원문 대조, 비교·증감 결론은 코드가 순위·증감률을 계산해 대조 | `agent/loop.py`, `agent/compare.py` | 미검증 근거 0%, 결론 정확도 91% → 95% |
| 6 | **보고서 = 위키 근거 + LLM 작성 + 저장본** | `report/` | 생성 약 14초, 공시당 약 6원, 저장본은 즉시 |

---

## 3. 질문 처리 (auto 모드) 따라가기

1. `api.py` `POST /ask` → `service.py` `RAGService.ask()` → `routing/auto.py` `AutoAnswerer.ask()`
2. **경로 판정** `AutoAnswerer.route()`
   - 규칙 라우터 `RuleRouter` (`routing/router.py`): `config/wiki_routes.yaml` 키워드, 별칭(`config/agent.yaml`), 기간 규칙(`resolve.py`)으로 판정. 0초
   - 규칙이 wiki 로 확정하지 못하면 LLM 라우터 `LLMQueryRouter`(Gemini)에게 JSON 으로 경로·회사·기간·섹션을 받는다. 보유하지 않은 회사·기간은 코드가 버린다
   - Gemini 가 실패·6초 초과·한도 소진이면 규칙 결과로 진행
3. **근거 수집**
   - wiki: `gather_wiki()` — 위키 섹션 페이지를 읽고 길면 질문 관련 문단만 발췌. 여러 페이지면 글자 예산을 페이지 수로 나눈다
   - hybrid/vector: `gather_search()` → `HybridRetriever.retrieve()` (회사·기간 필터)
4. **답변** `_answer()` — Luna 1회, JSON 스키마(`agent/prompts.py` `ANSWER_SCHEMA`) 강제. 출력이 잘리면 1회 짧게 재작성
5. **검증**
   - `check_and_fix()` (`agent/compare.py`): 비교·증감 질문이면 `comparison` 값으로 순위·증감률 계산 → 서술이 다르면 1회 재작성 → 답변 끝에 "계산 확인"
   - `verify_evidence()` (`agent/loop.py`): 인용 chunk 를 실제로 조회했는지, 인용 숫자가 원문에 있는지
6. 근거를 하나도 인용하지 못하면 에이전트(`agent/loop.py` `AgentLoop`)로 전환. 에이전트는 도구 4개
   (`resolve_company`, `list_filings`, `search_disclosures`, `get_section`, `agent/tools.py`)로 최대 4라운드 조사

모드: `auto`(기본) / `agent`(에이전트 강제) / `rag`(검색 1회 + LLM 1회, auto·agent 실패 시 대체). API `"mode"` 또는 `ASK_MODE`.

---

## 4. 보고서 생성

- 템플릿 `config/report_templates.yaml`: 섹션별로 읽을 위키 라우트, 지시문, 표 필요 여부, 공시 유형 제한(`report_types`)
- 흐름 `report/builder.py` `ReportBuilder`
  1. 저장본 확인: `data/reports/{corp_code}/{접수번호}[__vs_{비교접수번호}].company_analysis.json`, 버전이 같으면 즉시 반환
  2. 섹션마다 위키 근거 수집(검색 없음). 키워드 발췌가 너무 짧으면 페이지 앞부분으로 보완
  3. Luna 로 섹션 작성 (특징 headline → 설명 → 데이터 표 → 해석 → 확인할 점). 일반 섹션은 병렬, "한눈에 보기"는 마지막에 다른 섹션 결과만 읽고 작성
  4. 인용 숫자·표 숫자 원문 대조 → 오류 섹션이 없으면 저장
- 버전: 프롬프트·템플릿·모델 해시. **해당 공시 유형에 들어가는 섹션만** 해시에 넣어, 사업보고서 전용 섹션을 고쳐도 반기·분기 저장본은 유지
- 진입점: 파이프라인 `report` 단계(단일 보고서 미리 생성), `POST /report`(웹, 비교 보고서는 요청 시 생성), `python -m dart_rag report 기업 기간`

---

## 5. 폴더 지도

| 위치 | 역할 | 자주 손댈 일 |
|---|---|---|
| `dart_rag/domain.py` | 계층 간에 오가는 데이터 타입 (Filing, Section, Table, Chunk, Answer) | 거의 없음 |
| `dart_rag/service.py` | 구성 요소 조립 (모든 진입점이 이것만 사용) | 새 구성 요소 연결 |
| `dart_rag/api.py`, `cli.py`, `web/index.html` | HTTP API, 터미널 명령, 테스트 화면 | 기능 노출 |
| `dart_rag/ingestion/` | DART API, XML 파싱, 청킹 | 파싱 개선 |
| `dart_rag/pipeline/` | 단계 실행·이어하기·버전 관리, `catalog.db` | 새 단계 추가 (`Stage` 상속 → `STAGES`) |
| `dart_rag/embedding/`, `index/` | 임베더, 메모리/Qdrant 인덱스, BM25, SQLite/Postgres 문서 저장소 | DB 교체 |
| `dart_rag/retrieval/` | 질문 분석(회사·연도·별칭·동의어), 하이브리드 검색, 리랭커 | 검색 품질 |
| `dart_rag/routing/` | 경로 판정, auto 답변 | 라우팅 규칙 |
| `dart_rag/agent/` | 에이전트 루프·도구·프롬프트·LLM 호출, 비교 검증 | 도구 추가, 프롬프트 |
| `dart_rag/report/` | 보고서 생성·저장 | 보고서 형식 |
| `dart_rag/wiki.py` | 위키 생성·조회 | |
| `dart_rag/generation/`, `llm/` | rag 모드 (대체 경로) | |
| `scripts/eval_qa.py`, `eval/` | 평가 스크립트·평가셋·결과 | 문항 추가 |
| `tests/` | pytest 72개 (가짜 LLM, API 비용 0원) | |

---

## 6. 설정으로 바꾸는 것 (코드 수정 불필요)

| 하고 싶은 일 | 파일 |
|---|---|
| 질문 주제 → 위키 섹션 연결 추가 | `config/wiki_routes.yaml` |
| 회사 별칭 추가 (예: 현차 → 현대자동차) | `config/agent.yaml` `aliases` |
| 동의어 추가 (예: 빚 → 부채총계, BM25 보강) | `config/lexicon.yaml` |
| 보고서 섹션 추가·지시 변경 | `config/report_templates.yaml` |
| 라우터 종류(gemini/luna/rules), 규칙 우선, 출력 상한, 라운드 수 | `config/agent.yaml` |
| 임베딩·저장소·검색 후보 수 | `config/settings.yaml` 또는 `.env` |

---

## 7. 평가

- 평가셋 `eval/qa_eval.yaml` 56문항 (사실·동의어·개념·비교·추이·계산·상대 기간·원인·답 없음), 정답 숫자는 위키 원문에서 확인한 값
- `scripts/eval_qa.py`: `--check`(정답 원문 검사) · `--route` · `--retrieval` · `--answer` · `--rescore` (README 4-2 참고)

| 지표 (auto, 56문항) | 개선 전 | 개선 후 |
|---|---:|---:|
| 답변 정확도 | 95% | 98% |
| 결론 정확도 (비교·추이·계산) | 91% | 95% |
| 미검증 근거 비율 | 2% | 0% |
| 에이전트 전환 | 21% | 11% |
| 응답 시간 중앙값 / 90% | 5.7초 / 14.4초 | 4.4초 / 10.0초 |
| 질문당 비용 | 약 1원 | 약 1원 |

---

## 8. 알려진 한계와 다음 과제

- **에이전트 속도** (15~22초): 보유 기업·공시 목록을 프롬프트에 미리 넣고, auto 에서 넘길 때 라우터 결과와 근거를 함께 넘기면 1~2라운드를 줄일 수 있다
- **표 단위 누락**: 일부 표(예: 삼성전자 연구개발비)는 파싱 과정에서 단위가 빠진다 → `ingestion/parser.py`
- **Gemini 무료 한도** 분당 15회: 테스터가 많으면 규칙 라우터로 대체된다 → 유료 등급 또는 라우터를 Luna 로
- **계획서와 다른 점**: DB 는 PostgreSQL+pgvector 계획 대비 SQLite + 메모리 인덱스(Postgres·Qdrant 코드는 있음), LangChain 미사용(직접 구현), 대화 이력·재무 수치 구조화 미구현
- **평가셋**: 시스템 구조를 아는 사람이 만든 56문항이라 실제 사용자 질문보다 쉬울 수 있다
