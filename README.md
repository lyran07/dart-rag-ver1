# DART 공시 질의응답 백엔드 (v0.3)

> 코드 구조와 설계 이유: [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)

DART 정기보고서(사업·반기·분기)를 수집·가공해 두고, 자연어 질문에 공시 원문 근거를 붙여 답하는 FastAPI 백엔드입니다.

- 수집·가공: OpenDART 원문 → 섹션·표 파싱 → 청킹 → OpenAI 임베딩(text-embedding-3-small) → SQLite + 메모리 인덱스(dense + BM25) + 위키
- 답변: 라우터(Gemini)가 경로 선택(wiki / hybrid / vector / agent) → GPT-6 Luna 답변 → 인용 숫자를 원문과 대조 검증
- 대상 기업: 삼성전자, SK하이닉스, 현대자동차 (`corp_lists/team1.txt`)

## 1. 처음 받았을 때 (Windows)

```powershell
cd dart-rag-backend-main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

copy .env.example .env      # .env 를 열어 키 3개 입력 (아래 표)
```

| 키 | 용도 | 발급 |
|---|---|---|
| `OPENDART_API_KEY` | 공시 원문 수집 | https://opendart.fss.or.kr |
| `OPENAI_API_KEY` | GPT-6 Luna 답변, 질문 임베딩 | https://platform.openai.com |
| `GEMINI_API_KEY` | auto 모드 라우터 | https://aistudio.google.com |

`.env` 는 각자 만들어 쓰고 git 에 올리지 않습니다 (`.gitignore` 에 포함).

### 데이터 준비

`data/` 폴더(공시 원문·인덱스·위키·보고서, 압축 약 205MB)는 용량 때문에 코드와 따로 둡니다.
GitHub 저장소 오른쪽 **Releases** 에서 `data.zip` 을 받아 프로젝트 폴더에 풀면(`data/` 폴더가 생김) 바로 질문할 수 있습니다.

```powershell
Expand-Archive data.zip -DestinationPath .     # 프로젝트 폴더에서
```
직접 만들려면 아래 3번(데이터 수집)을 실행하세요. 임베딩에 OpenAI API 비용이 듭니다.

### 실행

```powershell
uvicorn dart_rag.api:app --port 8080        # http://localhost:8080 테스트 화면, /docs API 문서
python -m dart_rag ask "SK하이닉스 2026년 반기 연결 매출액은?"
python -m dart_rag chat                      # 대화형
```

## 2. 키 없이 동작 확인 · 테스트

```powershell
pip install -r requirements-dev.txt
pytest -q                                    # 72개 (PostgreSQL 전용 3개는 건너뜀)

$env:EMBED_PROVIDER="hashing"; $env:DART_DATA_ROOT="data_sample"
python -m dart_rag sample                    # 합성 샘플로 전체 파이프라인 (키 불필요)
```

## 3. 데이터 수집

```powershell
# 한 번에: team1.txt 3개 기업, 2025-01-01~오늘, 반기·분기보고서
powershell -ExecutionPolicy Bypass -File scripts\fetch_dart.ps1
powershell -ExecutionPolicy Bypass -File scripts\fetch_dart.ps1 -Limit 6     # 먼저 6건만 시험

# 단계별로
python -m dart_rag discover --corp-file corp_lists/team1.txt --from 20250101 --to 20261231
python -m dart_rag run --latest-only         # fetch → parse → chunk → embed → index → wiki → report
python -m dart_rag status                    # 진행 상황·용량
```

끊겨도 다시 실행하면 끝난 단계는 건너뜁니다. 파서·청커 설정을 바꾸면 해당 단계부터 자동으로 다시 실행됩니다.

## 4. 답변 모드

| 모드 | 동작 | 지정 방법 |
|---|---|---|
| `auto` (기본) | 라우터가 wiki / hybrid / vector / agent 중 경로 선택 → 답변 1회. 근거가 없으면 agent 로 전환 | `--mode auto`, API `"mode"`, `ASK_MODE` |
| `agent` | tool calling 에이전트가 도구 4개로 최대 4라운드 조사 | `--mode agent` |
| `rag` | 검색 1회 + LLM 1회 (auto·agent 실패 시 대체 경로) | `--mode rag` |

`python -m dart_rag route "질문"` 으로 어떤 경로가 선택되는지만 확인할 수 있습니다.

## 4-1. 기업분석 보고서

공시 1건(선택적으로 전년 동기 공시와 비교)을 투자자 관점의 섹션으로 정리합니다.
섹션: 기업 개요 · 실적·재무 요약 · 매출 및 수주 · 원재료·생산설비 · 리스크 요인 · 경영진 시각 · 주주·배당 · 종합 의견 (`config/report_templates.yaml`)

- **파이프라인**: `run` 의 마지막 `report` 단계가 공시마다 단일 보고서(비교 없음)를 만들어 `data/reports/` 에 저장합니다. `OPENAI_API_KEY` 필요.
- **웹**: `보고서` 탭에서 기업·공시를 고르면 저장본은 즉시 열립니다. 전년 동기 비교는 선택 시 생성(15초 안팎)하고 저장합니다.
- **CLI**: 한 건만 만들어 확인

```powershell
python -m dart_rag report SK하이닉스 2026.06 --out report.md                  # 단일 (기본)
python -m dart_rag report SK하이닉스 2026.06 --compare auto --out report.md   # 전년 동기 비교 (또는 2025.06)
python -m dart_rag run --stages report --latest-only                          # 기존 공시 전부 보고서 생성
```

섹션마다 위키 페이지를 읽고 GPT-6 Luna 를 1회 호출합니다 (섹션 병렬, 종합 의견은 마지막). 인용 숫자와 표 숫자를 원문과 대조해 불일치를 표시하고, 근거 수·잘린 페이지·시간·토큰을 섹션별 진단으로 남깁니다.
프롬프트나 템플릿을 바꾸면 버전이 바뀌어 저장본 대신 새로 생성합니다.

## 4-2. 성능 평가

`eval/qa_eval.yaml` (56문항, 정답은 위키 원문에서 확인한 값)으로 검색·라우터·답변을 측정합니다. 결과는 `eval/results/`.

```powershell
python scripts/eval_qa.py --check                     # 정답이 원문에 있는지 검사 (0원)
python scripts/eval_qa.py --route rules gemini        # 경로·회사·기간 인식 (약 4분, Gemini 무료 등급 속도 제한 적용)
python scripts/eval_qa.py --retrieval                 # 검색 Recall@5·MRR, dense vs dense+BM25 (거의 0원)
python scripts/eval_qa.py --answer auto               # 답변 정확도·결론·근거·시간·비용 (약 6분, 60원 안팎)
python scripts/eval_qa.py --answer auto --types compare --label test   # 유형 일부만
python scripts/eval_qa.py --answer auto --q "직원 수"                   # 질문 문구로 골라서
python scripts/eval_qa.py --rescore eval/results/A.json               # 저장된 답변을 다시 채점 (0원)
```

현재 결과 (auto, 56문항): 답변 정확도 98%, 결론 정확도 95%, 응답 중앙값 4.4초, 질문당 약 1원.
BM25 결합으로 검색 성공률 58% → 78% (dense 단독 대비).

## 5. 주요 설정

설정 파일은 `config/`, 우선순위는 **환경변수(.env) > YAML > 코드 기본값** 입니다.

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `EMBED_PROVIDER` | `openai` | `openai` \| `bge-m3` \| `bge-m3-onnx` \| `e5` \| `hashing`. 바꾸면 embed·index 단계를 다시 실행 |
| `DOCSTORE_BACKEND` | `sqlite` | 청크 본문 저장소. `postgres` 는 docker 필요 |
| `INDEX_BACKEND` | `memory` | 벡터 인덱스. `qdrant` 는 docker 또는 로컬 파일 모드 |
| `ASK_MODE` | `auto` | 답변 모드 |
| `LLM_ORDER` | `gemini,gemini_lite,openai` | rag 모드의 LLM 시도 순서 (`config/providers.yaml`) |
| `DART_DATA_ROOT` | `data` | 산출물 저장 위치 |

| 파일 | 내용 |
|---|---|
| `config/settings.yaml` | 청킹 크기, 임베딩, 저장소, 검색 후보 수 |
| `config/agent.yaml` | LLM 모델, 라운드·시간 제한, 회사 별칭 |
| `config/wiki_routes.yaml` | 질문 주제 → 위키 섹션 연결표 (20개) |
| `config/lexicon.yaml` | 동의어 사전 (BM25 검색 보강) |
| `config/providers.yaml` | rag 모드 LLM 목록 |
| `config/report_templates.yaml` | 보고서 섹션 구성·지시문·글자 예산 |

## 6. 폴더

```
dart_rag/
  domain.py config.py service.py api.py cli.py wiki.py
  ingestion/   dart_client.py sources.py parser.py chunker.py   DART API · 파싱 · 청킹
  pipeline/    stages.py runner.py catalog.py storage.py        단계 실행 · 진행 기록
  embedding/   base.py providers.py device.py                   임베더
  index/       base.py memory.py qdrant.py bm25.py tokenizer.py docstore.py
  retrieval/   query.py retriever.py reranker.py                하이브리드 검색
  routing/     router.py resolve.py routes.py auto.py           auto 모드
  agent/       tools.py loop.py prompts.py llm.py config.py     tool calling 에이전트
  generation/  answerer.py      llm/ router.py                   rag 모드
  report/      builder.py templates.py                          기업분석 보고서
config/  corp_lists/  scripts/  tests/  eval/  web/index.html
data/    (저장소 미포함) raw · parsed · chunks · emb · index · catalog.db · docstore.db · wiki
```

코드 구조·설계 이유는 **[docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)** 를 보세요. (`docs/ARCHITECTURE.md` 는 v0.2 기준이라 auto 모드·에이전트·보고서 내용이 없습니다.)

## 7. 선택 구성

### BGE-M3 로컬 임베딩 (GPU·API 없이)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```
`.env.local.example` 기준으로 BGE-M3 ONNX int8(570MB, 첫 실행 때 내려받음) + SQLite + 메모리 인덱스로 동작합니다.
OpenAI 임베딩으로 만든 `data/` 와는 인덱스가 따로 만들어지므로 `run --stages embed,index` 를 다시 실행해야 합니다.
torch 기반 BGE-M3·리랭커는 `requirements-ml.txt` 가 필요합니다.

### PostgreSQL + Qdrant

```powershell
docker compose up -d                         # PostgreSQL 16 + Qdrant
# .env: DOCSTORE_BACKEND=postgres, DATABASE_URL=postgresql://dart:dart@localhost:5432/dart
#       INDEX_BACKEND=qdrant, QDRANT_URL=http://localhost:6333
python -m dart_rag run --stages index,wiki --force
```

### API 보호

- `API_ACCESS_TOKEN` 을 설정하면 `/ask`, `/search`, `/feedback` 에 `X-Access-Token` 헤더가 필요합니다.
- `/ask` 는 IP당 분당 `ASK_RATE_PER_MIN`(기본 6)회로 제한됩니다.
- 응답 `status`: `answered` / `no_evidence` / `needs_clarification` / `extractive_fallback`
