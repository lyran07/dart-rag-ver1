# DART 공시 질의응답 백엔드 (v0.3)

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

`data/` 폴더는 저장소에 없습니다. 팀 드라이브의 `data.zip` 을 프로젝트 폴더에 풀면 바로 질문할 수 있습니다.
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
pytest -q                                    # 50개 (PostgreSQL 전용 3개는 건너뜀)

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
python -m dart_rag run --latest-only         # fetch → parse → chunk → embed → index → wiki
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
config/  corp_lists/  scripts/  tests/  eval/  web/index.html
data/    (저장소 미포함) raw · parsed · chunks · emb · index · catalog.db · docstore.db · wiki
```

`docs/` 의 문서는 v0.2 기준이라 auto 모드·에이전트 내용이 없습니다.

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
