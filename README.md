# DART 공시 질의응답 · 보고서 서비스 (v0.3)

삼성전자, SK하이닉스, 현대자동차의 DART 정기보고서(사업·반기·분기)를 바탕으로 두 가지를 제공합니다.

- **질문**: 자연어로 물으면 공시 원문을 근거로 답하고, 답에 쓴 숫자를 원문과 대조해 검증합니다.
- **보고서**: 공시 1건을 투자자 관점의 보고서(한눈에 보기 → 섹션별 특징·데이터·해석)로 정리합니다.

코드 구조와 설계 이유는 [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)에 있습니다.

---

## 빠른 시작: 웹사이트로 실행하기 (Windows)

아래 5단계를 마치면 브라우저에서 `http://localhost:8080` 으로 서비스를 쓸 수 있습니다.
필요한 것: Python 3.11 이상, 인터넷 연결, API 키 3개.

### 1단계. 설치

PowerShell을 열고 프로젝트 폴더로 이동해 가상환경을 만들고 패키지를 설치합니다.

```powershell
cd "C:\경로\dart-rag-backend-main"      # 이 README가 있는 폴더
python -m venv .venv
.venv\Scripts\activate                  # 프롬프트 앞에 (.venv) 가 붙으면 성공
pip install -r requirements.txt
```

> `activate` 에서 "스크립트를 실행할 수 없습니다" 오류가 나면 한 번만 실행하세요.
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 2단계. API 키 넣기

키는 **프로젝트 폴더 바로 아래의 `.env` 파일**에 넣습니다. 저장소에는 빈 양식 `.env.example` 만 있으므로 복사해서 만듭니다.

```powershell
copy .env.example .env
notepad .env
```

메모장에서 아래 세 줄의 `=` 뒤에 본인 키를 붙여 넣고 저장합니다. 따옴표나 공백 없이 넣습니다.

```ini
OPENDART_API_KEY=발급받은_DART_키
GEMINI_API_KEY=발급받은_Gemini_키
OPENAI_API_KEY=sk-...
```

| 키 | 어디에 쓰나 | 발급처 | 없으면 |
|---|---|---|---|
| `OPENAI_API_KEY` | 답변·보고서 작성(GPT-6 Luna), 질문 임베딩 | https://platform.openai.com → API keys | **질문·보고서 생성 불가** (필수) |
| `GEMINI_API_KEY` | 질문 경로 판정(라우터) | https://aistudio.google.com → Get API key | 규칙 라우터만으로 동작 (정확도 약간 하락) |
| `OPENDART_API_KEY` | 공시 원문 수집 | https://opendart.fss.or.kr → 인증키 신청 | 받은 `data.zip` 만 쓰면 없어도 됨 |

- 서버를 켤 때 `.env` 를 자동으로 읽습니다. 키를 바꿨다면 서버를 껐다 켜세요.
- `.env` 는 `.gitignore` 에 들어 있어 git에 올라가지 않습니다. **키를 코드나 채팅에 붙여 넣지 마세요.**
- 나머지 항목(`EMBED_PROVIDER=openai` 등)은 기본값 그대로 두면 됩니다.

### 3단계. 데이터 받기

공시 원문·인덱스·위키·미리 만든 보고서가 담긴 `data/` 폴더(압축 약 205MB)는 용량 때문에 코드와 따로 배포합니다.

1. [Releases (data-v1)](https://github.com/lyran07/dart-rag-ver1/releases/tag/data-v1) 에서 `data.zip` 을 내려받아 프로젝트 폴더에 둡니다.
2. 프로젝트 폴더에서 압축을 풉니다.

```powershell
Expand-Archive data.zip -DestinationPath .
```

`data\catalog.db`, `data\wiki`, `data\reports` 등이 보이면 성공입니다. (`data\data\...` 처럼 한 겹 더 생겼다면 안쪽 `data` 를 바깥으로 옮기세요.)
데이터를 직접 만들고 싶다면 아래 [데이터 직접 수집](#데이터-직접-수집)을 보세요.

### 4단계. 서버 켜기

```powershell
uvicorn dart_rag.api:app --port 8080
```

`Uvicorn running on http://127.0.0.1:8080` 이 나오면 준비가 끝났습니다. 이 창은 켜 둔 채로 사용합니다.
(첫 질문은 인덱스를 불러오느라 몇 초 더 걸립니다.)

### 5단계. 브라우저에서 사용

브라우저에서 **http://localhost:8080** 을 엽니다.

| 탭 | 사용법 |
|---|---|
| **질문** | 예: `SK하이닉스 2026년 반기 연결 매출액은?`, `세 회사 직원 수 비교`. 답변 아래에 근거 문단과 숫자 검증 결과가 붙습니다. |
| **보고서** | 기업과 공시를 고릅니다. 기본값 `비교 안 함`은 미리 만들어 둔 보고서라 바로 열립니다. `전년 동기`를 고르면 그 자리에서 생성(15초 안팎)하고 저장합니다. 인쇄 버튼으로 PDF 저장도 됩니다. |

- 화면이 예전 모습이면 `Ctrl + F5` 로 새로고침하세요.
- API 문서(직접 호출용)는 http://localhost:8080/docs 에 있습니다.
- 끝낼 때는 서버 창에서 `Ctrl + C`. 다음부터는 `.venv\Scripts\activate` → `uvicorn ...` 두 줄이면 됩니다.

**같은 와이파이의 팀원에게 보여주려면** `--host 0.0.0.0` 을 붙여 실행하고, 팀원은 `http://내PC의IP:8080` 으로 접속합니다. (`ipconfig` 의 IPv4 주소. Windows 방화벽 허용 창이 뜨면 허용)

```powershell
uvicorn dart_rag.api:app --host 0.0.0.0 --port 8080
```

### 자주 나는 문제

| 증상 | 원인과 해결 |
|---|---|
| `OPENAI_API_KEY` 관련 오류, 401 | `.env` 위치(프로젝트 폴더 바로 아래)와 키 값을 확인하고 서버를 재시작 |
| `429` / `RESOURCE_EXHAUSTED` (Gemini) | 무료 등급은 분당 15회 제한. 자동으로 규칙 라우터로 넘어가므로 답변은 계속 나옴 |
| "분당 6회를 초과했습니다" | 서버의 과다 호출 방지 제한. 1분 뒤 다시 시도 (`ASK_RATE_PER_MIN` 으로 조정) |
| 답이 "근거 없음"만 나옴 / 보고서 목록이 비어 있음 | `data/` 가 없거나 위치가 틀림. 3단계 다시 확인 |
| `address already in use` | 8080 포트 사용 중. `--port 8081` 로 바꾸고 주소도 8081로 |
| `uvicorn` 을 찾을 수 없음 | 가상환경이 꺼져 있음. `.venv\Scripts\activate` 후 다시 실행 |
| `.env` 가 메모장에서 `.env.txt` 로 저장됨 | `ren .env.txt .env` |

---

## 명령줄로 쓰기

```powershell
python -m dart_rag ask "SK하이닉스 2026년 반기 연결 매출액은?"
python -m dart_rag chat                                   # 대화형
python -m dart_rag route "질문"                            # 어떤 경로로 갈지만 확인
python -m dart_rag report SK하이닉스 2026.06 --out report.md                 # 보고서 1건 (단일)
python -m dart_rag report SK하이닉스 2026.06 --compare auto --out report.md  # 전년 동기 비교
```

## 답변 모드

| 모드 | 동작 | 지정 방법 |
|---|---|---|
| `auto` (기본) | 라우터가 wiki / hybrid / vector / agent 중 경로를 골라 1회 답변. 근거가 없으면 agent로 전환 | `--mode auto`, API `"mode"`, `ASK_MODE` |
| `agent` | 에이전트가 도구 4개로 최대 4라운드 조사 (느리지만 복잡한 질문에 강함) | `--mode agent` |
| `rag` | 검색 1회 + LLM 1회 (대체 경로) | `--mode rag` |

## 보고서

공시 1건(선택적으로 전년 동기 공시와 비교)을 다음 섹션으로 정리합니다. 각 섹션은 특징 → 데이터 → 해석 순서로 쓰고, 근거가 없는 섹션은 빼고 보여 줍니다.

한눈에 보기 · 사업 구조 · 실적 · 판매와 수주 · 생산과 원가 · 리스크 · 경영진 시각(사업보고서만) · 주주와 배당

- 섹션마다 위키 페이지를 읽고 GPT-6 Luna를 1회 호출합니다(병렬). 한눈에 보기는 마지막에 다른 섹션을 요약해 씁니다.
- 인용 숫자와 표 숫자를 원문과 대조해 불일치를 표시합니다.
- 파이프라인의 마지막 `report` 단계가 공시마다 단일 보고서를 만들어 `data/reports/` 에 저장합니다. 템플릿이나 프롬프트를 바꾸면 버전이 바뀌어 새로 만듭니다.
- 섹션 구성·지시문·글자 수: `config/report_templates.yaml`

## 데이터 직접 수집

`OPENDART_API_KEY` 와 `OPENAI_API_KEY` 가 필요하며, 임베딩·보고서 생성에 OpenAI 비용이 듭니다.

```powershell
# 한 번에: corp_lists/team1.txt 3개 기업, 2025-01-01 ~ 오늘
powershell -ExecutionPolicy Bypass -File scripts\fetch_dart.ps1
powershell -ExecutionPolicy Bypass -File scripts\fetch_dart.ps1 -Limit 6    # 6건만 시험

# 단계별로
python -m dart_rag discover --corp-file corp_lists/team1.txt --from 20250101 --to 20261231
python -m dart_rag run --latest-only     # fetch → parse → chunk → embed → index → wiki → report
python -m dart_rag status                # 진행 상황·용량
```

중간에 끊겨도 다시 실행하면 끝난 단계는 건너뜁니다.

## 테스트와 성능 평가

```powershell
pip install -r requirements-dev.txt
pytest -q                                  # 키 없이 실행 (PostgreSQL 전용 3개는 건너뜀)
```

`eval/qa_eval.yaml`(56문항)으로 검색·라우터·답변을 측정합니다. 결과는 `eval/results/` 에 저장됩니다.

```powershell
python scripts/eval_qa.py --check                 # 정답이 원문에 있는지 검사 (비용 0)
python scripts/eval_qa.py --route rules gemini    # 경로·회사·기간 인식 (약 4분)
python scripts/eval_qa.py --retrieval             # 검색 Hit@5·MRR
python scripts/eval_qa.py --answer auto           # 답변 정확도·시간·비용 (약 6분, 60원 안팎)
python scripts/eval_qa.py --answer auto --q "직원 수"   # 질문 일부만
```

현재 결과(auto, 56문항): 답변 정확도 98%, 결론 정확도 95%, 응답 중앙값 4.4초, 질문당 약 1원.

## 설정

우선순위는 **`.env`(환경변수) > `config/*.yaml` > 코드 기본값** 입니다.

| 환경변수 | 기본값 | 의미 |
|---|---|---|
| `EMBED_PROVIDER` | `openai` | 임베딩 모델. 바꾸면 embed·index 단계를 다시 실행해야 함 |
| `DOCSTORE_BACKEND` | `sqlite` | 청크 저장소 (`postgres` 는 docker 필요) |
| `INDEX_BACKEND` | `memory` | 벡터 인덱스 (`qdrant` 선택 가능) |
| `ASK_MODE` | `auto` | 기본 답변 모드 |
| `DART_DATA_ROOT` | `data` | 데이터 폴더 위치 |
| `ASK_RATE_PER_MIN` | `6` | IP당 분당 질문 수 제한 |
| `REPORT_RATE_PER_MIN` | `3` | 분당 새 보고서 생성 제한 (저장본 열기는 제외) |
| `API_ACCESS_TOKEN` | (비움) | 설정하면 API 호출에 `X-Access-Token` 헤더가 필요. 현재 웹 화면은 이 헤더를 보내지 않으므로 **로컬에서는 비워 두세요** |

| 파일 | 내용 |
|---|---|
| `config/settings.yaml` | 청킹 크기, 임베딩, 저장소, 검색 후보 수 |
| `config/agent.yaml` | LLM 모델, 라운드·시간 제한, 회사 별칭 |
| `config/wiki_routes.yaml` | 질문 주제 → 위키 섹션 연결표 |
| `config/lexicon.yaml` | 동의어 사전 (BM25 검색 보강) |
| `config/report_templates.yaml` | 보고서 섹션 구성·지시문 |

## 폴더 구조

```
dart_rag/
  api.py cli.py service.py config.py wiki.py domain.py
  ingestion/   DART API 호출 · XML 파싱 · 청킹
  pipeline/    단계 실행 · 진행 기록(catalog.db)
  embedding/   임베딩 모델
  index/       벡터 인덱스 · BM25 · 문서 저장소
  retrieval/   하이브리드 검색
  routing/     auto 모드 (라우터 · 위키 답변)
  agent/       tool calling 에이전트 · 비교 검증
  report/      보고서 생성
config/  corp_lists/  scripts/  tests/  eval/  docs/
web/index.html      웹 화면 (질문 · 보고서 탭)
data/               저장소 미포함 → Releases 의 data.zip
```

## 선택 구성

기본 실행에는 `requirements.txt` 만 있으면 됩니다. 아래 구성을 쓸 때만 추가로 설치합니다.

| 파일 | 내용 |
|---|---|
| `requirements-optional.txt` | Qdrant, PostgreSQL, BGE-M3 ONNX(torch 없이 CPU) |
| `requirements-ml.txt` | torch 기반 BGE-M3·e5 임베딩, 리랭커 (수 GB) |

- **PostgreSQL + Qdrant**: `pip install -r requirements-optional.txt`, `docker compose up -d` 후 `.env` 에 `DOCSTORE_BACKEND=postgres`, `DATABASE_URL=postgresql://dart:dart@localhost:5432/dart`, `INDEX_BACKEND=qdrant`, `QDRANT_URL=http://localhost:6333` 을 넣고 `python -m dart_rag run --stages index,wiki --force`.
- **BGE-M3 로컬 임베딩**: `scripts\setup_windows.ps1` 실행 (optional 패키지까지 설치) (`.env.local.example` 참고). OpenAI 임베딩으로 만든 `data/` 와 호환되지 않아 `run --stages embed,index` 를 다시 실행해야 합니다.
