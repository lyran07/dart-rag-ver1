# 백엔드 아키텍처

## 1. 계층과 책임

```
진입점        cli.py (python -m dart_rag)     api.py (FastAPI)
                 │                               │
조립          service.py  ── 설정을 읽어 구현체를 고르고 주입 (여기서만 new)
                 │
     ┌───────────┴──────────────────────────┐
오프라인 처리 (pipeline/)                 온라인 질의
  catalog.py   공시 목록·단계 진행 상태     retrieval/query.py      질문 → 필터·확장·섹션힌트
  storage.py   단계별 산출물 경로·원자적 쓰기 retrieval/retriever.py  3채널 검색 → RRF → 리랭크 → 확장
  stages.py    fetch/parse/chunk/embed/     retrieval/reranker.py   bge-reranker (GPU 자동)
               index/wiki                  generation/answerer.py  컨텍스트 예산 → LLM → 출처
  runner.py    이어서 하기·마감 시간·안전 종료
     │                                        │
교체 가능한 어댑터 (인터페이스 + 구현)
  ingestion/  sources.py(DartSource|SampleSource) parser.py chunker.py dart_client.py
  embedding/  Embedder ← BGEM3 | SentenceTransformer(e5) | OpenAICompat | Hashing
  index/      VectorIndex ← InMemoryIndex | QdrantIndex,  DocStore ← PostgresDocStore | SQLiteDocStore,
              BM25Encoder, Tokenizer
  llm/        LLMRouter ← providers.yaml (Groq, Gemini, Upstage, OpenAI, vLLM, Ollama)
  wiki/       WikiBuilder (사람이 읽는 뷰, 검색 경로와 독립)
     │
공통          domain.py (Filing, Section, Chunk, EmbeddingBatch, SearchFilter, Answer)   config.py
```

규칙: 위 계층이 아래 계층을 사용하고, 반대 방향 import 는 없다. 계층 사이에는 `domain.py` 타입만 오간다.

| 바꾸고 싶은 것 | 고칠 곳 |
|---|---|
| LLM 제공자 추가·순서 | `config/providers.yaml` (코드 수정 없음) |
| 임베딩 모델 | `EMBED_PROVIDER` / `settings.yaml` → 모델별로 인덱스 분리 |
| 벡터 DB | `VectorIndex` 구현 하나 추가 + `index/__init__.py:make_index` |
| 본문 DB | `DocStore` 구현 하나 추가 + `make_docstore` (현재 postgres / sqlite) |
| 동의어·섹션 힌트 | `config/lexicon.yaml` |
| 파싱/청킹 규칙 | `ingestion/` → 버전이 바뀌어 그 뒤 단계만 재실행 |

## 2. 저장 구조

```
{DART_DATA_ROOT}/                               GPU 서버에서는 2TB 영구 스토리지
  catalog.db                                    [RDB] filings, stage_runs
  raw/{corp_code}/{rcept_no}.xml.gz             L0 원문 (유일한 원본)
  parsed/{parser_ver}/{corp}/{rcept}.json.gz    L1 섹션 트리
  chunks/{chunk_ver}/{corp}/{rcept}.jsonl.gz    L2 청크
  emb/{model_key}/{corp}/{rcept}.npz            L3 dense(fp16) + sparse, content_hash 기준
  docstore.db                                   [RDB] 청크 본문 (DOCSTORE_BACKEND=sqlite 일 때만)
  index/qdrant/  또는 index/memory/{model_key}/  L4 벡터 인덱스 (필터용 payload 만)
  wiki/                                         위키 md + index.json
```

- 모든 파일은 임시 파일 → `os.replace` 로 쓴다. 강제 종료돼도 반쯤 쓴 파일이 남지 않는다.
- 버전이 경로에 들어간다. 파서·청커·모델을 바꿔도 이전 산출물을 덮어쓰지 않는다.
- 청크 본문은 Qdrant 가 아니라 DocStore 에 둔다. 모델을 바꿔도 본문은 한 벌이다.
- `table`(마크다운) 청크는 임베딩하지 않고 DocStore 에만 둔다. 벡터 수가 약 40% 준다.

### 관계형 DB

| DB | 엔진 | 어디에 | 누가 쓰나 |
|---|---|---|---|
| catalog | SQLite (`catalog.db`) | 영구 스토리지, 데이터와 함께 이동 | 파이프라인 1개 프로세스 |
| docstore | **PostgreSQL** (기본) / SQLite | 서빙 머신 (`docker compose`) | API 워커 여러 개 동시 읽기 + 증분 적재 |

catalog 를 SQLite 로 둔 이유: GPU 대여 서버에서 한 프로세스만 쓰고, 스토리지와 함께 옮겨 다녀야 해서
서버가 필요 없는 파일 DB 가 맞다. (주의: 스토리지가 NFS 같은 네트워크 파일시스템이면 동시 쓰기 금지)

```sql
-- catalog.db (SQLite)
filings(rcept_no PK, corp_code, corp_name, report_nm, report_type, period, rcept_dt,
        is_latest, source, added_at)                 -- 기재정정 시 최신본만 is_latest=1
stage_runs(rcept_no, stage, version, status, n_items, seconds, host, error, updated_at,
           PRIMARY KEY(rcept_no, stage))             -- 이어서 하기·처리 속도 측정

-- docstore (PostgreSQL)
chunks(chunk_id PK, rcept_no, corp_code, corp_name, parent_id, kind, table_index, row_start,
       col_group, section_path, text, meta JSONB, updated_at)
  -- INDEX (parent_id, kind), (rcept_no), (corp_name), GIN(meta jsonb_path_ops)
  -- 공시 1건 = 1 트랜잭션으로 교체 (DELETE rcept_no → INSERT)
```

## 3. 단계와 실행 위치

| 단계 | 입력 → 출력 | 장치 | 실행 위치 |
|---|---|---|---|
| fetch | DART API → raw | 네트워크 | GPU 서버 또는 노트북 |
| parse | raw → parsed | CPU | 〃 |
| chunk | parsed → chunks | CPU | 〃 |
| **embed** | chunks → emb | **GPU 권장** | **GPU 대여 시간** |
| index | chunks + emb → DocStore + Qdrant (BM25 계산 포함) | CPU | 서빙 머신 |
| wiki | parsed + chunks → wiki | CPU | 선택 |

완료 여부 = `stage_runs.status='done' AND version=현재 버전`.

## 4. GPU 대여 1회의 흐름

```
[대여 시작]  bash scripts/gpu_session.sh --until 06:30 --corp-file corp_lists/kospi200.txt
   venv·HF 모델 캐시는 영구 스토리지에 → 두 번째 대여부터 설치·다운로드 생략
   discover → run --stages fetch,parse,chunk,embed --latest-only --until 06:30 --stop-margin 20
   공시 1건 단위로 저장·기록 → 06:10 이후 새 공시 시작 안 함 → 정리 후 종료
   SIGTERM/Ctrl+C → 현재 공시만 마치고 종료
[대여 종료]  status 로 items/s 기록 → 다음 대여 때 몇 개사 처리할지 계산

[서빙 머신]  emb/ 를 복사(또는 스토리지 마운트) → run --stages index,wiki
             → CPU 로 질문만 bge-m3 인코딩 (문서와 같은 모델이면 GPU/CPU 벡터 호환)
```

## 5. 질의 흐름

```
질문
 → QueryAnalyzer   필터(연도·보고서·기간·회사·is_latest), 동의어 확장(BM25용), 섹션 힌트(가산점)
 → Embedder        dense + m3 sparse  (질문 1개, CPU 로 충분)
 → VectorIndex     dense / m3 / bm25 채널별 상위 50 → RRF (결과 0이면 필터 단계적 완화)
 → Reranker        GPU: bge-reranker 후보 40 → 규칙 가산은 약하게 (덧셈)
                   없음: RRF × 규칙 가산 (곱셈, 기존 로직을 설정값으로)
 → small-to-big    DocStore: 같은 섹션 표 KV, KV → 원본 표, (선택) 섹션 본문
 → AnswerGenerator 제공자별 컨텍스트 예산 → LLMRouter (실패 시 다음 제공자) → 출처 [n]
                   LLM 전부 불가 → 추출식 폴백
```

## 6. 기존 코드에서 바뀐 점

| 기존 | 변경 |
|---|---|
| `store.py` 한 파일에 임베더·저장·BM25·검색·점수 | embedding / index / retrieval 로 분리 |
| `HashingEmbedder.embed(is_query=)` TypeError | `embed_documents` / `embed_query` 인터페이스로 통일 |
| sentence-transformers 없으면 해싱으로 조용히 폴백 | 명시적 오류 (다른 모델 인덱스와 섞이는 사고 방지) |
| 섹션 후보를 강제 필터로 | 가산점으로 (라우팅이 틀려도 정답을 놓치지 않음) |
| 보고서마다 BM25 전체 재구축 + jsonl 전체 재저장 | BM25 를 sparse 벡터로, 증분 upsert, flush 1회 |
| `retrieve()` 가 전체 청크를 순회 | DocStore 에서 `parent_id` 인덱스 조회 |
| LLM Groq/OpenAI 고정 | `providers.yaml` 6개 제공자 + 자동 전환 |
| `data/raw` 캐시만, 재실행 시 전부 다시 | 단계별 산출물 + 버전 + catalog 로 이어서 하기 |
| 기재정정 전·후 공시 함께 검색 | `is_latest` 필터 |

## 7. 배포 구성

```
[GPU 대여 서버, 주간/야간]           [서빙 머신, 상시]
 fetch→parse→chunk→embed              docker compose: PostgreSQL + Qdrant
 결과: 2TB 스토리지의 chunks/ emb/    uvicorn dart_rag.api:app (bge-m3 CPU 로 질문만 인코딩)
          │   rsync (수백 건 ≈ 수 GB)       ▲
          └─────────────────────────────────┘  run --stages index,wiki
```
서빙 머신 최소 사양(수백 건 기준 추정): RAM 8GB (bge-m3 CPU ~2.5GB, Postgres ~1GB, Qdrant ~1~2GB), 디스크 30GB.
