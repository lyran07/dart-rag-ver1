#!/usr/bin/env bash
# GPU 대여 1회(주간 또는 야간) 실행 스크립트.
#
#   bash scripts/gpu_session.sh --until 06:30 --corp-file corp_lists/kospi200.txt
#
# 전제
#   - STORAGE: 2TB 영구 스토리지 마운트 경로 (다음 대여 때도 남아 있어야 함)
#   - 300GiB 로컬 디스크는 임시 공간으로만 사용 (반납 시 사라져도 되는 것만)
# 하는 일
#   1) venv·모델 캐시를 영구 스토리지에 둬서 매번 재설치/재다운로드하지 않음
#   2) 새 공시 discover (선택) → fetch,parse,chunk,embed 실행 (마감 20분 전 자동 정지)
#   3) 진행 상황·처리 속도 기록 (catalog.db) → 다음 대여 때 이어서
#   index 단계는 서빙 머신(노트북/CPU 서버)에서:  python -m dart_rag run --stages index,wiki
set -euo pipefail

STORAGE="${STORAGE:-/mnt/storage}"          # ← 대여처 스토리지 마운트 경로로 변경
SCRATCH="${SCRATCH:-/workspace/scratch}"
UNTIL=""; CORP_FILE=""; FROM="20230101"; TO="20241231"; DISCOVER=1; STAGES="fetch,parse,chunk,embed"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --until) UNTIL="$2"; shift 2;;
    --corp-file) CORP_FILE="$2"; shift 2;;
    --from) FROM="$2"; shift 2;;
    --to) TO="$2"; shift 2;;
    --no-discover) DISCOVER=0; shift;;
    --stages) STAGES="$2"; shift 2;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

export DART_DATA_ROOT="$STORAGE/dart"
export DART_SCRATCH="$SCRATCH"
export HF_HOME="$STORAGE/hf_cache"            # bge-m3 / reranker 모델 (약 4.5GB) 1회만 다운로드
export PIP_CACHE_DIR="$STORAGE/pip_cache"
export EMBED_PROVIDER="${EMBED_PROVIDER:-bge-m3}"
export EMBED_DEVICE="${EMBED_DEVICE:-cuda}"
export DART_SETTINGS="${DART_SETTINGS:-config/settings.gpu.yaml}"
VENV="$STORAGE/venv-dart-rag"
mkdir -p "$DART_DATA_ROOT" "$SCRATCH" "$HF_HOME"

# 1) 환경 (최초 1회만 오래 걸림)
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip
  "$VENV/bin/pip" install -r requirements.txt -r requirements-ml.txt
fi
PY="$VENV/bin/python"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv || true
df -h "$STORAGE" "$SCRATCH" || true

# 2) 수집 대상 등록
if [[ $DISCOVER -eq 1 && -n "$CORP_FILE" ]]; then
  "$PY" -m dart_rag discover --corp-file "$CORP_FILE" --from "$FROM" --to "$TO"
fi

# 3) 처리 (마감 시각이 있으면 20분 전부터 새 공시를 시작하지 않음)
LOG="$DART_DATA_ROOT/logs/session-$(date +%Y%m%d-%H%M).log"
mkdir -p "$(dirname "$LOG")"
ARGS=(--stages "$STAGES" --latest-only)
[[ -n "$UNTIL" ]] && ARGS+=(--until "$UNTIL" --stop-margin 20)
[[ -n "$CORP_FILE" ]] && ARGS+=(--corp-file "$CORP_FILE")
"$PY" -m dart_rag run "${ARGS[@]}" 2>&1 | tee "$LOG"

# 4) 요약 (처리 속도 items/s 가 다음 대여 계획의 근거)
"$PY" -m dart_rag status | tee -a "$LOG"
echo "완료. 영구 저장소: $DART_DATA_ROOT  (로그: $LOG)"
