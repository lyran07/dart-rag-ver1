<#
  Windows 노트북 로컬 실행 (bge-m3 ONNX, DB 없이)

  실행 (프로젝트 폴더 어디서든):
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1

  옵션:
    -Python <경로>   가상환경을 만들 Python (기본: py -3.11 → 없으면 python)
    -SkipSample      샘플 적재 건너뛰기 (이미 적재했을 때)
    -NoServer        마지막에 웹 서버를 띄우지 않기
    -Port 8080       웹 서버 포트

  하는 일: .venv 생성 → 패키지 설치 → .env 준비 → 샘플 적재(모델 자동 다운로드) → 웹 서버 실행
  두 번째부터는 이미 된 단계는 빠르게 지나간다.
#>
param(
    [string]$Python = "",
    [switch]$SkipSample,
    [switch]$NoServer,
    [int]$Port = 8080
)
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Green }
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "실패: $exe $($argv -join ' ') (exit $LASTEXITCODE)" }
}

# 스크립트 위치 기준으로 프로젝트 루트로 이동 → 저장 경로와 무관하게 동작
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "프로젝트 폴더: $Root"

# 한글 설정·출력 깨짐 방지
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

# 1) 가상환경
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Step "가상환경(.venv) 생성"
    if ($Python) {
        Run $Python @("-m", "venv", ".venv")
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        Run "py" @("-3.11", "-m", "venv", ".venv")
    } else {
        Run "python" @("-m", "venv", ".venv")
    }
} else {
    Step "가상환경(.venv) 이미 있음"
}

# 2) 패키지 (onnxruntime·tokenizers 는 requirements-optional.txt)
Step "패키지 설치"
Run $VenvPy @("-m", "pip", "install", "--upgrade", "pip", "-q")
Run $VenvPy @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-optional.txt", "-q")

# 3) .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.local.example" ".env"
    Step ".env 생성 (.env.local.example 복사) - LLM 답변을 원하면 .env 의 GROQ_API_KEY 를 채우세요"
} else {
    Step ".env 이미 있음 (EMBED_PROVIDER=bge-m3-onnx, DOCSTORE_BACKEND=sqlite, INDEX_BACKEND=memory 인지 확인)"
}

# 4) 샘플 적재 (첫 실행 때 bge-m3 ONNX 570MB 다운로드)
if (-not $SkipSample) {
    Step "샘플 적재 (첫 실행이면 모델 다운로드로 몇 분 걸림)"
    Run $VenvPy @("-m", "dart_rag", "sample")
    Step "검색 확인"
    Run $VenvPy @("-m", "dart_rag", "search", "2023년 배당금은?")
}

# 5) 웹 서버
if (-not $NoServer) {
    Step "웹 서버: http://localhost:$Port  (/health 의 embedder 가 bge-m3-onnx-int8 이면 정상, 종료는 Ctrl+C)"
    Run $VenvPy @("-m", "uvicorn", "dart_rag.api:app", "--port", "$Port")
}
