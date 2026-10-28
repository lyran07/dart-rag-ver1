<#
  OpenDART 실데이터 수집·적재 (기본: corp_lists\team1.txt 3개 기업, 2025-01-01~오늘, 반기·분기보고서)

  실행:
    powershell -ExecutionPolicy Bypass -File scripts\fetch_dart.ps1
  예:
    ... -File scripts\fetch_dart.ps1 -Limit 6                          # 먼저 6건만 시험
    ... -File scripts\fetch_dart.ps1 -Corp 삼성전자 -Period 2026.06    # 특정 기업·기간만
    ... -File scripts\fetch_dart.ps1 -ReportTypes A001,A002,A003       # 사업보고서 포함

  필요: .env 의 OPENDART_API_KEY. 중간에 끊겨도 다시 실행하면 끝난 단계는 건너뜀.
#>
param(
    [string[]]$Corp = @(),
    [string]$CorpFile = "corp_lists\team1.txt",
    [string[]]$ReportTypes = @("A002", "A003"),
    [string]$From = "20250101",
    [string]$To = (Get-Date -Format "yyyyMMdd"),
    [string]$Period = "",
    [int]$Limit = 0
)
$ErrorActionPreference = "Stop"
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Green }
function Run([string[]]$argv) {
    & $Py @argv
    if ($LASTEXITCODE -ne 0) { throw "실패: python $($argv -join ' ') (exit $LASTEXITCODE)" }
}

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "먼저 scripts\setup_windows.ps1 을 실행하세요 (.venv 없음)" }
if (-not (Select-String -Path .env -Pattern '^OPENDART_API_KEY=\S{10,}' -Quiet)) { throw ".env 에 OPENDART_API_KEY 가 없습니다" }

# -Corp 를 주면 그 기업만, 아니면 -CorpFile 목록
if ($Corp.Count -gt 0) { $target = @("--corp") + $Corp; $label = $Corp -join ', ' }
else { $target = @("--corp-file", $CorpFile); $label = $CorpFile }

Step "수집 대상 등록: $label  ($From ~ $To, 유형 $($ReportTypes -join ','))"
Run (@("-m", "dart_rag", "discover", "--skip-amend", "--from", $From, "--to", $To, "--report-types") + $ReportTypes + $target)

$runArgs = @("-m", "dart_rag", "run", "--latest-only") + $target
if ($Period) { $runArgs += @("--period", $Period) }
if ($Limit -gt 0) { $runArgs += @("--limit", "$Limit") }
Step "다운로드 -> 파싱 -> 청킹 -> 임베딩(CPU, 오래 걸림) -> 인덱스 -> Wiki"
Run $runArgs

Step "상태"
Run @("-m", "dart_rag", "status")

Step "확인 질문"
Run @("-m", "dart_rag", "ask", "SK하이닉스 2026년 반기 연결 매출액은?")
Write-Host "`n완료. 웹: powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -SkipSample" -ForegroundColor Green
