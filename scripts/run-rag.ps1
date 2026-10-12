$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ragRoot
$logDirectory = Join-Path $ragRoot 'outputs\rag'
$logPath = Join-Path $logDirectory 'startup.log'
$transcribing = $false
try {
    New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
    Start-Transcript -LiteralPath $logPath -Force | Out-Null
    $transcribing = $true

    $pythonExe = Join-Path $ragRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw 'RAG environment is missing. Run: powershell -ExecutionPolicy Bypass -File .\scripts\setup-rag.ps1'
    }
    Write-Host "Python: $pythonExe"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-local-llm.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Ollama startup failed (exit $LASTEXITCODE)." }

    & $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\rag.py')
    if ($LASTEXITCODE -ne 0) { throw "RAG stopped with exit code $LASTEXITCODE." }
} catch {
    Write-Host "RAG startup/run error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Log: $logPath"
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    Write-Host 'RAG has stopped. You can close this window, or type exit.'
}
