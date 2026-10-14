param(
    [ValidateSet('existing','structure','vl')][string]$Profile = 'vl',
    [ValidateSet('cpu','gpu')][string]$Device = 'gpu',
    [switch]$Console
)
$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ragRoot
$logDirectory = Join-Path $ragRoot 'outputs\rag'
$logPath = Join-Path $logDirectory "startup_$PID.log"
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

    if ($Console) {
        & $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\rag.py') --profile $Profile --device $Device
    } else {
        & $pythonExe -c 'import streamlit, pypdfium2'
        if ($LASTEXITCODE -ne 0) { throw 'UI packages are missing. Run: .\.venv\Scripts\python.exe -m pip install -r .\requirements-ui.txt' }
        $env:B2B_RAG_PROFILE = $Profile
        $env:B2B_RAG_DEVICE = $Device
        & $pythonExe -X utf8 -B -m streamlit run (Join-Path $ragRoot 'src\app.py') --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
    }
    if ($LASTEXITCODE -ne 0) { throw "RAG stopped with exit code $LASTEXITCODE." }
} catch {
    Write-Host "RAG startup/run error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Log: $logPath"
} finally {
    if ($transcribing) { Stop-Transcript | Out-Null }
    Write-Host 'RAG has stopped. You can close this window, or type exit.'
}
