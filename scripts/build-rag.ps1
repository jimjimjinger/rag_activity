$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $ragRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'RAG environment is missing. Run scripts/setup-rag.ps1 first.'
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-local-llm.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\build_corpus.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\rag.py') --build
exit $LASTEXITCODE
