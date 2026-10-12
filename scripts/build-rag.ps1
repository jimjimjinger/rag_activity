$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-local-llm.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python -B (Join-Path $ragRoot 'src\build_corpus.py')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& python -B (Join-Path $ragRoot 'src\rag.py') --build
exit $LASTEXITCODE
