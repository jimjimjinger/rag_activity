param(
    [ValidateSet('existing','structure','vl')][string]$Profile = 'existing',
    [ValidateSet('cpu','gpu')][string]$Device = 'gpu'
)
$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $ragRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw 'RAG environment is missing. Run scripts/setup-rag.ps1 first.'
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-local-llm.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Profile -eq 'existing') {
    & $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\build_corpus.py')
} else {
    $resultName = if ($Profile -eq 'vl') { 'vl_gpu_8gb' } else { 'structure_gpu_8gb' }
    & $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\build_corpus.py') --parsed (Join-Path $ragRoot "tests\results\$resultName") --output (Join-Path $ragRoot "data\rag\$Profile\corpus.jsonl") --no-corrections
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $pythonExe -X utf8 -B (Join-Path $ragRoot 'src\rag.py') --build --profile $Profile --device $Device
exit $LASTEXITCODE
