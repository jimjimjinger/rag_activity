$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $taskRepo
$taskPython = Join-Path $taskRepo '.runtime\venv-rag-lc\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    $taskBasePython = Join-Path $taskRepo '.runtime\venv-rag\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $taskBasePython)) { $taskBasePython = 'python' }
    & $taskBasePython -m venv '.runtime\venv-rag-lc'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create LangChain environment' }
}
& $taskPython -m pip install -r PM209/requirements-rag-lc.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $taskPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
Write-Output 'Ready. Existing PM209 chunks/cache and local Ollama models are required; no documents are re-embedded.'
Write-Output 'Build: .runtime\venv-rag-lc\Scripts\python.exe -X utf8 -u PM209\scripts\build_lc_qdrant.py'
