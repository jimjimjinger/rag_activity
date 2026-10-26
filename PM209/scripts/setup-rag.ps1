$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $repoRoot
$ragPython = Join-Path $repoRoot '.runtime\venv-rag\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) {
    $basePython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $basePython)) { $basePython = 'python' }
    & $basePython -m venv '.runtime\venv-rag'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create RAG environment' }
}
& $ragPython -m pip install -r PM209/requirements-rag.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
$tokenizerDirectory = Join-Path $repoRoot '.runtime\pm209-token-length'
New-Item -ItemType Directory -Force -Path $tokenizerDirectory | Out-Null
$tokenizerPath = Join-Path $tokenizerDirectory 'tokenizer.json'
if (-not (Test-Path -LiteralPath $tokenizerPath)) {
    $tokenizerUrl = 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/resolve/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/tokenizer.json'
    Invoke-WebRequest -Uri $tokenizerUrl -OutFile $tokenizerPath
}
$expectedHash = 'def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a'
if ((Get-FileHash -LiteralPath $tokenizerPath -Algorithm SHA256).Hash.ToLower() -ne $expectedHash) {
    throw 'Tokenizer hash mismatch'
}
& (Join-Path $PSScriptRoot 'start-ollama.ps1')
$env:OLLAMA_HOST = '127.0.0.1:11434'
& (Join-Path $repoRoot '.runtime\ollama\ollama.exe') pull 'qwen3-embedding:0.6b'
if ($LASTEXITCODE -ne 0) { throw 'Embedding model download failed' }
Write-Output 'Ready. Run run-pm209-rag.cmd sample for the three-manual sample.'
