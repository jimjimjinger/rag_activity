$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = & (Join-Path $PSScriptRoot 'get-runtime-dir.ps1') -RequiredFile 'paddle-env\Scripts\python.exe'
$parserPython = Join-Path $runtimeDir 'paddle-env\Scripts\python.exe'
$env:PADDLE_PDX_CACHE_HOME = Join-Path $runtimeDir 'paddle-cache'
$env:HF_HOME = Join-Path $runtimeDir 'huggingface-cache'
if (-not (Test-Path -LiteralPath $parserPython)) {
    throw 'Parser environment is missing. See requirements-parser.txt and README.md.'
}
& $parserPython -B (Join-Path $ragRoot 'src\parse_pdf.py') @args
exit $LASTEXITCODE
