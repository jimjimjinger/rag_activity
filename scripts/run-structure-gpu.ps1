# Run the StructureV3 quality candidate without replacing the original results.
$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = & (Join-Path $PSScriptRoot 'get-runtime-dir.ps1') -RequiredFile 'paddle-gpu-env\Scripts\python.exe'
$gpuPython = Join-Path $runtimeDir 'paddle-gpu-env\Scripts\python.exe'
$env:PADDLE_PDX_CACHE_HOME = Join-Path $runtimeDir 'paddle-cache'
$env:HF_HOME = Join-Path $runtimeDir 'huggingface-cache'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'
& $gpuPython -X utf8 -u -B (Join-Path $ragRoot 'tests\scripts\run_guarded_parser.py') --pipeline structure --device gpu:0 --memory-profile 8gb --output (Join-Path $ragRoot 'tests\results\structure_gpu_8gb') @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $gpuPython -X utf8 -B (Join-Path $ragRoot 'tests\scripts\compare.py')
exit $LASTEXITCODE
