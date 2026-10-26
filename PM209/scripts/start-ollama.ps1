param([int]$Port = 11434)
$ErrorActionPreference = 'Stop'
if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
$repositoryRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runtimeRoot = Join-Path $repositoryRoot '.runtime'
$ollamaExecutable = Join-Path $runtimeRoot 'ollama\ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaExecutable -PathType Leaf)) {
    throw "Project-local Ollama is missing: $ollamaExecutable"
}
$serverUrl = "http://127.0.0.1:$Port"
try {
    $version = Invoke-RestMethod -Uri "$serverUrl/api/version" -TimeoutSec 2
    Write-Output "Ollama is already running: $serverUrl, version $($version.version)"
    Write-Output 'This command did not change the existing server settings.'
    exit 0
} catch {}

$env:OLLAMA_HOST = "127.0.0.1:$Port"
$env:OLLAMA_MODELS = Join-Path $runtimeRoot 'models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
$outputLog = Join-Path $runtimeRoot "pm209-$Port.stdout.log"
$errorLog = Join-Path $runtimeRoot "pm209-$Port.stderr.log"
$serverProcess = Start-Process -FilePath $ollamaExecutable -ArgumentList 'serve' -WindowStyle Hidden -WorkingDirectory $runtimeRoot -RedirectStandardOutput $outputLog -RedirectStandardError $errorLog -PassThru
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if ($serverProcess.HasExited) { throw "Ollama stopped. See $errorLog" }
    try {
        $version = Invoke-RestMethod -Uri "$serverUrl/api/version" -TimeoutSec 2
        Write-Output "Ollama ready: $serverUrl, version $($version.version), PID $($serverProcess.Id)"
        Write-Output "Models: $env:OLLAMA_MODELS"
        Write-Output 'Parallel requests: 1; loaded models: 1; default context: 4096; cloud disabled.'
        Write-Output 'No model inference was started.'
        exit 0
    } catch {}
}
throw "Ollama did not become ready. See $errorLog"
