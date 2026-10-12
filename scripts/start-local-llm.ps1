$ErrorActionPreference = 'Stop'
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '4096'
try {
    $version = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 2
    Write-Output "Ollama already running: $($version.version)"
    exit 0
} catch {}

# Resolve only when starting a server; use the models beside that executable.
$runtimeDir = & (Join-Path $PSScriptRoot 'get-runtime-dir.ps1')
$env:OLLAMA_MODELS = Join-Path $runtimeDir 'models'
$ollamaExe = Join-Path $runtimeDir 'ollama\ollama.exe'
try {
    $executableFile = Get-Item -LiteralPath $ollamaExe -ErrorAction Stop
    if ($executableFile.PSIsContainer) { throw 'Expected an executable file, found a directory.' }
} catch {
    throw "Cannot access Ollama at '$ollamaExe'. User: $([Environment]::UserName). Detail: $($_.Exception.Message)"
}
Write-Output "Starting Ollama: $($executableFile.FullName)"
Start-Process -FilePath $executableFile.FullName -ArgumentList 'serve' -WindowStyle Hidden -WorkingDirectory $runtimeDir -RedirectStandardOutput (Join-Path $runtimeDir 'server.stdout.log') -RedirectStandardError (Join-Path $runtimeDir 'server.stderr.log') | Out-Null
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    try {
        $version = Invoke-RestMethod -Uri 'http://127.0.0.1:11435/api/version' -TimeoutSec 2
        Write-Output "Local Ollama ready: $($version.version), http://127.0.0.1:11435"
        exit 0
    } catch {}
}
throw "Ollama did not start. See $runtimeDir\server.stderr.log"
