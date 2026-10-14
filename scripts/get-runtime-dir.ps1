# Resolve the existing runtime from both ordinary and packaged-app storage.
param([string]$RequiredFile = 'ollama\ollama.exe')
$localDataDirectory = [Environment]::GetFolderPath('LocalApplicationData')
if ([string]::IsNullOrWhiteSpace($localDataDirectory)) {
    throw 'Windows LocalApplicationData directory could not be resolved.'
}
$runtimeCandidates = @((Join-Path (Split-Path -Parent $PSScriptRoot) '.runtime'), (Join-Path $localDataDirectory 'B2B-RAG'))
$packagePattern = Join-Path $localDataDirectory 'Packages\OpenAI.Codex_*\LocalCache\Local\B2B-RAG'
$runtimeCandidates += @(Get-ChildItem -Path $packagePattern -Directory -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty FullName)
foreach ($candidate in $runtimeCandidates) {
    if (Test-Path -LiteralPath (Join-Path $candidate $RequiredFile) -PathType Leaf) {
        return $candidate
    }
}
throw "B2B-RAG runtime file '$RequiredFile' was not found in: $($runtimeCandidates -join ', ')"
