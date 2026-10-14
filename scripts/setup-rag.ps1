$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$ragPython = Join-Path $ragRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) {
    $bootstrapPython = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs\Python\Python310\python.exe'
    if (-not (Test-Path -LiteralPath $bootstrapPython)) {
        try {
            $runtimeDir = & (Join-Path $PSScriptRoot 'get-runtime-dir.ps1') -RequiredFile 'python310\python.exe'
            $bootstrapPython = Join-Path $runtimeDir 'python310\python.exe'
        } catch {}
    }
    if (-not (Test-Path -LiteralPath $bootstrapPython)) {
        $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
        if (-not $pythonCommand -or $pythonCommand.Source -like '*\WindowsApps\*') {
            throw 'Install Python 3.10 or later first.'
        }
        $bootstrapPython = $pythonCommand.Source
    }
    & $bootstrapPython -m venv (Join-Path $ragRoot '.venv')
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $ragPython -m pip install -r (Join-Path $ragRoot 'requirements-ui.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $ragPython -m pip check
exit $LASTEXITCODE
