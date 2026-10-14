# Install a project-local portable Ollama and the two existing RAG models.
$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $ragRoot '.runtime'
$ollamaExe = Join-Path $runtimeDir 'ollama\ollama.exe'
$ragPython = Join-Path $ragRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $ragPython)) { throw 'Run scripts/setup-rag.ps1 first.' }
if (-not (Test-Path -LiteralPath $ollamaExe)) {
    $downloads = Join-Path $runtimeDir 'downloads'
    New-Item -ItemType Directory -Path $downloads -Force | Out-Null
    $archive = Join-Path $downloads 'ollama-windows-amd64.zip'
    & curl.exe --fail --location --connect-timeout 30 --max-time 900 --output $archive 'https://github.com/ollama/ollama/releases/download/v0.40.4/ollama-windows-amd64.zip'
    if ($LASTEXITCODE -ne 0) { throw 'Official Ollama download failed.' }
    $expected = '7485a1172f8527db4342a63521b9342fc466185ff0855da378cc864ecf6354cf'
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Ollama archive checksum mismatch.' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($archive, (Join-Path $runtimeDir 'ollama'))
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'start-local-llm.ps1')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$pullCode = @'
import httpx
import json
with httpx.Client(base_url="http://127.0.0.1:11435", trust_env=False, timeout=1200) as client:
    for model in ("bge-m3:567m", "qwen3:4b-instruct"):
        print("Preparing " + model, flush=True)
        previous = None
        with client.stream("POST", "/api/pull", json={"model": model, "stream": True}) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                record = json.loads(line)
                if "error" in record:
                    raise RuntimeError(record["error"])
                percent = int(record.get("completed", 0) * 100 / record["total"]) if record.get("total") else None
                state = (record.get("status"), percent // 10 if percent is not None else None)
                if state != previous:
                    print(model, record.get("status"), (str(percent) + "%") if percent is not None else "", flush=True)
                    previous = state
        if record.get("status") != "success":
            raise RuntimeError("Model pull did not finish: " + model)
    print("Ready models:", [item["name"] for item in client.get("/api/tags").json()["models"]], flush=True)
'@
$pullScript = Join-Path $runtimeDir 'prepare-models.py'
[IO.File]::WriteAllText($pullScript, $pullCode, (New-Object System.Text.UTF8Encoding($false)))
& $ragPython -X utf8 -u -B $pullScript
exit $LASTEXITCODE
