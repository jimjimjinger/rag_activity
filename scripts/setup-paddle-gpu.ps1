# Windows / RTX 50-series: install the CUDA 12.9 Paddle build in a separate venv.
param([string]$PythonExe = 'python')
$ErrorActionPreference = 'Stop'
$ragRoot = Split-Path -Parent $PSScriptRoot

$smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if (-not $smi) {
    throw 'NVIDIA GPU/driver not found. Run this script on the home GPU computer after installing the NVIDIA driver.'
}
& $smi.Source --query-gpu=index,name,memory.total,driver_version --format=csv
if ($LASTEXITCODE -ne 0) { throw 'nvidia-smi failed. Check the NVIDIA driver.' }

& $PythonExe -c "import struct,sys; assert sys.version_info[:2] == (3,10) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.10 for this project'; print(sys.executable)"
if ($LASTEXITCODE -ne 0) { throw 'Select 64-bit Python 3.10 with -PythonExe.' }

$runtimeDir = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'B2B-RAG'
$gpuEnvironment = Join-Path $runtimeDir 'paddle-gpu-env'
$gpuPython = Join-Path $gpuEnvironment 'Scripts\python.exe'
if (-not (Test-Path -LiteralPath $gpuPython -PathType Leaf)) {
    & $PythonExe -m venv $gpuEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'GPU virtual environment creation failed.' }
}

function Invoke-GpuPython([string[]]$PythonArguments) {
    & $gpuPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "GPU setup failed (exit $LASTEXITCODE). See the error above." }
}

# Both distributions provide 'import paddle'; never install the CPU one here.
Invoke-GpuPython @('-c', "from importlib.metadata import distributions; assert not any(d.metadata['Name'].lower() == 'paddlepaddle' for d in distributions()), 'CPU Paddle is installed in the GPU environment; use a separate clean environment'")
Invoke-GpuPython @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-GpuPython @('-m', 'pip', 'install', 'paddlepaddle-gpu==3.3.1', '--index-url', 'https://www.paddlepaddle.org.cn/packages/stable/cu129/')

# Reuse the known parser dependencies while excluding the CPU framework pin.
$gpuDependencies = Join-Path $runtimeDir 'requirements-parser-gpu-deps.txt'
Get-Content -LiteralPath (Join-Path $ragRoot 'requirements-parser-lock.txt') -Encoding UTF8 |
    Where-Object { $_ -notmatch '^paddlepaddle==' } |
    Set-Content -LiteralPath $gpuDependencies -Encoding UTF8
Invoke-GpuPython @('-m', 'pip', 'install', '-r', $gpuDependencies, 'paddleocr[doc-parser]==3.7.0')
Invoke-GpuPython @('-m', 'pip', 'check')

# Run an actual GPU operation; importing the GPU wheel alone is insufficient.
Invoke-GpuPython @('-c', "import paddle; assert paddle.is_compiled_with_cuda(), 'CPU build loaded'; assert paddle.device.cuda.device_count() > 0, 'No CUDA GPU detected'; paddle.set_device('gpu:0'); print('GPU:', paddle.device.cuda.get_device_name(0)); print('CUDA build:', paddle.version.cuda()); paddle.utils.run_check(); x=paddle.ones([16,16]); print('GPU matrix result:', float(paddle.matmul(x,x).sum())); print('Device:', x.place)")
Write-Host "GPU environment ready: $gpuPython"
Write-Host 'Next: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1 --pages 9 37'
Write-Host 'Then: powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-vl-gpu.ps1 --pages 9 37'
