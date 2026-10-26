# Run from the rag_activity repository. Installs only the isolated sample environment.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location -LiteralPath $repoRoot
$paddlePython = Join-Path $repoRoot '.runtime\venv-paddle-tables\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $paddlePython)) {
    & '.venv\Scripts\python.exe' -m venv '.runtime\venv-paddle-tables'
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
& $paddlePython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip upgrade failed' }
# Official Windows NVIDIA 50-series Python 3.10 wheel, CUDA 12.9 / cuDNN 9.9.
# https://www.paddleocr.ai/main/en/version3.x/paddlepaddle_installation.html
$paddleWheel = 'https://paddle-qa.bj.bcebos.com/paddle-pipeline/Develop-TagBuild-Training-Windows-Gpu-Cuda12.9-Cudnn9.9-Trt10.5-Mkl-Avx-VS2019-SelfBuiltPypiUse/86d658f56ebf3a5a7b2b33ace48f22d10680d311/paddlepaddle_gpu-3.0.0.dev20250717-cp310-cp310-win_amd64.whl'
& $paddlePython -m pip install $paddleWheel
if ($LASTEXITCODE -ne 0) { throw 'Official GPU wheel installation failed' }
& $paddlePython -m pip install 'paddleocr[doc-parser]==3.3.2' 'paddlex[ocr]==3.3.11'
if ($LASTEXITCODE -ne 0) { throw 'OCR package installation failed' }
& $paddlePython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
# Sample recognition is a separate, explicit command:
# & $paddlePython -X utf8 -u -B PM209/scripts/run_paddle_table_sample.py
