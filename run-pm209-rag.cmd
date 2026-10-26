@echo off
setlocal
cd /d "%~dp0"
if not exist ".runtime\venv-rag\Scripts\python.exe" (
  echo Run powershell -ExecutionPolicy Bypass -File PM209\scripts\setup-rag.ps1 first.
  exit /b 1
)
if "%~1"=="" (
  ".runtime\venv-rag\Scripts\python.exe" -X utf8 PM209\scripts\rag_pipeline.py --help
  exit /b
)
if not "%~1"=="chunk" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File PM209\scripts\start-ollama.ps1
  if errorlevel 1 exit /b 1
)
".runtime\venv-rag\Scripts\python.exe" -X utf8 -u PM209\scripts\rag_pipeline.py %*
exit /b %errorlevel%
