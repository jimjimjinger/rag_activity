@echo off
setlocal
cd /d "%~dp0"
if /i "%~1"=="answer" goto langchain
if /i "%~1"=="search" goto langchain
if not exist ".runtime\venv-rag\Scripts\python.exe" (
  echo Run powershell -ExecutionPolicy Bypass -File PM209\scripts\setup-rag.ps1 first.
  exit /b 1
)
if "%~1"=="" (
  echo answer/search: LangChain + Qdrant native dense/BM25. Use --manual ID --query QUESTION.
  echo Optional: --keyword-percent 50 --top-k 5 --no-translate
  echo Other stages below retain the existing preparation pipeline.
  ".runtime\venv-rag\Scripts\python.exe" -X utf8 PM209\scripts\rag_pipeline.py --help
  exit /b
)
if not "%~1"=="chunk" (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File PM209\scripts\start-ollama.ps1
  if errorlevel 1 exit /b 1
)
".runtime\venv-rag\Scripts\python.exe" -X utf8 -u PM209\scripts\rag_pipeline.py %*
exit /b %errorlevel%

:langchain
if not exist ".runtime\venv-rag-lc\Scripts\python.exe" (
  echo Run powershell -ExecutionPolicy Bypass -File PM209\scripts\setup-rag-lc.ps1 first.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File PM209\scripts\start-ollama.ps1
if errorlevel 1 exit /b 1
".runtime\venv-rag-lc\Scripts\python.exe" -X utf8 -u PM209\scripts\langchain_rag.py %*
exit /b %errorlevel%
