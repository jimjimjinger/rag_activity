# 로컬 PDF 저장 위치

분석할 PDF를 이 폴더에 넣습니다. 원본 PDF와 추출 결과는 Git에 올리지 않습니다.

현재 `src/parse_pdf.py`의 기본 파일명은 `B2B Dashboard Manual_v3.pdf`입니다. 다른 PDF는 아래처럼 지정할 수 있습니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\parse-pdf.ps1 --pdf ".\documents\manual.pdf"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-rag.ps1
```

명령은 프로젝트 루트에서 실행합니다. Python·파싱 환경·Ollama 모델 준비 방법은 [프로젝트 README](../README.md)를 참고하세요.
