# 문서 분석 실험

같은 B2B 매뉴얼 42페이지의 보정 전 결과를 비교합니다. 기존 CPU Structure와 새 8GB GPU Structure·VL 모두 전체 분석을 완료했습니다.

먼저 [분석 보고서](reports/analysis_review.md)를 읽고, [페이지별 비교표](reports/comparison.md)에서 실제 출력을 확인하세요.

## 폴더 구성

```text
tests/
├── README.md                 # 이 안내
├── scripts/                  # 분석 실행·자원 감시·비교·평가 코드
├── results/                  # 완료한 문서 분석 결과
│   ├── structure_cpu/        # L: 기존 CPU 경량 Structure, 42페이지
│   ├── structure_gpu_8gb/    # S: GPU Structure 후보, 42페이지
│   └── vl_gpu_8gb/           # V: PaddleOCR-VL 1.6, 42페이지
├── reports/                  # 분석 보고서·비교표·평가 결과·설치 목록
├── fixtures/                 # 원본 PDF에서 확인한 평가 기준
└── logs/                     # 성공한 GPU 실행의 자원 사용 기록
```

| 코드 | 역할 |
|---|---|
| [run_parser.py](scripts/run_parser.py) | 모델 설정 → PDF 렌더링 → 분석 → Markdown·JSON 저장. 완료 페이지를 검증하고 이어서 실행 |
| [run_guarded_parser.py](scripts/run_guarded_parser.py) | 별도 프로세스에서 GPU 사용량·온도·가용 RAM 감시 |
| [native_cpu_loader.py](scripts/native_cpu_loader.py) | VL 초기 로딩 때 가중치를 CPU에서 준비한 후 GPU에 배치. PaddleX 3.7.2용 |
| [compare.py](scripts/compare.py) | 세 완료 결과에 같은 청킹 규칙을 적용하고 비교표 생성 |
| [evaluate_parsing.py](scripts/evaluate_parsing.py) | 원본 대조 기준으로 본문·제목 번호·표 셀 검사 |
| [evaluate_images.py](scripts/evaluate_images.py) | 저장된 Structure·VL 캡처의 이미지 보존·JSON 지정 문자열·Markdown 전달 평가. 모델 추론 없음 |
| [evaluate_rag.py](scripts/evaluate_rag.py) | 고정 질문으로 기존·Structure·VL의 검색을 비교하고 지정한 질문의 답변 기록 |

각 결과 폴더에는 `page_XX/` 아래 Markdown·이미지가 있고, 새 GPU 결과에는 JSON도 있습니다. `run_summary.json`은 실행 조건과 완료 상태, `corpus.jsonl`은 동일한 청킹 규칙을 적용한 보정 전 결과입니다.

중단된 CPU VL 1페이지, 이전 GPU VL 1페이지·Structure 2페이지, 5GB·6GB 초기 로딩 실패 기록은 삭제했습니다. 완료한 결과와 평가 근거는 보존했습니다. 실패 경위와 해결 과정은 분석 보고서에 요약했습니다.

## 현재 결과와 평가 자료

- [분석 보고서](reports/analysis_review.md): 실행 조건·품질·속도·메모리·실패 사례와 한계
- [페이지별 비교표](reports/comparison.md): 세 구성의 같은 페이지 열기
- [원본 대조 기준](fixtures/reference_checks.json) / [평가 결과](reports/analysis_checks.json)
- [이미지 평가 기준과 결과](reports/image_evaluation.md): Structure·VL의 대표 캡처 7개·지정 문자열 35개, 설정 차이와 한계
- [이미지 정답·좌표](fixtures/image_checks.json) / [이미지 평가 원자료](reports/image_evaluation.json) / [원본 확인 이미지](reports/image_evaluation/source_previews/)
- [자원 사용 요약](reports/resource_summary.json) / [설치 패키지](reports/gpu_packages.txt)
- [RAG 검색 평가](reports/rag_evaluation.md): 고정 질문 10개와 기존 검색 결과
- [RAG 질문 기준](fixtures/rag_questions.json): 동일한 질문 10개와 실시간 점수·작은 캡처 진단 질문
- [VL RAG 검증 보고서](reports/rag_vl_review.md): 세 프로필 검색과 기존·VL 실제 답변, 이미지 지원 범위
- [학습 체크리스트](reports/study_checklist.md): 공부하고 설명할 내용

선택한 본문·정형 표에서는 VL이 더 잘 보존했고 Structure는 빨랐습니다. 작은 화면 캡처의 오인식, 번호·읽기 순서·출력 필터 문제는 남았습니다. 자세한 수치는 보고서를 참고하세요. 청크 수·글자 수는 정확도 점수가 아닙니다.

## 집 GPU PC에서 실행: RTX 5060 계열 / Windows 11

저장소 루트에서 PowerShell로 실행합니다. 필요 시 `scripts/setup-paddle-gpu.ps1`로 전용 환경을 준비합니다. 실제 사용한 구성은 Python 3.10.11, paddlepaddle-gpu 3.3.1 cu129, paddleocr 3.7.0, paddlex 3.7.2입니다.

현재 두 실행 스크립트는 8GB 프로필을 사용합니다. 배치 1, 페이지 폭 1,200px, Paddle 할당자 상한 6,144 MiB입니다. VL은 영역 최대 262,144픽셀·생성 최대 2,048토큰이며 초기 가중치 준비만 CPU에서 수행합니다.

자원 감시는 장치 GPU 사용량 7,168 MiB 초과, 가용 RAM 2,048 MiB 미만, GPU 온도 80도 이상이면 해당 분석을 중단합니다. 로그는 `logs/<결과 폴더 이름>/`에 저장하며 자동 재시도하지 않습니다.

```powershell
# 두 파이프라인을 순서대로 실행. 같은 설정이면 완료 결과를 검증하고 재사용합니다.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-vl-gpu.ps1
```

모델·해상도·패키지·로딩 설정을 바꾸면 `--output .\tests\results\새실험이름`으로 별도 결과 폴더를 지정합니다. 기본 스크립트 실행만으로 기존 완료 결과를 다시 추론할 필요는 없습니다.

## 분석 결과만 다시 비교하기

아래 명령은 모델 추론 없이 저장된 결과로 비교표와 평가 결과를 생성합니다.

```powershell
$runtime = powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\get-runtime-dir.ps1 -RequiredFile 'paddle-gpu-env\Scripts\python.exe'
$parserPython = Join-Path $runtime 'paddle-gpu-env\Scripts\python.exe'
& $parserPython -X utf8 -B .\tests\scripts\compare.py
& $parserPython -X utf8 -B .\tests\scripts\evaluate_parsing.py
# 캡처 평가: 기존 .venv의 Python/Pillow로 실행하며 GPU 모델을 로드하지 않습니다.
& .\.venv\Scripts\python.exe -X utf8 -B .\tests\scripts\evaluate_images.py
```

기존 운영 `outputs/parsing/`과 `data/rag/corpus.jsonl`은 유지합니다. 새 결과는 `data/rag/structure/`·`data/rag/vl/`에 별도 RAG 프로필로 연결했습니다. 운영 RAG의 수동 보정은 새 프로필에 적용하지 않습니다. 파싱 품질과 RAG 검색·답변 품질은 별도로 평가합니다.

## VL RAG 실행과 비교

`run-rag.cmd`는 이제 VL·GPU 기본의 Streamlit 화면을 엽니다. 근거 원본 페이지·추출 이미지·인식된 내용을 확인할 수 있습니다. 기존 터미널은 `run-rag.cmd -Console -Profile vl`입니다. 기존 평가 원자료는 CPU 실험 기록이며 화면 추가와 구분합니다.

저장소 루트에서 `.\scripts\run-rag.ps1 -Profile vl`을 실행하면 VL 코퍼스를 사용합니다. 필요한 경우 `.\scripts\build-rag.ps1 -Profile vl`로 별도 코퍼스와 인덱스를 준비합니다. 기존 프로필은 `existing`, 새 Structure는 `structure`입니다. 동일한 BGE-M3·Qwen을 사용하며 기본 장치는 GPU, `-Device cpu`를 지정하면 CPU입니다. 아래 평가 스크립트는 이전 비교 조건을 유지하기 위해 CPU로 실행합니다.

```powershell
# 저장된 세 인덱스의 고정 질문 10개 검색 비교
& .\.venv\Scripts\python.exe -X utf8 -B .\tests\scripts\evaluate_rag.py

# 기존·VL 답변을 지정한 질문에서 비교. 진단 질문 R1·I1은 검색 지표에서 제외합니다.
& .\.venv\Scripts\python.exe -X utf8 -B .\tests\scripts\evaluate_rag.py --answers Q3 Q8 Q9 Q10 R1 I1
```

결과는 [실행 원자료](reports/rag_vl_checks.json)에 저장합니다. 기존 보정 코퍼스와 새 보정 없는 결과의 차이를 함께 고려하며, 페이지 검색 성공을 답변 정확도로 해석하지 않습니다. 이미지 경로가 있어도 현재 Qwen에는 이미지 파일을 전달하지 않습니다.
