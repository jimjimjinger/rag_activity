# 문서 분석 비교

이 폴더는 자동 코드 테스트 대신 **경량 Structure·고정밀 후보 Structure·PaddleOCR-VL의 보정 전 분석 결과**를 보관합니다.

기존 Structure 42페이지와 회사 CPU에서 분석한 VL 1페이지 표본을 보관합니다. CPU 전체 분석은 사용자 요청으로 중단했습니다.
집 GPU 분석의 진행 상태와 완료 페이지 수는 [comparison.md](comparison.md)에서 확인합니다.

| 경로 | 내용 |
|---|---|
| `pp_structurev3/page_XX/page_XX.md` | 기존 Paddle 분석 결과의 원본 복사. 수동 보정 없음 |
| `pp_structurev3/run_summary.json` | 기존 분석 당시 실행 기록. 현재 코드로 재실행한 결과가 아님 |
| `pp_structurev3_gpu/` | 집 GPU에서 새로 실행할 Structure 고정밀 후보의 Markdown·JSON·실행 설정 |
| `paddleocr_vl/page_XX/page_XX.md` | PaddleOCR-VL-1.6의 새 분석 결과 |
| `paddleocr_vl/page_XX/page_XX.json` | VL의 영역·읽기 순서 등 구조화 결과 |
| `paddleocr_vl/run_summary.json` | 모델 설정, 원본 해시, 완료 페이지와 실행 상태 |
| `paddleocr_vl_gpu/` | 집 GPU PC에서 실행한 VL 결과. CPU 표본과 분리 |
| 각 폴더의 `corpus.jsonl` | 같은 청킹 규칙을 적용한 결과. 원본 대조 보정은 적용하지 않음 |
| `comparison.md` | 페이지별 결과 링크와 비교 지표 |
| `run_parser.py` | Structure·VL 공통 분석 실행. 완료 페이지를 검사하고 이어서 처리 |
| `compare.py` | 세 구성의 비교표와 보정 없는 청킹 결과 생성 |

현재 RAG가 사용하는 `data/rag/` 및 `outputs/parsing/`과 별도로 관리합니다.
운영 RAG의 `markdown_corrections.json`은 이 실험에 적용하지 않습니다.
분석 모델이 문서 구조를 만드는 단계와, `src/build_corpus.py`가 검색용 청크를 만드는 단계를 구분해서 확인합니다.
기존 Structure의 경량 모델 설정과 VL의 모델 구성이 다르므로, 결과 차이를 VL 구조 하나의 효과로 단정하지 않습니다.
VL은 같은 폭 2,000픽셀로 메모리에서 렌더링하며, 중간 페이지 PNG 파일은 만들지 않습니다.

## 실행

RAG 폴더의 PowerShell에서 실행합니다. 기존 Paddle 전용 환경과 모델 캐시를 사용합니다.

```powershell
$ragRuntimeDir = powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\get-runtime-dir.ps1 -RequiredFile 'paddle-env\Scripts\python.exe'
$ragParserPython = Join-Path $ragRuntimeDir 'paddle-env\Scripts\python.exe'
$env:PADDLE_PDX_CACHE_HOME = Join-Path $ragRuntimeDir 'paddle-cache'
$env:HF_HOME = Join-Path $ragRuntimeDir 'huggingface-cache'
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = 'True'
& $ragParserPython -X utf8 -u -B .\tests\run_parser.py --pipeline vl
python -X utf8 -B .\tests\compare.py
```

VL 모델은 처음 실행할 때 내려받습니다. 모델 가중치는 프로젝트 밖 캐시에 저장합니다.
문서 분석 결과는 사람이 검토할 실험 자료입니다. 글자 수·제목 수·청크 수는 정확도 점수가 아닙니다.

## 집 GPU PC에서 실행: RTX 5060 계열 / Windows 11

RTX 50 계열은 Blackwell이므로 **CUDA 12.9용 PaddlePaddle GPU 패키지**를 사용합니다.
구성은 Python 3.10 64비트, `paddlepaddle-gpu==3.3.1`(cu129), `paddleocr==3.7.0`입니다.
프로젝트의 기존 의존성 잠금 목록에서 CPU Paddle 항목을 제외하고 설치합니다.

1. 집 PC에 Python 3.10 64비트와 최신 NVIDIA 드라이버를 설치합니다.
2. 이 프로젝트 폴더를 가져온 뒤 PowerShell에서 아래 명령을 실행합니다.

```powershell
# GPU 이름·실제 VRAM·드라이버 확인
nvidia-smi

# 전용 환경 설치 + 실제 GPU 연산 확인
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-paddle-gpu.ps1

# 먼저 설명 페이지와 표 페이지로 두 파이프라인의 실제 작동 확인
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1 --pages 9 37
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-vl-gpu.ps1 --pages 9 37

# 정상 저장됐으면 각각 전체 분석. 완료된 표본 페이지는 재사용
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-vl-gpu.ps1
```

Python이 여러 개라면 설치 명령에 `-PythonExe 'C:\경로\Python310\python.exe'`를 붙입니다.
설치는 `%LOCALAPPDATA%\B2B-RAG\paddle-gpu-env`에 이루어지며 기존 CPU 환경과 분리합니다.
설치 스크립트는 모델 분석을 시작하지 않습니다. 각 GPU 실행 명령을 처음 실행할 때 해당 모델들을 내려받고 분석합니다.

**CUDA는 사용하지만 전체 CUDA Toolkit을 수동 설치할 필요는 없습니다.** GPU 패키지 설치 시 필요한 런타임을 설치합니다.
NVIDIA 드라이버는 별도로 필요합니다. `nvidia-smi`의 CUDA Version은 드라이버가 지원하는 최대 버전이며 Toolkit 설치 여부를 뜻하지 않습니다.
Blackwell 공식 안내는 CUDA 12.9 이상을 지원하는 드라이버를 요구합니다. 최신 드라이버에서 13.x로 표시되는 것도 가능합니다.

Structure GPU 결과는 `tests/pp_structurev3_gpu/`, VL GPU 결과는 `tests/paddleocr_vl_gpu/`에 저장됩니다.
기존 Structure 42페이지 및 CPU VL 1페이지 표본은 유지합니다. 운영 RAG의 코퍼스·인덱스에는 반영하지 않습니다.
같은 명령을 다시 실행하면 같은 설정으로 저장된 완료 페이지를 확인하고 이어서 처리합니다. Ctrl+C로 중단할 수 있습니다.
표본 실행의 상태는 `partial`, 42페이지 전체 완료는 `complete`입니다. 모델·패키지 버전이 달라진 결과는 다른 출력 폴더에 저장해야 합니다.
`compare.py`는 세 구성을 나란히 표시하며, VL은 GPU 실행 기록이 있으면 GPU 결과를 선택합니다.
분석이 끝난 뒤 [comparison.md](comparison.md)에서 세 구성의 같은 페이지를 열어 비교합니다.

현재 회사 노트북에서는 설치 스크립트와 실행 옵션을 준비한 상태이며, 실제 GPU 설치·실행 검증은 집 PC에서 필요합니다.

공식 안내: [Blackwell용 PaddleOCR-VL](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL-NVIDIA-Blackwell.html),
[Paddle 설치와 CUDA 런타임](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html).

## Structure GPU 후보를 고른 근거

**GPU 자체가 정확도를 올리는 것은 아닙니다.** GPU에서 실행할 여유를 활용해 배치·OCR·표 모델과 OCR 입력 크기를 바꾼 후보입니다.
현재 설치된 PaddleOCR 3.7.0의 공식 Structure 구성·지원 모델을 기준으로 선택했습니다. RTX 5060에서의 최적 조합이나 실제 성능 향상은 아직 검증하지 않았습니다.

| 항목 | L: 기존 CPU 경량 | S: GPU 고정밀 후보 | 선택 근거 |
|---|---|---|---|
| 배치 분석 | `PP-DocLayout-S` | `PP-DocLayout_plus-L` | 공식 Structure 구성의 큰 배치 모델로 영역 누락·병합 차이를 비교 |
| 문서 블록 검출 | 꺼짐 | `PP-DocBlockLayout` | 공식 Structure 구성의 추가 블록 구분도 사용해 영역 그룹·읽기 순서 비교 |
| 글자 검출 | `PP-OCRv5_mobile_det` | `PP-OCRv5_server_det` | 공식 평가에서 더 높은 검출 성능을 보고한 서버 모델로 작은 글자 누락 비교 |
| 글자 인식 | `en_PP-OCRv5_mobile_rec` | `PP-OCRv5_server_rec` | 서버 인식기를 후보로 시험. 영어 전용 모델보다 반드시 좋다는 보장은 없음 |
| 유선표 구조 | `SLANet_plus` | `SLANeXt_wired` | 표 구조 인식 강화 후보 |
| 무선표 구조 | 기존 기본값 `SLANet_plus` | `SLANeXt_wireless` | 선이 없는 표도 별도 전용 모델로 시험 |
| OCR 입력 최대 변 | 1,280 | 2,000 | 같은 폭 2,000 페이지 입력에서 OCR 내부 축소로 잃는 작은 글자 비교 |

표 종류 분류와 유선·무선 셀 검출 등 보조 모델도 함께 실행합니다. 위 목록은 변경한 주요 모델이며 전체 모델 수가 아닙니다.
방향·왜곡 보정, 수식·차트·도장 인식은 비교 목적과 현재 매뉴얼에 맞춰 끕니다. 새 실행의 배치는 1로 제한합니다.
이 설정은 VRAM 부담을 줄이기 위한 출발점이며, 집 PC에서 실제 VRAM과 GPU 오류 여부를 확인해야 합니다.

`PP-OCRv5_server_rec`의 지원 언어는 중·영·일 중심이며 한글용 인식기가 아닙니다.
한글이 필요한 문서는 `korean_PP-OCRv5_mobile_rec`처럼 한글·영어 지원 인식기를 별도 후보로 검토합니다.
예를 들어 인식기만 바꾼 실험은 아래처럼 **별도 출력 폴더**를 지정합니다. 기본 세 구성 비교표에는 이 추가 실험을 자동 합치지 않습니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1 --pages 1 9 --text-recognition-model korean_PP-OCRv5_mobile_rec --output .\tests\pp_structurev3_gpu_korean
```

L과 S의 차이는 모델 크기·언어 지원·OCR 입력 제한을 함께 바꾼 **구성 전체의 차이**입니다.
어느 한 모델의 효과를 따로 알고 싶으면 인식기 등 한 조건만 바꾸고 별도 결과로 비교합니다.
S와 V는 같은 집 GPU·동일 PDF·폭 2,000 입력·수동 보정 없음으로 비교합니다. L의 과거 CPU 시간과 새 GPU 시간을 직접 비교하지 않습니다.

선정 근거: [Structure 공식 구성·모델 목록](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html),
[OCR 인식기와 지원 언어](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html),
[표 구조 모델](https://www.paddleocr.ai/latest/en/version3.x/module_usage/table_structure_recognition.html).

## 집에서 실행한 뒤 확인할 것

1. GPU 설치 확인에서 GPU 이름과 `Device: Place(gpu:0)`에 해당하는 출력이 나오는지 봅니다.
2. 두 GPU 결과 폴더에 9·37페이지의 Markdown·JSON이 저장되는지 봅니다.
3. 전체 실행 후 두 `run_summary.json`의 `status`가 `complete`이고, 결과가 1~42페이지 모두 있는지 확인합니다.
4. [comparison.md](comparison.md)에서 같은 페이지의 경량 Structure·고정밀 후보 Structure·VL 결과를 엽니다. 원본 PDF를 함께 보고 아래 항목을 비교합니다.

| 페이지 | 확인할 내용 | L 경량 | S 고정밀 후보 | V VL |
|---|---|---|---|---|
| 1 | 한글 메모·날짜의 누락·오인식과 지원 언어 | 확인 후 기록 | 확인 후 기록 | 확인 후 기록 |
| 9 | 1·2·3번 제목과 설명 연결, Suggested Questions 경계 | 확인 후 기록 | 확인 후 기록 | 확인 후 기록 |
| 11·22 | Page Type 제목과 설명 문장의 연결 | 확인 후 기록 | 확인 후 기록 | 확인 후 기록 |
| 20 | 번호가 따로 찍힌 제목·설명·읽기 순서 | 확인 후 기록 | 확인 후 기록 | 확인 후 기록 |
| 23 | 강조된 Download 문장과 항목 경계 | 확인 후 기록 | 확인 후 기록 | 확인 후 기록 |
| 37·39 | 표의 헤더·행·열·값 연결 | 확인 후 기록 | 확인 후 기록 | 확인 후 기록 |

관찰은 `누락 1건: 3번 제목이 본문에 붙음`처럼 위치와 실제 예시를 적습니다.
현재 VL 기본 설정도 그림으로 분류된 영역의 내부 글자를 전부 OCR하지는 않습니다. JSON의 `model_settings.use_ocr_for_image_block`을 함께 확인하세요.
이미지 파일이 저장되었다는 사실, 이미지 속 텍스트가 추출되었다는 사실, 답변 모델이 이미지를 이해한다는 사실은 각각 다릅니다.

**분석 비교와 RAG 비교는 별도 실험입니다.** 지금 VL 결과는 운영 RAG에 연결하지 않았습니다.
따라서 이 비교만으로 VL이 RAG 답변을 개선했다고 결론 내리지 않습니다.
추후 그 효과를 보려면 두 분석 결과에 같은 청킹 규칙·임베딩 모델·질문·답변 모델을 적용해 비교해야 합니다.

## RAG 검색 평가: 고정 질문 10개

[원본 PDF](../documents/B2B%20Dashboard%20Manual_v3.pdf)의 해당 페이지를 화면으로 대조해 필요한 근거와 답변 요점을 정했습니다.
아래 질문은 학습용 소규모 평가 목록이며 실제 사용자 전체를 대표하는 검증 자료는 아닙니다.

| ID | 질문 | 필요한 페이지 | 답변에 들어갈 요점 |
|---|---|---|---|
| Q1 | AX Agent에서 개선 가이드를 보려면 어떤 카테고리를 선택해야 하나요? | 9 | How to Fix |
| Q2 | AX Agent에서 부서별로 자주 사용하는 질문은 어디에서 볼 수 있나요? | 9 | Suggested Questions |
| Q3 | Spec Monitoring에서 이슈 필터를 초기화하려면 어떻게 하나요? | 20 | Clear filter 클릭 |
| Q4 | Spec Monitoring에서 기준 스펙과 수집된 스펙 값을 비교하려면 어떻게 하나요? | 20 | 목록에서 페이지·모델을 찾고 Detail 클릭 |
| Q5 | 모니터링 항목별 상세 조건과 점수는 대시보드 어디에서 어떻게 확인하나요? | 22 | REFERENCE > Monitoring Item > Business 또는 Shared, 항목 화살표로 상세 조건·점수 펼치기 |
| Q6 | REFERENCE의 DICT 데이터를 CSV 파일로 내려받으려면 어떻게 하나요? | 23 | Download로 선택 데이터 CSV 내보내기 |
| Q7 | Only Monitoring 항목도 총점 계산에 포함되나요? | 35 | 상태·이슈 확인용이며 총점 제외 |
| Q8 | SEO 모니터링에서 PDP의 Title Length 최대 배점은 몇 점인가요? | 37 | 20pt |
| Q9 | SEO 모니터링에서 H1 Tag는 점수를 받는 항목인가요? | 37 | M: Monitoring only, 점수 미반영 |
| Q10 | 모니터링 항목의 상세 조건을 여는 방법과 SEO PDP Title Length의 최대 배점을 함께 알려주세요. | 22, 37 | 상세 펼치기 방법과 20pt를 모두 설명 |

별도 거절 확인 질문: **지금 India HVAC의 실제 모니터링 점수는 몇 점인가요?**
현재 시스템은 실시간 대시보드를 조회하지 않으므로 현재 점수를 만들어 답하지 않아야 합니다.
이 질문은 조회 경로를 묻는 질문과 구분하며, 위 10개 질문의 검색 지표에는 포함하지 않습니다.

### 현재 측정한 검색 결과

2026-10-14, 현재 운영 코퍼스 73개 청크·BGE-M3·FAISS에서 검색만 실행했습니다. Qwen 답변은 평가하지 않았습니다.
top-5를 한 번 검색하고 그 결과의 앞 1·3·5개를 사용했습니다. 현재 정확 검색의 순위는 k에 따라 바뀌지 않습니다.

| ID | top-5에서 나온 페이지: 순위 순서, 중복 유지 | 필요한 페이지를 모두 찾았나? |
|---|---|---|
| Q1 | 9, 9, 9, 4, 29 | 예 |
| Q2 | 9, 9, 9, 4, 38 | 예 |
| Q3 | 20, 38, 39, 37, 5 | 예 |
| Q4 | 38, 35, 37, 35, 39 | 아니오: 20 누락 |
| Q5 | 22, 21, 22, 21, 10 | 예 |
| Q6 | 23, 31, 23, 24, 24 | 예 |
| Q7 | 35, 21, 21, 35, 38 | 예 |
| Q8 | 37, 21, 39, 38, 21 | 예 |
| Q9 | 37, 39, 38, 35, 35 | 예 |
| Q10 | 37, 21, 38, 21, 39 | 아니오: 22 누락 |

| 지표 | k=1 | k=3 | k=5 |
|---|---|---|---|
| 필요한 페이지를 모두 찾은 질문 | 8/10 | 8/10 | 8/10 |
| 페이지 단위 Recall@k의 질문별 평균 | 0.85 | 0.85 | 0.85 |

페이지 단위 Recall은 `찾은 필수 페이지 수 / 필수 페이지 수`입니다. Q10은 2개 중 1개로 0.5점입니다.
같은 페이지의 여러 청크는 하나의 페이지로 셉니다. 같은 페이지라도 답변에 필요한 다른 항목일 수 있어, 이 지표만으로 청크 내용이 충분하다고 판단할 수 없습니다.
**위 숫자는 답변 정확도가 아닙니다.** k=3이 최적이라는 근거도 아닙니다.

Q4를 원문 용어가 포함된 `Spec Monitoring에서 reference specification과 LG.com에서 수집한 값을 비교하는 Detail은 어디에서 여나요?`로 바꾸면 top-1이 `p20-s3`이었습니다.
이는 한·영 표현과 질문 문구에 따른 검색 민감도를 보여주는 진단 사례입니다. 질문을 바꾼 뒤 기존 평가 성적에 합산하지 않았고, 검색 로직도 수정하지 않았습니다.
Q10은 여러 근거를 조합하는 질문의 실패 사례입니다. 추가 검색, 키워드 검색 결합, 재순위화 등은 이후 비교할 후보입니다.

재현 기준: 임베딩 모델 digest `7907646426070047a77226ac3e684fbbe8410524f7b4a74d02837e43f2146bab`,
코퍼스 SHA-256 `93a946cba290734d16b0a56f0126f31510018ba0d6e6ce29d7f769724be111e1`.
파일·모델이 바뀌면 이 결과는 이전 버전의 기록으로 읽어야 합니다.

### 직접 재현하고 답변까지 확인하기

새 PC의 RAG 환경·Ollama·모델 준비는 [프로젝트 README](../README.md)의 새 PC 준비를 따릅니다. GPU VL 설치 환경과는 별개입니다.
Ollama가 실행 중인 상태에서 아래와 같이 같은 질문을 비교합니다. Q1~Q10에도 동일하게 적용할 수 있습니다.

```powershell
$ragQuestion = 'Spec Monitoring에서 기준 스펙과 수집된 스펙 값을 비교하려면 어떻게 하나요?'
foreach ($ragTopK in 1, 3, 5) {
    .\.venv\Scripts\python.exe -X utf8 -B .\src\rag.py $ragQuestion --search-only --top-k $ragTopK
}

# --search-only를 빼면 검색 근거로 Qwen 답변까지 생성
.\.venv\Scripts\python.exe -X utf8 -B .\src\rag.py $ragQuestion --top-k 3
```

검색 평가는 필요한 **청크 본문**이 들어왔는지, 답변 평가는 **질문에 답했는지·문서 근거를 지켰는지·출처가 맞는지**로 나눕니다.
답변 생성 시간도 함께 비교할 수 있습니다. 현재 표에는 답변 평가와 시간 비교 결과를 포함하지 않았습니다.
검색한 근거가 있어도 입력 길이로 제외되었는지 터미널 안내를 확인합니다.

| 실패 단계 | 확인 방법 | 이후 실험 후보 |
|---|---|---|
| 원본 문서에 정보 없음 | PDF에도 현재 값·설명이 없는지 확인 | 한계 안내, 필요한 원본 보완 |
| 분석 누락 | PDF에는 있는데 원래 Markdown에는 없음 | OCR·문서 분석 모델 또는 설정 비교 |
| 청킹 문제 | Markdown에는 있는데 청크에서 문맥·설명이 갈라짐 | 청킹 경계·상위 문맥 조정 |
| 검색 문제 | 적절한 청크가 코퍼스에 있지만 top-k에 없음 | 질의 표현·키워드 결합·재순위화 비교 |
| 답변 문제 | 충분한 근거가 Qwen 입력에 들어갔는데 오답 | 프롬프트·답변 모델 비교 |

한 번에 하나의 조건을 바꾸고 같은 질문을 다시 사용합니다. 새로 만든 질문도 추가해 기존 질문에만 맞춘 개선인지 확인합니다.
평가 개념 참고: [LangChain의 RAG 평가 구분](https://www.langchain.com/langsmith/evaluation).

## 이번 공부에서 설명할 수 있어야 할 것

| 주제 | 설명·실습할 내용 |
|---|---|
| Paddle 구성 | PaddlePaddle은 실행 프레임워크, PaddleOCR은 도구 모음, PP-OCR은 검출·인식, Structure는 영역·표를 결합한 문서 분석 파이프라인 |
| 모델 원리 | 글자 영역 좌표를 찾는 검출과 문자열을 읽는 인식의 차이. 표 구조·레이아웃 모델이 별도로 필요한 이유 |
| 발전 과정 | 문자 인식과 문서 분석 계열의 발전 축을 구분하고, 버전별로 해결하려던 문제·변경 방법·효과를 한 줄씩 설명 |
| VL의 원리 | 배치 분석으로 영역·순서를 찾고, 영역 이미지를 시각·언어 모델로 인식하는 전체 파이프라인. 그림 링크와 의미 이해의 차이 |
| 청킹 | 분석된 Markdown을 우리 규칙으로 분할함. 번호별 분할의 이점과 표·불명확한 페이지를 유지하는 이유 |
| 검색 모델 | BGE-M3는 학습된 인코더로 문서·질문을 같은 벡터 공간에 표현. 현재는 dense 임베딩 기능만 사용하며 Qwen은 근거 텍스트로 답변 생성 |
| 유사도·FAISS | 정규화 벡터의 제곱 L2와 코사인 순위가 같은 이유. Flat 검색의 정확 검색 비용과 대규모 인덱스 선택의 차이 |
| LangChain | Document·임베딩·Retriever·프롬프트·LLM의 연결 계약. 도입만으로 검색 품질이 개선되는 것은 아니며 컴포넌트 교체·조합을 도와줌 |
| 선택 근거 | 로컬 실행·공개 코드·모델 선택·비용·속도·문서 특성을 설명하고, 측정하지 않은 성능을 최적이라고 주장하지 않기 |
| 실험 결론 | 잘 된 사례·실패 사례·바꾼 조건·측정 결과·남은 한계를 함께 설명 |

실습 범위의 표현 예시: **“PaddleOCR 문서 분석 결과를 비교하고, LangChain 기반 로컬 RAG를 구축해 검색 실패 원인을 단계별로 점검했습니다.”**
작성 시점(2026-10-14)에 완료된 것은 Structure 분석·RAG 구성·위 검색 실험입니다. 집에서 VL 분석과 실제 비교를 마쳐야 비교 결론을 쓸 수 있습니다.
답변 품질 평가도 별도 수행해야 합니다. 프레임워크 내부 구현·모델 학습·파인튜닝은 추가 학습 범위로 구분합니다.

원리 학습 참고: [PaddleOCR 3.0 기술 보고서](https://arxiv.org/abs/2507.05595),
[PaddleOCR-VL 파이프라인 설명](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html),
[BGE-M3 모델 설명](https://huggingface.co/BAAI/bge-m3).
