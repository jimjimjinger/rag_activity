# PM209 문서 보강·청킹·임베딩

그림 설명·표 추출의 문제와 대응 과정은 [시행착오 정리](reports/TRIAL_AND_ERROR_20261023.md)에 있습니다. 자동 처리로 그림 23,807개와 표 2,308개를 저장했고, 남은 표 12개는 2026-10-26 원본 이미지 기준 수기 보강본을 별도로 만들었습니다. 이 12개는 검색용 요약 전사이며 정확도 재검수가 필요합니다.

청킹 규칙은 [설계 문서](PM209_IMAGE_RETRIEVAL_COMPARISON.md#청킹-규칙-초안-v02--원-논문-참고)에 정리했습니다. 공백 포함 500자·최대 100자 겹침으로 3개 매뉴얼 48페이지를 처리하고 Qwen3-Embedding-0.6B·Qdrant 로컬 저장·검색까지 실행했습니다. [실행 안내](RAG_PIPELINE.md)와 [샘플 평가](reports/RAG_SAMPLE_20261026.md)를 확인하세요. 이후 전체 209개 매뉴얼의 126,938개 벡터 저장·검증을 완료했습니다([전체 결과](reports/RAG_FULL_RUN_20261026.md)). 답변 생성은 아직 구현하지 않았고, 한국어 검색 품질에는 보완이 필요합니다.

제공된 페이지 JSONL을 재사용하고, `Product Image`·`illustration`·`graphic` 영역에 영어 `description`을 추가합니다. 새 설명은 짧은 영어 2~4문장, 최대 100단어를 목표로 핵심 대상·관계·필수 표기만 담습니다. 전체 페이지 OCR이나 그림 영역 재검출은 수행하지 않습니다.

## 데이터 준비

기본 경로는 `datasets/PM209`입니다. `data/train.jsonl`, `val.jsonl`, `test.jsonl`과 각 레코드의 `image_filename`이 가리키는 JPG가 필요합니다. 공식 ZIP에는 메타데이터가 없으므로 [공식 meta_data](https://github.com/AIM3-RUC/MPMQA/tree/master/meta_data)도 별도로 준비합니다.

현재 로컬 데이터는 209개 매뉴얼·10,018페이지·22,021 QA입니다. `meta_data/name2meta.json`의 209개 매뉴얼 ID와 데이터의 매뉴얼 ID를 대조합니다. `name2meta`는 브랜드·제품 선택용이며 설명 생성 코드가 직접 읽지는 않습니다.

## 실행 방법

아래 명령의 작업 위치는 저장소 루트 `rag_activity`입니다. 기존 Python 환경은 보존했습니다. 새 PC에서는 Python 3.10 이상으로 환경을 만들고 의존성을 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r PM209/requirements-images.txt
```

모델 실행 없이 그림 3개를 잘라 확인합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B PM209/src/describe_images.py --types illustration graphic --limit 3 --preview
```

설명 생성에는 이미지 입력을 지원하는 Ollama와 `qwen3.5:4b`가 필요합니다. 로컬 설정과 현재 장비는 [실행 환경](ENVIRONMENT.md)에 정리했습니다. 다음 명령 중 앞의 두 명령은 서버·모델 준비만 수행하며, 마지막 명령은 실제 설명 생성이므로 이후 실행 단계에서 사용합니다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File PM209/scripts/start-ollama.ps1
.\.venv\Scripts\python.exe -X utf8 -B PM209/scripts/prepare_model.py

# 실제 설명 생성은 이후 실행 단계에서 사용
.\.venv\Scripts\python.exe -X utf8 -B PM209/src/describe_images.py --types illustration graphic --limit 3
```

기본 서버는 `http://127.0.0.1:11434`입니다. 다른 서버에는 `--url`을 지정합니다. `--manual`로 매뉴얼을 한정하고, `--split train val`처럼 분할을 선택할 수 있습니다. `--limit`은 이번 실행에서 **새로 생성할 그림 수**입니다. 기본값은 5이며, `--all`은 명시적으로 선택한 경우에만 전량 처리합니다.

전체 train·val·test의 세 그림 유형을 이어서 처리할 때는 다음 실행기를 사용합니다. 100개마다 해당 split 전체를 검증·저장하며, 이미 저장된 설명은 재사용합니다. 한 번에 실행기 하나만 허용합니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -B PM209/scripts/run_full.py --batch-size 100
```

실시간 상태는 저장소 루트의 `.runtime/pm209-full/status.json`에 기록합니다. `completed`는 현재 배치에서 생성한 설명도 포함하며, `saved`는 정상 JSONL에 저장된 수입니다. GPU 기록은 같은 폴더의 `gpu.jsonl`에 있습니다. 정상 중단하려면 해당 폴더에 `STOP` 파일을 만듭니다. 진행 중인 요청이 끝난 뒤 저장하고 종료하며, 재개할 때는 `STOP`을 제거하고 같은 명령을 실행합니다. Windows 자동 유휴 절전은 실행기 동작 중에만 방지합니다.

출력 한도·잘못된 JSON·단어 수 초과는 그림마다 최대 세 번 시도합니다. 첫 요청은 최대 100단어, 재시도는 두 문장과 최대 60단어·40단어를 요청하며, 응답 토큰 한도는 768로 유지합니다. 문장 수는 프롬프트 지침이고 단어 수·JSON 형식·출력 잘림은 코드로 검증합니다. 전체 실행기는 세 번 모두 실패한 그림을 기록하고 다음 그림으로 넘어간 뒤, 다른 모든 항목을 처리하고 미완료 그림을 한 차례 더 시도합니다. `attempt-errors.jsonl`에는 실패 응답의 일부와 원인, `figure-failures.jsonl`에는 그림 ID와 실패 기록을 남깁니다. 이 파일들은 `.runtime/pm209-full`에 있으며, 보강본 그림에는 성공한 `description`만 추가합니다.

Ollama가 생성 도중 `token repeat limit reached` 오류를 반환한 경우도 동일하게 그림별 재시도합니다. 이 경우 응답 객체가 없으므로 진단 로그의 응답 미리보기는 null이며 오류 메시지를 기록합니다. 오류 처리 버전은 `format-length-token-repeat-v2`입니다.

`deferred`는 현재 재처리를 기다리는 그림 수입니다. 두 차례 전체 순회 후에도 빠진 설명이 있으면 상태는 `needs_review`이며, 모든 대상 설명을 저장한 경우에만 `complete`입니다. 서버 연결·GPU·파일 오류와 원본 검증 실패는 정상 결과를 저장하고 중단하여 원인을 확인합니다.

## 저장과 재개

```text
datasets/PM209/data/vlm/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── previews/<split>/                 미리보기 PNG와 manifest.jsonl
```

- 보강한 split은 원본의 모든 페이지·영역·QA를 담습니다. 매뉴얼을 한정하거나 그림 1개만 처리해도 다른 페이지를 제외하지 않습니다.
- 그림 객체에 추가하는 필드는 `description` 하나입니다. 숫자·단위·읽을 수 있는 표기도 설명 안에 담습니다.
- 같은 출력 경로로 재실행하면 기존 description을 재사용하고 다음 그림을 처리합니다. 입력·모델 설정을 바꾼 비교에는 별도 `--output` 경로를 사용합니다.
- 기존 보강본이 원본 페이지·좌표·본문·QA와 다르면 중단합니다.
- 전체 실행기는 그림별 응답 오류를 기록하고 계속 진행합니다. 서버 오류 등 나머지 생성 오류나 Ctrl+C가 발생하면 완료한 설명과 나머지 페이지를 저장한 뒤 종료합니다. 완성한 임시 파일로 기존 결과를 교체하므로 생성 중에 보강본 일부만 남기지 않습니다.
- 프로세스를 강제 종료하거나 전원이 끊기면 `.jsonl.tmp`가 남을 수 있습니다. 다음 실행은 덮어쓰지 않고 중단합니다. 임시 파일을 별도 보관·확인한 뒤, 마지막 정상 JSONL부터 재개합니다. 마지막 정상 저장 이후의 설명은 다시 생성될 수 있습니다.
- 처리 한도에 도달하면 현재 split을 끝까지 복사해 완성합니다. 그 이후 split은 새로 만들지 않습니다.

원본 JSONL은 수정하지 않습니다. QA는 보강본에서 보존하지만, VLM에는 잘라낸 그림과 고정 지시문만 전달합니다.

## 확인 방법과 현재 상태

저장·재개·오류 처리 검증은 실제 모델 없이 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B -m unittest discover -s PM209/tests -v
```

2026-10-21 저장 형식을 최종 설계에 맞게 수정했습니다. GPU 표본 23개를 생성해 속도·메모리와 일부 원본을 확인한 뒤, 같은 날 19:15 KST 전체 23,807개 처리를 시작했습니다. [실측 보고서](reports/GPU_SAMPLE_20261021.md)의 전체 시간 추정은 약 14시간이며, 작업 여유를 두면 15~20시간입니다. 현재 실행 결과는 `.runtime/pm209-full/status.json`으로 확인합니다. 청킹·임베딩·Qdrant·질문 답변은 후속 작업입니다.

첫 전체 실행은 2026-10-21 21:14 KST 그림 하나가 출력 한도에 도달해 중단됐고 3,493개를 보존했습니다. 2026-10-22 간결한 프롬프트와 그림별 재시도·실패 기록을 추가했습니다. 문제 그림을 먼저 재처리하여 저장한 뒤 07:25 KST 3,494개부터 전체 작업을 재개했습니다. 앞서 저장된 3,493개는 이전 프롬프트 결과이며 재생성하지 않았습니다. 현재 프롬프트 버전은 `concise-2-4-sentences-v1`입니다.

2026-10-22 13:43 KST Ollama의 토큰 반복 감지 오류로 22,798개에서 중단됐습니다. 해당 오류의 재시도 처리와 회귀 검사를 추가하고 14:46 KST 남은 1,009개 처리를 재개했습니다. 기존 설명을 재사용하여 약 19분 38초 뒤인 **15:06 KST 전체 23,807개 생성·저장을 완료**했습니다. 재개 중 반복 감지 오류 1회는 재시도로 해결했으며, 미완료 그림은 없습니다.

15:25 KST 원본과 보강본 train·val·test를 전부 대조하여 **10,018페이지·211,609영역·22,021개 QA 보존**과 대상 그림 23,807개의 비어 있지 않은 `description`을 확인했습니다. [최종 검증 기록](reports/full_run_validation_20261022.json)에 분할별 수치를 저장했습니다. 이 검증은 누락·데이터 보존 검사이며, 설명 내용의 의미 정확도를 평가한 결과는 아닙니다.

세부 배경은 [데이터 명세](PM209_DATA_SPEC.md), 현재 결정은 [설계와 GPU 작업](PM209_IMAGE_RETRIEVAL_COMPARISON.md)에 있습니다.

## 표 추출 실험 중단

2026-10-22 사용자 요청으로 이번 Qwen 기반 표 추출 실험을 실패로 정리하고 중단했습니다. Markdown 구조 오류와 반복 출력, 과도한 형식 검증으로 실패가 많아 결과를 채택하지 않았습니다.

HTML·Markdown 표 추출 결과, 표 전용 코드·설정·테스트·비교 보고서와 실행 로그를 삭제했고, 매시간 진행 보고 예약도 중지했습니다. 표 추출 작업은 자동으로 재개되지 않습니다.

원본 PM209 데이터와 `data/vlm/{train,val,test}.jsonl`의 그림 description 23,807개, 기존 페이지·영역·ID·좌표·QA는 보존했습니다. 이 시점에는 그림 설명까지 완료했고, 이후 아래 PaddleOCR 표 추출로 전환했습니다.

## PaddleOCR 표 3개 샘플

2026-10-22 후속 요청으로 `TableRecognitionPipelineV2`를 사용해 일반 표·병합 표·작은 글자가 많은 표를 하나씩만 인식했습니다. 기존 Table 좌표로 메모리에서 자른 이미지에 OCR과 표 구조 인식을 적용했으며, 전체 표 처리는 실행하지 않았습니다. 기존 데이터와 그림 description은 수정하지 않았습니다.

[원본 이미지와 HTML 비교](reports/paddle_table_sample_20261022/review.html), [수동 검수·실행 명령](reports/paddle_table_sample_20261022/manual_review.json), [셀·보존 검증](reports/paddle_table_sample_20261022/validation.json)에 결과가 있습니다. `tables.sample.jsonl`은 원래 Table 필드에 `table_content`를 추가한 별도 샘플이며, `pages.sample.jsonl`은 기존 페이지·본문·그림 description·QA를 함께 보존한 세 페이지 샘플입니다.

세 표 모두 HTML을 반환했고 행·열과 병합 구조, 마지막 행을 보존했습니다. 그러나 일반 표는 온도 단위 오류와 `-` 누락으로 정규화 후 11/16셀, 병합 표는 `Inactive`의 I/l 오인식으로 20/21셀이 일치했습니다. 긴 표는 11행×2열과 숫자·단위 확인 항목을 보존했지만 버튼·운전 모드 아이콘이 누락되거나 오인식됐습니다. HTML 저장 성공을 완전한 인식 성공으로 간주하지 않습니다. 반환 HTML은 보정하지 않고 그대로 보관했습니다.

전용 환경은 `.runtime/venv-paddle-tables`입니다. 공식 Windows RTX 50용 PaddlePaddle GPU(CUDA 12.9·cuDNN 9.9), PaddleOCR 3.3.2와 PaddleX 3.3.11을 사용했습니다. 모델은 PP-OCRv5_server_det/rec, SLANeXt wired/wireless, RT-DETR-L 표 셀 검출입니다. 레이아웃·방향 분류·왜곡 보정은 끄고 표별 순차 처리, OCR 배치 1, Paddle 메모리 할당 한도 6,144MiB를 적용했습니다. 한글 경로를 열지 못하는 Windows 모델 로더 문제를 피하기 위해 모델 캐시는 `%LOCALAPPDATA%/PM209-PaddleTables/paddlex`에 둡니다.

```powershell
# 저장소 루트에서 실행합니다. 기존 샘플 HTML이 있으면 덮어쓰기를 거부합니다.
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -u -B PM209/scripts/run_paddle_table_sample.py
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -B PM209/scripts/review_paddle_table_sample.py
```

설치 재현 명령은 `PM209/scripts/setup-paddle-tables.ps1`, 패키지 목록은 샘플 폴더의 `requirements.lock.txt`에 있습니다. 표별 실측은 약 1.71초·1.04초·1.87초이며, 초기 모델 준비 약 200초는 별도입니다. 장치 전체 VRAM 관측 최대치는 2,810MiB, 온도는 60°C였고 종료 후 GPU 메모리는 해제됐습니다.

## PaddleOCR 전체 표 추출

2026-10-22 샘플 검수 후 사용자 승인으로 전체 2,320개 Table 처리를 시작했습니다. 위 샘플의 HTML 세 개는 그대로 재사용합니다. `data/vlm`은 읽기 전용 입력으로 취급하고, 별도 `data/paddle_tables_html/{train,val,test}.jsonl`의 Table 객체에만 `table_content`를 추가합니다. 전체 결과가 문자·단위까지 정확하다는 뜻은 아니며, 청킹·임베딩·검색은 포함하지 않습니다.

```powershell
# GPU 추론 없이 샘플 재사용·입력 보존·전체 JSONL 저장을 확인
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -B PM209/scripts/run_paddle_tables.py --prepare-only
# 미완료 표만 GPU로 순차 처리
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -u -B PM209/scripts/run_paddle_tables.py --all
```

표마다 `assets/<split>/<표 식별자 해시>/image.png`, `table.html`, `raw.json`, `complete.json`을 저장합니다. 완료 기록은 나머지 파일 저장 후 마지막에 쓰며, 입력 이미지·좌표·설정과 각 파일의 해시가 일치할 때만 재사용합니다. 실패 표는 `failure.json`과 `.runtime/pm209-paddle-tables/table-failures.jsonl`에 기록하고 다음 표로 넘어갑니다. 실행을 다시 요청하면 실패·미처리 표만 처리합니다. 같은 실행에서 실패 표를 반복 시도하지 않습니다. 디스크·권한 오류나 GPU 상태가 손상된 오류는 정상 결과를 보존하고 종료합니다.

100개 처리 또는 5분마다 전체 페이지 JSONL에 체크포인트하고, 정상 중단·종료 시에도 반영합니다. 상태 파일은 `.runtime/pm209-paddle-tables/status.json`입니다. `completed`는 복구 파일까지 저장된 HTML 수, `saved`는 전체 JSONL 반영 수, `journal_pending`은 그 차이입니다. `eta_seconds`는 최근 최대 100개 처리 시간과 관측 체크포인트 비용으로 계산한 이번 순회 잔여 추정치이며, 실패 표 재처리와 수동 검수 시간은 제외합니다.

`data/paddle_tables_html/review_index.html`에서 원본 표·HTML·OCR 결과를 확인합니다. 단위 표기, 버튼·모드 안내, 낮은 OCR 신뢰도, 빈 셀과 병합 셀은 검수 우선 대상으로 표시합니다. 이 표시는 오류 확정이 아니며 아이콘이나 글자 누락을 전부 검출하지 못합니다. 샘플 세 개는 알려진 오류가 있는 수동 검수 결과이고, 나머지는 모두 수동 검수 전 상태입니다. HTML은 자동 보정하지 않습니다.

중단하려면 `.runtime/pm209-paddle-tables/STOP` 파일을 만듭니다. 현재 표가 끝난 뒤 체크포인트하고 종료합니다. 재개 요청 시에만 STOP을 제거하고 다시 실행합니다. 실행 중에만 Windows 자동 유휴 절전을 방지하며, 화면이 꺼져도 작업은 계속됩니다. 최종 `final-validation.json`은 원본·그림 보강본 해시와 페이지·영역·ID·좌표·본문·그림 description·QA 보존을 검증합니다. `status=complete`는 추출 완료를 뜻하며 의미 정확도 검수 완료와 구분합니다.

## 실패한 Paddle 표 232개 보완

2026-10-22 사용자의 후속 요청으로 첫 순회의 실패 232개만 보완했습니다. 기존 `data/paddle_tables_html`의 성공 2,088개와 실패 원시 파일은 보존하며, 새 결과는 **`data/paddle_tables_html_repaired`**에 저장합니다. 이 폴더의 train/val/test JSONL은 기존 성공 결과와 채택한 보정본을 합친 전체 페이지 데이터입니다. 기존 그림 description·본문·ID·좌표·QA는 변경하지 않습니다.

`src/repair_paddle_html.py`는 HTML에서 생략되거나 잘못 짝지어진 행 그룹 태그를 명시화하고, OCR 본문의 `<`, `&`, 코드 문자열을 문자로 표시되도록 이스케이프합니다. 셀 내용·순서·속성·병합 위치는 보정 전후 대조합니다. 끝나지 않은 행/셀, 겹치는 병합, 표 밖으로 나가는 병합, 반복된 빈 꼬리 행은 임의로 채우지 않습니다. 병합 셀에 완전히 덮이는 빈 행은 유효하므로 유지합니다.

- **208개:** GPU 재실행 없이 저장된 결과에서 형식을 복구했습니다. 이는 의미 정확도 수동 검수 완료를 뜻하지 않습니다.
- **24개만 GPU 재인식:** 동일 `TableRecognitionPipelineV2`에서 RT-DETR 셀 좌표→HTML, OCR 결합 방식, 1.5배 입력 크기, 24px 흰 여백, SLANeXt 자체 셀 좌표 사용을 필요한 표에만 단계적으로 비교했습니다. QA는 모델에 입력하지 않았습니다. PP-OCRv5와 기존 로컬 모델·8GB GPU 설정을 사용하며 Qwen/VLM은 사용하지 않습니다.
- 재인식 후보를 원본 이미지와 대조해 **12개를 채택**, 나머지 **12개는 내용 누락·행열 연결 오류가 남아 미완료**로 보관했습니다. 채택한 표에도 온도 기호·아이콘·철자 등의 알려진 OCR 오류가 남습니다. 형식 복구/추출 성공은 정확도 인증이 아닙니다.
- 합산 결과는 **2,308 / 2,320개(99.48%)**, 이번 복구는 **220 / 232개**입니다. 전체 완료 상태는 아닙니다.

[보완 232개 이미지·HTML 비교 목록](datasets/PM209/data/paddle_tables_html_repaired/review_index.html), [보존 검증 및 미완료 목록](datasets/PM209/data/paddle_tables_html_repaired/final-validation.json), [24개 재인식 후보 검수 기록](datasets/PM209/data/paddle_tables_html_repaired/manual-decisions.json)을 확인합니다. 개별 `repair.json`은 변경 방식·원시 입력 해시·셀 보존 검사를 기록하고 `attempts/`에는 채택하지 않은 후보도 남깁니다. 보정 HTML과 모델이 반환한 원시 HTML은 구분하여 보존합니다.

실행 명령은 저장소 루트 기준입니다. 각 단계는 완료 복구 파일을 재사용하며, 기존 성공 표를 재인식하지 않습니다. `retry-review`와 `retry-e2e`는 검수 기록에서 `needs_retry`로 지정한 항목만 처리합니다. 현재 남은 12개는 `unresolved`로 기록되어 자동 재실행하지 않습니다.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B PM209/scripts/repair_paddle_tables.py repair
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -B PM209/scripts/repair_paddle_tables.py retry
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -B PM209/scripts/repair_paddle_tables.py retry-review
.\.runtime\venv-paddle-tables\Scripts\python.exe -X utf8 -B PM209/scripts/repair_paddle_tables.py retry-e2e
.\.venv\Scripts\python.exe -X utf8 -B PM209/scripts/repair_paddle_tables.py accept
.\.venv\Scripts\python.exe -X utf8 -B PM209/scripts/repair_paddle_tables.py finalize
```

`accept`는 `.runtime/pm209-paddle-repair/manual-decisions.json`에서 원본 대조 후 채택한 후보만 반영합니다. `finalize`는 원본·그림 보강본 해시, 이전 결과 파일 목록/크기/수정 시각 및 JSONL 해시, 기존 성공 셀 내용과 전체 페이지 필드 보존을 확인합니다. 표 보완 관련 회귀 검사 13개를 통과했습니다. 이 표 보완 단계에는 청킹·임베딩·검색을 포함하지 않았으며, 이후 진행 내용은 [RAG 실행 안내](RAG_PIPELINE.md)에 있습니다.

## 남은 12개 검색용 수기 보강본

`paddle_tables_html_repaired`에서 비어 있던 12개를 원본 표 영역 이미지와 대조하여 `table_content` HTML로 작성했습니다. 원본 및 기존 2,308개 결과를 보존하고 `data/paddle_tables_manual_20261026/{train,val,test}.jsonl`에 별도 저장했습니다. 검색에 유용하도록 행·항목별 내용을 연결했지만, 긴 설명은 줄여 적었고 작은 아이콘·일부 보조 표시는 빠져 있습니다. **2,320개에 HTML이 존재한다는 뜻이지, 셀 단위 정확도 검수가 끝났다는 뜻은 아닙니다.**

각 표의 원본 이미지와 HTML은 [12개 검수 화면](datasets/PM209/data/paddle_tables_manual_20261026/review.html)에서 나란히 볼 수 있습니다. 원본 경로와 주의 사항은 같은 폴더의 `manual-review.json`에 있습니다. 재생성 명령은 저장소 루트 기준 `python -X utf8 -B PM209/scripts/complete_remaining_tables.py`이며 출력 폴더가 이미 있으면 덮어쓰지 않습니다. 전체 10,018페이지를 대조하여 12개 표의 `table_content` 외 모든 기존 필드가 같고, 2,320개 표에 HTML이 있음을 확인했습니다. 수기 보강 항목은 임베딩 전 주의 사항을 검수하거나 검색 대상에서 분리하는 것이 좋습니다.

## Git과 별도 전달 결과

Git에는 코드·설정·문서·샘플 검수 결과·전체 실행 요약을 포함합니다. 전체 JSONL·검수 이미지 묶음·벡터 DB·원본 데이터셋·모델 가중치는 로컬에 보관하고 별도 전달합니다. [2026-10-22 분할 압축본 안내](results/README.md)는 해당 날짜의 그림·표 결과를 설명하며, 이후 수기 보강본과 청킹·임베딩 결과는 포함하지 않습니다. 분할 파일과 manifest를 별도로 받은 경우 `PM209/results/20261022/`에 놓고 `python PM209/scripts/result_bundle.py restore`로 복원합니다. 파일별 SHA256을 확인하고 기존의 다른 결과를 덮어쓰지 않습니다.
