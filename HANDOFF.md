# 집 GPU 실험 인수인계

기준일: 2026-10-14. 저장소 `jimjimjinger/rag_activity`, 브랜치 `main`.

최근 사용자 요청은 원격 접속 검토를 중단하고, Structure·VL을 직접 사용해 RAG를 구현한 경험을 프로젝트 서머리에 정리하는 것입니다. Tailscale·Cloudflare는 설치하지 않았고 외부 접속도 열지 않았습니다. `PROJECT_SUMMARY.md`와 README 구성 표를 실제 LangChain 문서 준비·검색·답변 흐름에 맞춰 정리했습니다. 이번 정리는 문서만 수정하고 모델을 다시 실행하지 않습니다.

## 현재 완료 상태

**실제 VRAM 약 8GB인 RTX 5060 Laptop GPU에서 새 Structure와 VL 구성 모두 42페이지 분석을 완료했습니다.** 먼저 [분석 보고서](tests/reports/analysis_review.md)와 [페이지별 비교표](tests/reports/comparison.md)를 읽으세요.

| 구성 | 입력 폭 | 완료 | 보정 전 청크 | 폴더 |
|---|---:|---:|---:|---|
| L: 기존 CPU 경량 Structure | 2,000px | 42/42 | 59 | `tests/results/structure_cpu/` |
| S: GPU Structure 후보, 8GB 프로필 | 1,200px | 42/42 | 44 | `tests/results/structure_gpu_8gb/` |
| V: PaddleOCR-VL-1.6, 8GB 프로필 | 1,200px | 42/42 | 55 | `tests/results/vl_gpu_8gb/` |

Markdown 해시, 전체 페이지 집합, JSON 존재와 이미지 참조를 검증했습니다. 새 결과를 원본과 대조한 대표 사례와 한계는 보고서에 있습니다. 큰 모델·더 많은 청크가 항상 더 정확한 것은 아닙니다.

## 후속 RAG 연결·실행 완료

후속 사용자 요청으로 Streamlit 화면 `src/app.py`를 추가했습니다. 이제 `run-rag.cmd`의 기본은 VL·GPU 브라우저 화면이며, `-Console`로 이전 터미널 모드를 사용할 수 있습니다. 답변 근거 페이지 버튼, 원본 PDF 미리보기, 추출 이미지·인식 텍스트 탭과 PDF 다운로드를 제공합니다. 이미지는 표시용이며 Qwen에는 텍스트만 전달합니다. 사용자 요청에 따라 모델 추론 벤치마크는 추가 실행하지 않습니다.

사용자의 후속 요청으로 기본 실행 장치를 GPU로 바꿨습니다. `run-rag.cmd -Profile vl` 또는 `scripts/run-rag.ps1 -Profile vl -Device gpu`로 직접 실행합니다. `-Device cpu`도 선택할 수 있습니다. 사용자가 검증 중단을 요청했으므로 GPU 설정 변경 후 추론·벤치마크는 실행하지 않았습니다. 아래 수치는 이전 CPU 검증 기록입니다.

집 PC에 `.venv`와 프로젝트 `.runtime/`의 portable Ollama 0.40.4·기존 BGE-M3·Qwen을 준비했습니다. 세 프로필의 인덱스를 생성하고 검색·답변을 실행했습니다. 임베딩·답변 모두 CPU 4스레드와 `num_gpu=0`입니다. 마지막 Ollama Qwen 상태는 `size_vram=0`이었습니다.

- 기본 `existing`: 기존 보정 코퍼스 73개·`data/rag/index.pkl`. 코퍼스 해시는 이전과 같습니다.
- `structure`: 보정 없는 새 Structure 44개·`data/rag/structure/`.
- `vl`: 보정 없는 VL 55개·`data/rag/vl/`. `scripts/run-rag.ps1 -Profile vl`로 실행합니다.
- 고정 질문 10개 검색: top-3 모두 찾은 질문 existing 8/10, structure 6/10, vl 8/10. top-5는 8/10, 9/10, 9/10.
- 기존·VL 각각 Q3·Q8·Q9·Q10·R1·I1의 6개 답변을 실행했습니다. VL의 Q10은 개선됐지만 Q9 H1 Tag는 두 프로필 모두 오답입니다. 기호·범례 해석, VL footer 누락과 입력 길이 제외를 점검해야 합니다.
- 이미지는 경로·해시 메타데이터만 연결하며 Qwen에 픽셀을 보내지 않습니다. 이미지 인식 답변·이미지 벡터 검색은 구현하지 않았습니다.

위 이미지 메타데이터를 Streamlit 화면에서도 사용합니다. 설치는 `requirements-ui.txt`의 Streamlit 1.65.0·pypdfium2 5.14.0이며 기존 RAG 잠금 버전을 유지했습니다. `scripts/setup-rag.ps1`은 UI 의존성도 설치합니다. 원본 페이지 PNG는 디스크에 저장하지 않습니다.

세부 근거와 제한은 [VL RAG 검증 보고서](tests/reports/rag_vl_review.md), 실제 입력·답변·환경·해시는 [실행 원자료](tests/reports/rag_vl_checks.json)에 있습니다. 원래 파싱 출력과 기존 보정은 수정하지 않았고 기본 프로필도 유지합니다.

## 실제 환경과 실행 설정

- Windows / RTX 5060 Laptop GPU / VRAM 8,151 MiB / RAM 약 23.4 GiB / NVIDIA 드라이버 610.47.
- 별도 Python 3.10.11과 `paddle-gpu-env`: paddlepaddle-gpu 3.3.1 cu129, paddleocr 3.7.0, paddlex 3.7.2.
- 배치 1, 페이지 폭 1,200px. VL은 영역 최대 262,144픽셀, 생성 최대 2,048토큰.
- Paddle 할당자 상한 6,144 MiB. 별도 감시는 장치 GPU 7,168 MiB 초과, 가용 RAM 2,048 MiB 미만, GPU 80도 이상이면 해당 분석만 중단합니다.
- VL 체크포인트 읽기·정밀도 변환은 CPU에서 수행하고 최종 모델은 GPU에 올립니다. 실제 파라미터에는 FP32와 BF16이 함께 있습니다.
- `native_cpu_loader.py`는 PaddleX 3.7.2에 고정된 임시 로더 보완입니다. 패키지 버전을 바꾸면 먼저 검토하세요. 설치 파일·캐시 가중치를 수정하지 않습니다.
- 성공한 표본·전체 실행의 관찰 최대 GPU 사용량은 S 3,196 MiB, V 6,682 MiB이며 최대 온도는 모두 68도입니다. 1초 관찰값이며 순간 최대값은 아닙니다.

## 이전 실패와 해결 범위

기존 코드에 16GB를 할당하는 값은 없었지만 사전 장비 정보는 16GB였고 실제 GPU는 8GB였습니다. 2,000px 기본 VL 실행과 재시도는 37페이지에서 Windows HYPERVISOR_ERROR로 재부팅됐습니다.

8GB 설정의 기본 로딩도 5GB·6GB 할당자 상한에서 MemoryError로 정상 중단됐습니다. 상한을 더 올리지 않고 CPU에서 가중치를 준비하도록 바꾼 후 새 표본과 전체 분석을 통과했습니다. 이번 실행에서는 추가 재부팅이나 자원 감시 중단이 없었습니다. 기존 시스템 오류의 근본 원인을 덤프로 확정한 것은 아닙니다.

중단된 CPU VL·이전 GPU 표본과 5GB·6GB 로딩 실패 원자료는 사용자 요청으로 삭제했습니다. 실패 원인과 해결 과정의 요약은 이 문서와 분석 보고서에 남겼습니다.

## 이번 결과의 핵심

- 선택한 원문 구절 정규화 일치는 L 9/19, S 8/19, V 18/19입니다. 구두점 차이도 포함하며 전체 정확도가 아닙니다.
- 선택한 정형 표 37~39페이지의 셀 일치는 L 167/168, S 108/168, V 168/168입니다. 작은 화면 캡처의 표는 이 검사에 포함되지 않습니다.
- S 38페이지는 표 구조가 깨졌습니다. V도 작은 화면 캡처의 날짜·필드명을 잘못 출력했고 31페이지 읽기 순서가 틀립니다.
- V 37페이지 범례는 JSON에는 있지만 `footer` 분류 때문에 Markdown에서 제외됐습니다. 인식과 출력 필터링의 실패를 구분하세요.
- 새 출력은 별도 RAG 프로필로 연결했습니다. 소규모 검색·답변 비교를 수행했지만 전반적인 답변 정확도 향상으로 일반화하지 않습니다.

## 이어서 작업할 때

1. 현재 Git 변경 사항과 사용자의 새 요청부터 확인합니다. 결과를 덮어쓰거나 기존 수동 보정을 모델 성능으로 제시하지 않습니다.
2. 두 GPU 스크립트는 기본적으로 8GB 프로필과 자원 감시를 사용합니다. 동일 설정이면 완료 페이지를 검사하고 재사용합니다. 모델·입력·패키지·로딩 방식을 바꾸면 별도 출력 폴더를 사용합니다.
3. 같은 설정을 반복해서 전체 실행할 필요는 없습니다. 추가 실험은 해상도, 인식기, Markdown 필터 등 한 조건만 바꿔 비교합니다.
4. RAG 비교가 요청되면 같은 청킹·BGE-M3·Qwen·질문 10개와 별도 인덱스로 검색·답변을 나눠 평가합니다. 작은 화면 캡처의 오류를 먼저 확인합니다.

실행 명령과 상세 설정은 [tests/README.md](tests/README.md)에 있습니다. 실제 설치 목록은 `tests/reports/gpu_packages.txt`, 자원 사용 원자료는 `tests/logs/` 아래 구성별 폴더의 `resources_*.jsonl`입니다.

## 기존 운영 RAG와 작업 원칙

- 회사 CPU 환경에서 PDF 분석 → 원본 대조 보정 → 기능 항목별 청킹 → LangChain Document → BGE-M3 → FAISS → Qwen의 동작을 확인했습니다. 이후 집 PC에서 파싱 비교와 별도 CPU RAG 프로필의 검색·답변도 실행했습니다.
- 운영 코퍼스 `data/rag/corpus.jsonl` 73개와 `markdown_corrections.json`을 유지합니다. 청킹은 Paddle이 아니라 `src/build_corpus.py`가 수행합니다.
- `bge-m3:567m`, `qwen3:4b-instruct`와 LangChain 구성은 바꾸지 않습니다. 새 기능·벡터 저장소·프레임워크를 이 작업을 계기로 추가하지 않습니다.
- 코드 준비와 실제 검증을 구분합니다. 현재 확인한 것은 이 장비·PDF·설정의 한 번의 전체 실험입니다.
- `tests/test_rag.py`는 이전 사용자 요청으로 삭제한 파일이므로 복구하지 않습니다.
- Windows 가상화·보안 설정이나 BIOS를 임의로 변경하지 않습니다. 이전의 2,000px 무제한 VL 실행을 자동 재시도하지 않습니다.
