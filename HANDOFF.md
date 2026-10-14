# 집 GPU 노트북 작업 인수인계

기준일: 2026-10-14. 코드 기준 커밋: `808c927` (이 문서를 추가하기 직전).
저장소: https://github.com/jimjimjinger/rag_activity / 브랜치: `main`.

이 파일은 긴 대화에서 최종 결정과 현재 상태만 추린 요약입니다.
새 Codex는 먼저 실제 코드·Git 상태·현재 장비를 확인하고 이어서 작업합니다.

## 1. 목표와 작업 원칙

**다음 목표는 집 GPU에서 PP-StructureV3와 PaddleOCR-VL을 실행하고, 같은 PDF의 보정 전 분석 결과를 비교하는 것입니다.**

- RAG는 이미 동작합니다. 당장은 기능 확장보다 Paddle 문서 분석 공부와 비교가 우선입니다.
- 구현은 최대한 간단하게 유지합니다. 작은 개선 때문에 구조를 크게 바꾸지 않습니다.
- BGE-M3와 Qwen 모델을 바꿀 계획은 없습니다.
- 기존 분석 결과를 덮어쓰거나 수동 보정한 결과를 원래 모델의 성능으로 제시하지 않습니다.
- 새 GPU 분석 결과를 운영 RAG 코퍼스에 자동 반영하지 않습니다.
- 코드가 준비된 상태와 실제 장비에서 검증된 상태를 구분합니다.
- 설명은 한국어로, 바꾼 이유와 모델별 역할을 이해하기 쉽게 작성합니다.
- `tests/test_rag.py`는 사용자 요청으로 삭제했습니다. IDE에 탭이 남아 있어도 복구할 필요 없습니다.
  현재 `tests/`는 분석 비교 자료와 실행 코드 폴더입니다.

## 2. 먼저 읽을 파일

| 파일 | 읽는 목적 |
|---|---|
| [README.md](README.md) | 전체 구조·흐름·새 PC의 RAG 설치 |
| [tests/README.md](tests/README.md) | GPU 설치·실행, 모델 선정 근거, 비교 기준, 고정 질문과 검색 평가 |
| [src/README.md](src/README.md) | 운영 Python 4개와 LangChain 연결 방식 |
| [paddle.md](paddle.md) | Paddle 계열 역할과 발전 과정 |
| [tests/run_parser.py](tests/run_parser.py) | 새 Structure·VL 실험의 실제 모델 설정과 저장·재개 동작 |

자세한 명령은 위 문서와 스크립트를 기준으로 합니다. 이 파일에 없는 대화를 추정해 새 요구사항을 만들지 않습니다.

## 3. 현재 운영 RAG

```text
PDF
 → parse_pdf.py: PP-StructureV3 분석 → 페이지별 Markdown·추출 이미지
 → build_corpus.py: 원본 대조 보정 + 기능 항목별 청킹 → corpus.jsonl
 → rag.py --build: LangChain Document → BGE-M3 임베딩 → FAISS 저장

질문
 → BGE-M3 질문 임베딩
 → LangChain FAISS 검색, 기본 top-k=3
 → 검색 본문·페이지 출처 + 프롬프트
 → Qwen 답변
```

- 원본: `documents/B2B Dashboard Manual_v3.pdf`, 총 42페이지.
- 운영 코퍼스: `data/rag/corpus.jsonl`, 현재 73개 청크.
- 청킹은 Paddle이 아니라 직접 작성한 `src/build_corpus.py`가 합니다.
  기능 항목별 분리를 기본으로 하며 표·경계가 불명확한 페이지는 전체를 유지합니다.
- `data/rag/markdown_corrections.json`에는 원본 PDF로 확인한 OCR·제목 누락 보정이 있습니다.
  보정 대상 원본·Markdown 검사는 유지합니다. 비교 실험에는 보정을 적용하지 않습니다.
- 임베딩: Ollama의 `bge-m3:567m`, 답변: `qwen3:4b-instruct`.
- 연결: `OllamaEmbeddings`, `langchain_community.vectorstores.FAISS`,
  `ChatPromptTemplate | ChatOllama | StrOutputParser`.
- FAISS는 `from_documents()`, `as_retriever().invoke()`를 사용합니다.
  생성·불러오기 양쪽의 `normalize_L2=True`로 정규화하며, 정규화 L2 검색은 코사인 검색과 같은 순위입니다.
- Windows 한글 경로 문제를 피하려고 LangChain의 `serialize_to_bytes()` /
  `deserialize_from_bytes()`로 `data/rag/index.pkl` 하나에 저장합니다.
  직접 생성한 로컬 인덱스만 불러오며 이 파일은 Git에서 제외합니다.
- 직접 작성했던 검색기·정규화 수식·점수 환산·인덱스 해시 검사·벡터 재사용은 제거했습니다.
  코퍼스가 바뀌면 `scripts/build-rag.ps1`이 청킹과 임베딩·인덱스 생성을 순서대로 실행합니다.
- 기존 `index.json`, `index.faiss`는 삭제했습니다.
- 이미지 경로는 청크 metadata에 연결합니다. 현재 Qwen에는 텍스트만 전달하며 이미지 의미 검색·이미지 답변은 없습니다.
- `langchain-community==0.4.2`는 유지보수가 종료된 패키지입니다. 현재 동작 확인한 버전으로 고정했습니다.
  이 인수인계를 계기로 다른 벡터 저장소나 프레임워크로 다시 바꾸지 않습니다.

## 4. 완료와 미완료

| 항목 | 현재 상태 |
|---|---|
| 운영 RAG | 회사 CPU PC에서 73개 인덱스 생성·저장·불러오기·검색·Qwen 답변 확인 |
| 검색 평가 | 고정 질문 10개의 검색 결과는 `tests/README.md`에 기록. 필요한 페이지를 모두 찾은 질문 8/10, 답변 정확도 아님 |
| 기존 Structure | `tests/pp_structurev3/`에 42페이지의 수동 보정 전 분석 복사본과 보정 없는 59개 청크 |
| CPU VL | `tests/paddleocr_vl/`에 1페이지 표본만 있음. 느려서 사용자 요청으로 중단, 전체 완료 아님 |
| GPU 실행 준비 | 설치·실행 스크립트, 모델 설정, 비교 코드 준비. 구문·설정 연결·모의 실행 확인 |
| 집 GPU 검증 | 아직 안 함. 실제 설치·CUDA 연산·모델 추론·VRAM 적합성 확인 필요 |
| GPU 분석 결과 | `tests/pp_structurev3_gpu/`, `tests/paddleocr_vl_gpu/`는 아직 생성 전 |
| 비교 결론 | GPU 결과를 원본 PDF와 확인한 후 작성해야 함. 어느 구성이 더 정확한지 아직 결론 없음 |

한 질문의 Qwen 답변 동작은 확인했지만, 고정 질문 10개 전체의 답변 정확도를 평가한 것은 아닙니다.
GPU가 빨라졌다는 사실과 모델 정확도가 높아졌다는 사실을 구분합니다.

## 5. 집 PC에서 가장 먼저 할 일

사용자가 알려준 장비는 **Windows 11 / RTX 5060 계열 / VRAM 16GB**입니다.
정확한 GPU 이름·실제 VRAM·드라이버는 `nvidia-smi`로 확인해야 합니다.

1. 저장소를 clone하거나 기존 작업 폴더에서 변경 상태를 확인한 뒤 최신 `main`을 받습니다.
2. Git, Python 3.10 64비트, NVIDIA 드라이버와 GPU를 확인합니다.
3. [GPU 설치 스크립트](scripts/setup-paddle-gpu.ps1)를 읽고 전용 환경을 준비합니다.
4. 두 파이프라인 모두 9·37페이지로 실제 추론·파일 저장을 확인합니다.
5. 표본 실행이 정상인 경우 각각 42페이지 전체를 실행합니다. 두 파이프라인은 순서대로 실행합니다.
6. [비교표](tests/comparison.md)와 원본 PDF를 보고 누락·표 구조·읽기 순서·제목/설명 연결을 비교합니다.

RAG 폴더의 PowerShell에서:

```powershell
nvidia-smi
python --version

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-paddle-gpu.ps1

powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1 --pages 9 37
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-vl-gpu.ps1 --pages 9 37

# 위 두 표본의 성공을 확인한 뒤 전체 실행
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-structure-gpu.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-vl-gpu.ps1
```

현재 준비한 GPU 구성은 `paddlepaddle-gpu==3.3.1`의 cu129 빌드,
`paddleocr==3.7.0`, 배치 1, 공통 렌더링 폭 2,000입니다.
설치 스크립트는 패키지 설치와 실제 GPU 연산 확인까지만 하며, 분석 명령을 실행할 때 모델을 내려받습니다.
이 조합은 집 PC에서 아직 검증하지 않았습니다. 설치·추론 오류가 있으면 실제 오류와 공식 안내를 확인하고 필요한 부분만 수정합니다.

회사 PC의 `C:\Users\CNXK\...` 경로를 집 PC에 복사해 하드코딩하지 않습니다.
환경과 모델 캐시는 스크립트가 찾는 로컬 runtime 경로를 사용합니다.

## 6. 비교할 세 구성과 완료 기준

| 구분 | 주요 구성 | 출력 |
|---|---|---|
| L: 기존 경량 Structure | PP-DocLayout-S, 영어 mobile OCR 등 기존 구성 | `tests/pp_structurev3/` |
| S: 새 Structure 후보 | PP-DocLayout_plus-L, PP-DocBlockLayout, server OCR, SLANeXt wired/wireless | `tests/pp_structurev3_gpu/` |
| V: VL | PP-DocLayoutV3 + PaddleOCR-VL-1.6-0.9B를 사용하는 준비된 파이프라인 | `tests/paddleocr_vl_gpu/` |

S는 검증 전 후보이며 최적 모델 조합이라는 뜻이 아닙니다.
`PP-OCRv5_server_rec`는 한글 전용 인식기가 아니며, 영어 전용 mobile 모델보다 항상 낫다는 보장도 없습니다.

- 새 실행 결과의 `run_summary.json`에서 실제 모델·패키지·GPU 정보·완료 페이지를 확인합니다.
- 표본 실행은 `partial`, 전체 42페이지가 완료되어야 `complete`입니다.
- 같은 명령을 다시 실행하면 같은 설정으로 저장된 완료 페이지를 확인하고 재개합니다.
- 모델·패키지·설정을 바꾸는 추가 실험은 별도 출력 폴더로 분리합니다.
- `compare.py`는 같은 청킹 규칙을 보정 없이 적용하고 비교표를 만듭니다.
  GPU 실행 wrapper는 성공 후 이 코드를 호출합니다.
- L의 과거 CPU 시간과 S/V의 GPU 시간을 공정한 속도 비교로 제시하지 않습니다.
- 글자 수·청크 수는 정확도가 아닙니다. 원본 PDF의 실제 제목·설명·표 값과 비교해 구체적인 사례를 기록합니다.
- 분석만 비교한 결과를 RAG 답변 개선 효과로 설명하지 않습니다.

관찰할 대표 페이지: 1(언어), 9(번호·제목), 11·22(설명 연결),
20(읽기 순서), 23(Download 문장), 37·39(표).
상세 기준과 기록 표는 [tests/README.md](tests/README.md)에 있습니다.

## 7. Git으로 옮겨지는 것과 새로 준비할 것

원본 PDF, 분석 결과·추출 이미지, 코퍼스, 코드와 문서는 저장소에 있습니다.
가상환경, 모델 가중치, 모델 캐시, 로컬 FAISS `index.pkl`은 들어 있지 않습니다.

**Paddle 비교만 먼저 한다면 Ollama·BGE-M3·Qwen과 RAG 인덱스 설치는 필수가 아닙니다.**
RAG도 실행할 때만 [README의 새 PC 준비](README.md#새-pc-준비)를 따릅니다.
그 경우 `scripts/setup-rag.ps1` → Ollama·두 모델 준비 → `scripts/build-rag.ps1` → `run-rag.cmd` 순서입니다.
현재 모델 이름은 유지합니다.

`.gitattributes`는 `tests/**/page_*.md`의 줄바꿈을 변환하지 않도록 설정되어 있습니다.
원본 분석 파일을 에디터로 재저장하면 검증 해시가 바뀔 수 있습니다. 비교할 때는 원본을 보존합니다.

## 8. 새 Codex에 보낼 시작 메시지

```text
이 저장소의 HANDOFF.md를 먼저 읽고, README.md와 tests/README.md를 확인한 뒤 작업을 이어가 줘.

지금 우선순위는 RAG 확장이 아니라 집 GPU에서 PP-StructureV3와 PaddleOCR-VL을 실행하고,
같은 PDF의 수동 보정 전 분석 결과를 비교하는 거야.
장비의 실제 GPU·VRAM·드라이버·Python을 먼저 확인하고,
준비된 스크립트로 환경 설치 → 9·37페이지 표본 검증 → 전체 분석 순서로 진행해 줘.
회사 PC에서 GPU 실행을 검증한 상태는 아니야.
기존 분석 결과와 운영 RAG는 보존하고, 임베딩·답변 모델은 바꾸지 말고,
오류가 있으면 필요한 부분만 최소한으로 수정해 줘.
작업 후 실제 검증 결과와 다음 진행 상태를 HANDOFF.md와 비교 문서에 갱신해 줘.
```

전체 대화 복사 없이 이 메시지와 저장소 파일로 시작하고, 문서에 없는 결정이 필요할 때만 사용자에게 확인합니다.
목표·관련 파일·제약·완료 기준을 전달하는 구성은 [OpenAI 공식 Codex 안내](https://learn.chatgpt.com/guides/best-practices)를 참고했습니다.
