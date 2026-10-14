# 기존 RAG와 VL RAG 연결·실행 결과

실험일: 2026-10-14. 이 PC에 RAG 전용 환경과 로컬 Ollama를 준비하고, 저장된 파싱 결과로 별도 인덱스를 생성했습니다. PDF 파싱은 다시 실행하지 않았습니다.

후속 사용자 요청으로 실행 기본값을 GPU로 바꿨습니다. 이 보고서의 수치는 CPU 검증 기록이며, GPU 설정 후 추가 검증은 사용자 요청에 따라 실행하지 않았습니다. CPU로 재현하려면 실행 명령에 `-Device cpu` 또는 `--device cpu`를 붙입니다.

이후 Streamlit 원본 페이지·이미지 확인 화면을 추가했습니다. `run-rag.cmd`는 브라우저 화면을 열고 `-Console`로 이전 콘솔을 사용할 수 있습니다. 아래 모델·검색·답변 결과는 화면 추가 전 기록이며 이미지 표시가 모델의 직접 이미지 이해를 추가한 것은 아닙니다.

**세 프로필의 검색과 기존·VL의 실제 답변 생성이 실행됐습니다. VL은 복합 질문에서 개선됐지만 H1 Tag의 점수 여부를 잘못 답했습니다. 현재 결과만으로 기본 RAG를 VL로 교체하지 않습니다.**

## 연결한 구성

| 프로필 | 코퍼스 | 청크 | 인덱스 | 보정 |
|---|---|---:|---|---|
| `existing` | `data/rag/corpus.jsonl` | 73 | `data/rag/index.pkl` | 기존 PDF 대조 보정 |
| `structure` | `data/rag/structure/corpus.jsonl` | 44 | `data/rag/structure/index.pkl` | 없음 |
| `vl` | `data/rag/vl/corpus.jsonl` | 55 | `data/rag/vl/index.pkl` | 없음 |

기본값은 `existing`입니다. 기존 코퍼스의 SHA-256은 `93a946cba290734d16b0a56f0126f31510018ba0d6e6ce29d7f769724be111e1`로 이전 기록과 같습니다. 원본 파싱 Markdown·JSON도 수정하지 않았습니다.

세 프로필은 같은 `split_page()` 규칙, BGE-M3 dense 임베딩, 정규화 FAISS 검색을 사용합니다. 기존 코퍼스에는 수동 보정이 있으므로 기존·VL의 차이를 파싱 모델 하나의 효과로 해석하지 않습니다. 새 Structure와 VL은 보정 없는 결과입니다.

## 환경과 메모리

- Python 3.10 전용 `.venv`, 기존 `requirements-rag-lock.txt` 버전 설치 및 `pip check` 통과.
- 공식 portable Ollama 0.40.4를 `.runtime/ollama/`에 준비하고 공식 체크섬을 검증했습니다. 모델은 `.runtime/models/`에 보관하며 둘 다 Git에서 제외합니다.
- 기존 모델 [BGE-M3 567m](https://ollama.com/library/bge-m3:567m), [Qwen3 4b-instruct](https://ollama.com/library/qwen3:4b-instruct)를 그대로 사용했습니다.
- **임베딩·답변 모두 `num_gpu=0`, CPU 4스레드**입니다. 인덱스 생성은 4개 청크씩 처리합니다. 모델은 한 번에 하나를 유지합니다.
- 로컬 서버 주소 `http://127.0.0.1:11435`, 문맥 4,096토큰, 최대 답변 256토큰, temperature 0.2입니다.
- 실행 중 NVIDIA 관찰값은 GPU 메모리 0 MiB였으며, 마지막 Ollama 모델 상태도 Qwen의 `size_vram=0`을 확인했습니다. 이 결과는 GPU RAG 성능 검증이 아닙니다.

환경·모델 digest·코퍼스·인덱스 해시와 실제 답변·입력 문맥은 [실행 원자료](rag_vl_checks.json)에 있습니다. 설치 방식은 [Ollama Windows 공식 안내](https://docs.ollama.com/windows), 고정 실행 파일은 [공식 v0.40.4 배포](https://github.com/ollama/ollama/releases/tag/v0.40.4)를 사용했습니다.

## 고정 질문 10개의 검색

[질문과 정답 기준](../fixtures/rag_questions.json)은 기존 평가 목록을 유지했습니다. 질문 벡터를 한 번 생성해 각 인덱스의 top-5를 검색하고 앞 1·3·5개를 비교했습니다. 별도 진단 R1·I1은 검색 점수에 포함하지 않습니다.

| 지표 | 기존 보정 RAG | 새 Structure RAG | VL RAG |
|---|---:|---:|---:|
| 필요한 페이지를 모두 찾은 질문: top-1 | 8/10 | 6/10 | 5/10 |
| 필요한 페이지를 모두 찾은 질문: top-3 | 8/10 | 6/10 | 8/10 |
| 필요한 페이지를 모두 찾은 질문: top-5 | 8/10 | 9/10 | 9/10 |
| 평균 페이지 Recall@3 | 0.85 | 0.65 | 0.80 |
| 평균 페이지 Recall@5 | 0.85 | 0.90 | 0.90 |

복합 질문 Q10은 필수 페이지가 22·37 두 개입니다. 기존은 37만 찾고 VL은 top-3에서 둘을 찾았습니다. 반대로 VL은 Q6 DICT 다운로드에서 23페이지를 top-5까지 찾지 못했고, Q5 상세 조건은 22페이지가 5위여서 기본 top-3에는 없습니다. 파싱 정확도가 검색 성공을 자동으로 보장하지 않습니다.

이 점수는 페이지 검색 결과이며 답변 정확도가 아닙니다. 같은 페이지를 찾았어도 필요한 문장이 없는 경우와 입력 길이로 제외된 경우를 별도로 확인합니다.

## 실제 답변: 기존·VL 각 6개

같은 top-3와 프롬프트로 Q3·Q8·Q9·Q10·R1·I1을 순서대로 실행했습니다. 생성된 12개 답변은 원자료에 저장했습니다. 아래는 주요 내용의 수동 점검이며 전체 답변 정확도 점수로 집계하지 않습니다.

| 질문 | 기존 RAG | VL RAG |
|---|---|---|
| Q3: 이슈 필터 초기화 | `Clear filter` 클릭, p20 인용 | 같은 안내, p20 인용 |
| Q8: PDP Title Length 최대 배점 | 20점, p37 인용 | 20점. p37 외 p22에 대한 부가 설명도 생성했으며 부가 인용의 정확성은 별도 미평가 |
| Q9: H1 Tag 점수 여부 | **오답:** 점수를 받는다고 답함 | **오답:** M을 점수 기준이라고 해석함 |
| Q10: 상세 조건 열기 + 20pt | 20pt만 답하고 상세 조건은 확인 불가 | 항목 화살표 클릭과 20pt를 함께 답함, p22·37 인용 |
| R1: 현재 India HVAC 점수 | 숫자를 만들어 내지 않음 | 숫자를 만들어 내지 않음 |
| I1: p23 화면 캡처 UPDATE DATE | 해당 근거를 못 찾아 확인 불가 | 해당 근거를 못 찾아 확인 불가 |

Q9의 원본 정답은 `M: Monitoring only (not scored)`입니다. 기존 입력 문맥에는 범례가 있어도 Qwen이 잘못 해석했습니다. VL p37 JSON에는 정확한 범례가 있지만 `footer` 분류 때문에 Markdown에서 제외됐으며, Q9의 다른 검색 페이지 p39·38도 8,000자 입력 제한 때문에 제외됐습니다. VL의 이 답변에는 p37만 전달됐습니다.

VL의 다른 질문도 입력 길이 제한으로 일부 페이지가 제외됐습니다. `skipped_ids`와 실제 `included_pages`를 원자료에서 확인할 수 있습니다. 표 셀마다 반복되는 HTML 속성이 입력 공간을 사용하므로 검색과 답변 입력을 나눠 점검해야 합니다.

R1에서는 현재 점수 숫자를 생성하지 않았지만, 실시간 조회 불가에 대한 설명과 주변 문구 전체를 정식 채점한 것은 아닙니다. I1은 세 프로필 모두 p23을 top-5에서 찾지 못했습니다. 따라서 이 실험에서 p23의 잘못 인식된 날짜가 답변으로 전파됐다고 주장하지 않습니다. [파싱 보고서](analysis_review.md)에서 확인한 날짜 오인식은 여전히 VL 코퍼스에 있습니다.

## 이미지의 현재 범위

코퍼스는 Markdown에서 참조한 추출 이미지의 경로와 SHA-256을 `images`에 보관합니다. 페이지의 청크는 같은 이미지 목록을 공유합니다. **검색은 텍스트 임베딩이며 Qwen에 이미지 파일·픽셀을 전달하지 않습니다.**

세 코퍼스·인덱스의 문서 수와 1,024차원 벡터, 이미지 116개의 경로·해시를 검증했습니다. 기존 19개, 새 Structure 22개, VL 75개이며 이것은 이미지 인식 정확도 점수가 아닙니다.

VL이 그림에서 인식해 Markdown으로 만든 글자·표는 RAG에 들어갑니다. 이미지로만 남은 부분은 Qwen이 읽을 수 없습니다. 작은 화면 캡처에서 잘못 추출한 텍스트도 들어 있으므로 이미지 경로 연결을 이미지 이해로 설명하지 않습니다.

## 실행

저장소 루트 PowerShell에서 실행합니다.

```powershell
# 지금 준비된 VL 인덱스로 질문
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run-rag.ps1 -Profile vl

# 단일 질문. --search-only를 붙이면 검색만 확인
& .\.venv\Scripts\python.exe -X utf8 -B .\src\rag.py 'Spec Monitoring에서 이슈 필터를 초기화하려면 어떻게 하나요?' --profile vl

# 코퍼스·인덱스가 필요한 새 PC
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-rag.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-local-llm.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-rag.ps1 -Profile vl
```

`run-rag.cmd`를 더블클릭하면 VL·GPU의 Streamlit 화면이 열립니다. 기존 콘솔은 `-Console -Profile existing`, 새 Structure는 `-Profile structure`로 선택합니다.

후속 개선은 범례를 JSON에서 보존하는 출력 정책, 표 HTML을 입력에 맞게 정리하는 방법, 기호·범례 해석과 출처 검증부터 다룹니다. 작은 캡처와 검색 누락도 별도로 평가합니다. 이번에는 원래 출력과 기본 검색 조건으로 연결·비교했으며, 이 실패에 맞춘 보정이나 질문 변경은 적용하지 않았습니다.
