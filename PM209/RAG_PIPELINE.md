# PM209 청킹·임베딩·Qdrant 실행

2026-10-26, 매뉴얼 3개의 **48페이지 → 646개 청크 → 1,024차원 벡터 → Qdrant 로컬 저장 → 검색**을 먼저 확인한 뒤, **전체 209개 매뉴얼의 126,938개 벡터 저장까지 완료했다.** [전체 완료 기록](reports/RAG_FULL_RUN_20261026.md)과 [샘플 검색 평가·한계](reports/RAG_SAMPLE_20261026.md)를 구분해 참고한다.

같은 날 **20:20 KST 시작해 22:12 KST 완료**했다. 전체 209개·10,018페이지의 126,938개 청크를 GPU로 임베딩한 뒤 Qdrant에 저장하고 모든 payload·벡터와 원본/보강본 9개 파일 해시를 검증했다. `.runtime/pm209-rag-full/status.json`과 `final-validation.json`은 complete이며 실패는 0개다. 답변 생성과 전체 검색 품질 평가는 별도다.

전체 출력은 `PM209/datasets/PM209/rag/full_char500_v1/`이며 기존 샘플을 유지한다. 원본 그림 23,807개·표 2,320개 연결을 사전 확인했다. 제목/표 문맥이 긴 26개 청크는 기존 예외 규칙에 따라 500자 초과 상태로 보존하고 검수 표시했다(최대 1,776자, 전체 최대 500토큰). QA는 청크에 포함하지 않는다. 전체 검수 화면은 매뉴얼별로 나누어 `review.html`에서 연결한다.

장시간 실행 명령은 다음과 같다. **현재 작업은 완료되었으므로 조회만 하려면 재실행할 필요가 없다.** 실행기는 잠금으로 중복 작업을 막고 배치별 임베딩 캐시를 저장한다. `.runtime/pm209-rag-full/STOP` 파일을 만들면 현재 배치 저장 후 멈추며, 명시적으로 해당 파일을 제거한 뒤 같은 명령을 실행하면 캐시와 동일 ID upsert를 재사용한다. 실패는 `status.json`과 로그에 기록한다. 전체 실행기는 QA 평가나 답변 생성을 자동 실행하지 않는다.

```powershell
.\.runtime\venv-rag\Scripts\python.exe -X utf8 -u PM209/scripts/run_rag_full.py --all
```

## 단계와 파일

| 단계 | 역할 | 결과 |
| --- | --- | --- |
| `chunk` | 본문·그림 설명·표 HTML을 500자 기준으로 나누기 | `chunks.jsonl`, `chunks-manifest.json`, `review.html` |
| `embed` | Qwen3-Embedding-0.6B로 문장을 벡터로 변환 | `embedding-cache.sqlite3`, `embedding-report.json` |
| `index` | 벡터·청크 내용·출처를 Qdrant에 저장 | `qdrant/`, `index-report.json` |
| `search` | 질문 임베딩 후 선택한 매뉴얼에서 코사인 유사도 검색 | 터미널 결과, `last-search.json` |
| `evaluate` | 저장된 인덱스를 QA 질문·근거 ID로 평가 | `retrieval-evaluation.json` |

청킹만 실행하면 임베딩과 DB 저장은 발생하지 않는다. `sample` 명령이 앞의 세 단계와 평가를 순서대로 연결한다. 이 연결은 직접 작성한 Python 실행기이며, 이번 검색 단계에 LangChain을 추가하지 않았다. 기존 그림 설명 생성의 ChatOllama 구성과는 별개다. 답변 생성 모델·채팅 화면·이미지 표시 UI는 후속 작업이다.

Qdrant는 Python 라이브러리의 **로컬 디스크 모드**다. 별도 DB 서버·Docker·계정·포트 설정 없이 사용한다. 테스트와 개발용 구성으로, 전체 대규모 데이터 및 여러 사용자의 동시 접속에는 Qdrant 서버 모드를 별도로 검토한다. 같은 로컬 DB 경로를 사용하는 명령은 동시에 실행하지 않는다. [공식 Qdrant Python 클라이언트](https://github.com/qdrant/qdrant-client)

## 실행

작업 위치는 저장소 루트 `rag_activity`다. 현재 PC의 환경은 준비되어 있다. 다음 명령은 **선정된 train 매뉴얼 3개만** 처리한다.

```powershell
.\run-pm209-rag.cmd chunk
.\run-pm209-rag.cmd embed
.\run-pm209-rag.cmd index
.\run-pm209-rag.cmd evaluate

# 위 네 단계를 순서대로 실행. 완료된 임베딩은 재사용한다.
.\run-pm209-rag.cmd sample

# 질문 검색. --manual은 정확한 매뉴얼 ID다.
.\run-pm209-rag.cmd search --manual "alienware-17-r5_users-guide_en-us" --query "이 노트북의 무게와 크기는 어떻게 되나요?" --top-k 5
```

다른 매뉴얼은 `--manual "ID"`를 지정하고 별도 `--output`을 사용한다. `--all`은 명시적으로 지정해야 하며 기본 샘플 출력 폴더에는 사용할 수 없다. 모든 매뉴얼은 train·val·test에 나뉘어 있으므로 개별 단계의 전체 처리 명령에는 split도 명시해야 한다. 위 명령은 샘플용이고, 2026-10-26 완료한 전체 실행은 문서 상단의 전용 실행기를 사용했다.

현재 결과 위치:

```text
PM209/datasets/PM209/rag/sample_char500_v1/
├── chunks.jsonl                 QA를 제외한 검색 자료
├── chunks-manifest.json         입력 해시·범위·길이·검수 표시 집계
├── review.html                  원본 페이지와 청크 비교
├── embedding-cache.sqlite3      모델 digest·입력 내용 기준 캐시
├── embedding-report.json        최근 임베딩 실행 기록
├── runs/                        최초 실행·재실행 기록
├── qdrant/                      디스크에 영속 저장한 벡터 DB
├── index-identity.json           청크 해시·모델 불일치 방지
├── index-report.json
├── retrieval-evaluation.json    영어 QA 검색 평가, 청크와 별도
└── validation.json              문자 보존·DB 대조·한국어 질문 3개
```

소스는 `data/paddle_tables_manual_20261026/{split}.jsonl`이며 읽기 전용으로 사용한다. 기존 그림 description·본문·표 HTML·ID·좌표·QA는 변경하지 않는다. `qa_data`는 청킹 함수에 전달하지 않는다. 평가에서만 질문과 근거 ID를 읽으며 정답 문자열은 사용하지 않는다. 평가 질문 벡터는 캐시에는 저장되지만 문서 컬렉션에는 들어가지 않는다.

## 적용한 청킹 규칙과 제한

- 검색 문자열은 공백·제목·반복 헤더를 포함해 500자를 기본 상한으로 삼는다. 전체 실행에서 긴 표 문맥 26개는 길이 예외로 보존하고 표시했다. 일반 연속 분할은 마지막 완전한 문장 최대 100자, 긴 문장은 구절·단어 경계에서 나누고 끝부분 최대 100자를 겹친다. 표는 앞 행을 겹치지 않는다.
- 페이지·매뉴얼을 넘겨 묶지 않는다. 같은 제목·같은 열에서 가까운 본문만 합치며 그림 설명은 그림별로 분리한다. 제목/열 연결은 좌표 휴리스틱이므로 원본 검수가 필요하다. 번호 단계도 원문 번호를 유지하지만 절차의 전제 조건을 의미적으로 추론해 붙이는 기능은 없다.
- 표 HTML을 파싱해 병합 셀 참조와 열 이름을 보존한다. 확실한 헤더는 반복하고 긴 표는 행별, 긴 행은 셀별로 나눈다. 헤더 태그가 없는 표는 첫 행을 `header unconfirmed` 문맥으로 반복하며 임의로 확정하지 않는다. 이 방식은 첫 행이 데이터인 경우에도 그 사실을 숨기지 않는다.
- 열 수/병합 관계가 불일치하면 셀 텍스트를 행 순서로 보존하고 `table_structure_unresolved`를 붙인다. 이 예외에서는 정확한 열 대응과 헤더 반복을 보장하지 않는다. 원본 HTML은 그대로 남아 있다.
- 표와 90% 이상 겹치는 OCR도 청크 문자열에 원문이 정확히 포함된 경우에만 중복 제거한다. 연결은 `deduplicated_region_ids`에 남긴다. 수기 요약 12개는 `manual_condensed_review_required`를 이어받고 주변 OCR을 제거하지 않는다.
- `source_spans`는 정규화한 OCR/그림 설명의 문자 범위, 표는 행·셀 좌표와 긴 셀의 분할 범위를 기록한다. 원본 OCR JSON의 바이트 위치는 아니다. 공간적으로 같은 영역의 부분 청크를 찾았다고 영역 전체 내용이 검색됐다고 볼 수는 없다.
- 제목 자체나 분할 불가능한 식별자가 지나치게 길면 삭제하지 않고 예외 표시한다. 샘플에는 500자 초과 예외가 없었다. 2,000토큰을 넘는 예외는 임베딩 전에 검토하도록 막는다.

## 모델·자원·재실행

별도 Python 환경 `.runtime/venv-rag`를 사용한다. GPU는 RTX 5060 Laptop, VRAM 8,151MiB다. Ollama의 `qwen3-embedding:0.6b`는 Q8_0, 샘플 실행의 벡터 차원은 1,024였다. 문서에는 추가 지시문을 붙이지 않고 질문에만 검색용 영어 지시문을 붙인다. [Qwen 모델 안내](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

기본 배치 4, 요청 동시성 1, 문맥 2,048토큰, `num_batch=128`, CPU 스레드 4다. `/api/embed`에 `truncate=false`를 사용하므로 모델 한계를 넘는 입력을 조용히 잘라 저장하지 않는다. 768 출력 토큰 설정은 임베딩에 적용되지 않는다. [Ollama 임베딩 API](https://docs.ollama.com/api/embed)

임베딩 캐시는 입력 문자열·모델 digest·옵션·문서/질문 용도로 구분한다. 배치 실패 시 개별 청크로 확인하고, 완료분은 즉시 저장한다. 오류가 남으면 DB 적재를 완료 처리하지 않는다. 재실행은 캐시를 읽고 Qdrant에 동일 UUID로 upsert하여 중복을 막는다. 모델이나 원본/청킹 결과가 바뀌면 기존 인덱스에 섞지 않고 새 출력 폴더를 사용한다.

새 환경을 준비할 때는 데이터와 기존 portable Ollama가 있는 저장소에서 실행한다. 이 스크립트는 Windows 드라이버나 시스템 Python을 변경하지 않는다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File PM209/scripts/setup-rag.ps1

# 청킹 회귀 검사 및 실제 저장 결과 검증
.\.runtime\venv-rag\Scripts\python.exe -X utf8 -m unittest discover -s PM209/tests -p test_chunking.py -v
.\.runtime\venv-rag\Scripts\python.exe -X utf8 PM209/scripts/validate_rag_sample.py
```

원본 데이터·보강 JSONL·모델·가상환경·샘플 및 전체 벡터 DB는 `.gitignore`에 따라 Git 대상에서 제외된다. Git에는 실행 코드·설정·이 실행 문서·요약 보고서와 작은 검수 샘플을 보관한다. 다른 PC에서는 입력 데이터와 실행 환경을 별도로 준비해야 한다.
