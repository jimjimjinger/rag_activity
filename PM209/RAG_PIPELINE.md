# PM209 청킹·임베딩·검색·답변 실행

2026-11-02, `answer`·`search`를 **LangChain + Qdrant 내부 dense/BM25 검색**으로 전환했다. 한국어 질문을 영어로 번역해 검색하고, Qwen3.5-4B가 원래 한국어 질문과 검색 근거로 답한다. 현재 경로는 `src/lc_qdrant_rag.py`와 `scripts/langchain_rag.py`다. 채팅 화면과 실제 그림 표시는 아직 구현하지 않았다. 같은 날 앞서 수행한 SQLite 키워드 검색·Python 결합 실험은 아래의 이전 실험 기록으로 보존한다.

전체 인덱스 보존 검증과 30문항 × 5개 비중 × Top-1/5의 새 답변 300개 검수를 마쳤다. 이번 표본의 최고는 키워드 0%·Top-5의 20/30개이며 텍스트 질문은 9/10개다. [청킹·임베딩·검색·답변 핵심 정리와 체크리스트](reports/PM209_RAG_PIPELINE_SUMMARY_20261102_v2.md), [문항별 답변·근거·검수](reports/qa30_lc_qdrant_20261102/report.html), [시스템 검증](reports/qa30_lc_qdrant_20261102/system-validation.json)을 참고한다. Qdrant 내부의 가중 RRF와 BM25 처리 방식이 이전 외부 결합과 달라 순수 프레임워크 효과로 볼 수 없다.

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
| `search` | LangChain 번역·질문 임베딩 → Qdrant dense/BM25·내부 가중 RRF | 터미널 결과, 새 인덱스의 `answers/*.json` |
| `answer` | 같은 검색 → LangChain ChatPromptTemplate·ChatOllama → 근거 ID 검증 | 터미널 답변, 새 인덱스의 `answers/*.json` |
| `evaluate` | 저장된 인덱스를 QA 질문·근거 ID로 평가 | `retrieval-evaluation.json` |

청킹만 실행하면 임베딩과 DB 저장은 발생하지 않는다. `sample` 명령은 기존 청킹·임베딩·저장·검색 평가를 순서대로 연결하며 답변 생성은 포함하지 않는다. 현재 `answer`는 LangChain RunnableSequence로 번역→검색→답변을 연결하고, `QdrantVectorStore`가 실제 Qdrant 검색을 호출한다. 기존 준비 명령은 `.runtime/venv-rag`, 새 검색·답변은 별도 `.runtime/venv-rag-lc`를 사용한다.

Qdrant는 Python 라이브러리의 **로컬 디스크 모드**다. 별도 DB 서버·Docker·계정·포트 설정 없이 사용한다. 테스트와 개발용 구성으로, 전체 대규모 데이터 및 여러 사용자의 동시 접속에는 Qdrant 서버 모드를 별도로 검토한다. 같은 로컬 DB 경로를 사용하는 명령은 동시에 실행하지 않는다. [공식 Qdrant Python 클라이언트](https://github.com/qdrant/qdrant-client)

## 실행

한국어 답변을 받으려면 저장소 루트에서 다음 명령을 실행한다. 대상은 `datasets/PM209/rag/full_char500_lc_qdrant_v1`의 **126,938개 dense+BM25 인덱스**이며 지정한 매뉴얼만 검색한다. 기존 문서 벡터는 SQLite 캐시에서 읽기 전용으로 재사용한다. 새 질문 벡터는 `.runtime/pm209-lc-qdrant/query-cache`의 JSON에 저장한다. SQLite FTS 키워드 검색은 현재 경로에서 사용하지 않는다.

```powershell
.\run-pm209-rag.cmd answer --manual "alienware-17-r5_users-guide_en-us" --query "이 노트북의 무게와 크기는 어떻게 되나요?" --keyword-percent 50 --top-k 5

# 답변 생성 없이 같은 검색 결과 확인
.\run-pm209-rag.cmd search --manual "alienware-17-r5_users-guide_en-us" --query "무게는 얼마인가요?" --keyword-percent 0 --top-k 1
```

기본값은 `--keyword-percent 50 --top-k 5`이며 최적값으로 확정한 설정은 아니다. 비중은 0~100, Top-k는 1~20이다. `--no-translate`로 번역을 끌 수 있다. 0%는 dense만, 100%는 BM25만 검색한다. 중간 비중은 Qdrant의 `RrfQuery(k=60, weights=[1-w,w])`를 사용한다. 각 분기의 후보 20개를 합친 뒤 최종 Top-k만 전달한다. 가중치는 원시 유사도나 답변 수의 비율이 아니다.

BM25는 LangChain `FastEmbedSparse(Qdrant/bm25)`로 문서·질문 희소 벡터를 만들고, Qdrant의 `Modifier.IDF`를 적용한다. 영어 형태 정규화·불용어 처리를 사용하며 `k=1.2, b=0.75, avg_len=256`은 모델 기본 설정이다(실제 코퍼스 평균 길이가 아니다). 양쪽 Prefetch에 같은 매뉴얼 필터를 적용한다. Qdrant 1.20의 가중 점수는 순위 r(1부터)에 대해 `1/(r/w+k-1)`이며 예전 Python의 `w/(60+r)`와 다르다.

임베딩·답변 모델 이름과 digest를 고정한다. 기존 문서 벡터와 동일한 질문 지시문·옵션·`truncate=false`를 유지하며 8GB GPU에서 질문 임베딩 모델을 내린 다음 답변 모델을 사용한다. 로컬 Qdrant는 실행마다 전체 인덱스를 여는 비용이 있고 같은 경로를 동시에 열 수 없다. 새 인덱스의 준비·재개는 `scripts/build_lc_qdrant.py`, 환경 설치는 `scripts/setup-rag-lc.ps1`로 수행한다. 기존 `full_char500_v1`은 보존한다.

답변 프롬프트·스키마·검증은 기존 `src/answering.py`를 그대로 재사용한다. LangChain 연결은 `src/lc_qdrant_rag.py`, 명령·저장은 `scripts/langchain_rag.py`에 있다. 한국어로 답하고 숫자·단위·조건을 보존하며, 근거가 부족하면 확인하지 못했다고 답하게 한다. 모델 입력은 사용자 질문과 검색 청크의 텍스트·유형·검수 표시·짧은 근거 ID만 포함한다. 원본 QA와 이미지 바이트는 넣지 않는다.

응답은 `answer`, `source_ids`, `insufficient_evidence`로 받는다. `S1` 등의 ID는 실제 청크·페이지·영역으로 코드가 다시 연결하며 모델이 파일 경로나 좌표를 만들지 않는다. `citations`에 채택한 근거만 저장하고, `cited_figures`에는 인용된 그림 청크의 본문 영역 ID·JPG 경로·좌표만 남긴다. 인접 그림이나 제목 영역을 자동으로 그림 근거로 붙이지 않는다. 이 메타데이터를 다음 UI 단계에서 사용할 수 있다.

답변 설정은 문맥 8,192토큰·출력 최대 1,024토큰·`think=false`·temperature 0·요청 제한 300초다. Qwen3.5의 byte-BPE 특성에 따라 UTF-8 바이트 수와 템플릿 여유 512를 입력 토큰 상한으로 보수적으로 잡고, 출력 예산도 따로 확보한다. 초과 청크는 자르지 않고 통째로 제외해 기록한다. 질문은 UTF-8 2,000바이트 이내다. 형식·종료 사유·한국어·근거 ID 검사 실패는 간결한 재시도로 한 번 더 확인하며, 계속 실패하면 정상 답변으로 저장하지 않는다. 모델·서버·통신 오류는 자동 반복하지 않는다.

각 성공 결과는 새 인덱스의 `answers/<시각>-<고유값>.json`에 남는다. `--save`로 출력 경로를 지정할 수 있다. 실패 시 오류를 출력하고 종료 코드 1을 반환한다. 유효한 근거 ID는 의미 정확도 보증이 아니며, OCR 오인식·검색 누락·답변의 오역은 별도 검수가 필요하다. API 형식은 [Ollama Chat](https://docs.ollama.com/api/chat), [구조화 출력](https://docs.ollama.com/capabilities/structured-outputs)을 따른다.

### 이전 번역·키워드 검색 비교 실험 — SQLite/Python 경로

아래는 전환 전 실험 기록이다. 현재 `run-pm209-rag.cmd answer/search`의 Qdrant 내부 검색과 구분한다. 새 비교는 `scripts/run_lc_qdrant_comparison.py` 및 `reports/qa30_lc_qdrant_20261102`에 저장한다. 새 실험은 같은 30개 질문·영어 번역·답변 프롬프트를 고정하고 5개 비중 × Top-1/5의 답변 300개를 생성한다. 평가 정답은 모델 입력에 넣지 않는다.

`scripts/run_retrieval_comparison.py`는 고정한 30개 질문으로 기존 한국어 의미 검색(A), 영어 번역 후 의미 검색(B), 같은 영어 번역에 의미+BM25 검색을 적용한 경우(C)를 비교한다. `answer`의 기본 검색 설정을 바꾸는 명령은 아니다. 질문만 로컬 Qwen에 번역시키며 답변 생성에는 원래 한국어 질문과 검색 근거를 사용한다.

BM25는 기존 청크만 `.runtime/pm209-hybrid/bm25-v1.sqlite3`에 색인한다. C는 양쪽 후보 20개씩을 RRF 가중치 **0.5:0.5**, 상수 60으로 합쳐 최종 5개를 전달한다. 기존 문서 벡터를 다시 만들지 않는다. 자동 번역 출력과 실험 조건은 검색 전에 고정하며, 번역 자체의 오류도 평가에 남긴다. 실행 명령·평가 범위·결과 위치는 [사전 실험 조건](reports/qa30_hybrid_20261102/PROTOCOL.md)에 정리했다.

이번 30문항의 원본 기준 정답은 A 17개, B 20개, C 18개였다. B/C에 출력 형식 검증 실패가 각각 1개 있으며 분모 30개에 포함했다. 50:50 결합은 일부 정답 표를 최종 5개 밖으로 밀어냈고, 이번 표본에서는 번역만 적용한 B가 높았다. [문항별 비교 보고서](reports/qa30_hybrid_20261102/REPORT.md)와 [비교 화면](reports/qa30_hybrid_20261102/report.html)을 참고한다. 이 결과는 별도 미사용 질문으로 검증한 일반 성능 수치가 아니다.

추가로 영어 번역과 검색 후보를 고정하고 키워드 비중을 **0·25·50·75·100%**로 비교했다. 정답은 순서대로 **20·17·18·15·13개 / 30개**였다. 0%와 50%는 B/C를 재사용하고, 25%·75%·100%는 `scripts/run_weight_sweep.py --keyword-percent 25 75 100`으로 90개 답변을 새로 생성했다. 추가 실행은 약 2분 45초였고 출력 형식 오류 2개도 평가 분모에 포함했다. 가중치는 원시 유사도나 청크 개수의 비율이 아닌 RRF 순위 결합 비중이다. 이번 표본에서는 영어 의미 검색이 가장 높았으며 기본 `answer` 설정은 유지했다. [가중치별 보고서](reports/qa30_weights_20261102/REPORT.md), [30문항·5조건 비교 화면](reports/qa30_weights_20261102/report.html)에 실제 답변·판정 이유·검색 근거를 보관한다.

같은 조건에서 **Top-1 청크 하나만** 넣은 150개 답변도 추가 비교했다. 키워드 비중 0·25·50·75·100%의 정답은 각각 **15·14·14·12·11개 / 30개**로, 모든 조건에서 기존 Top-5보다 적었다. 실행은 `scripts/run_top1_comparison.py`로 약 3분 3초였고 출력 형식 오류 1개도 분모에 포함했다. 검색 영역 Hit@1은 23·23·22·17·16개였지만, 제목만 찾거나 표의 조건·수치가 다른 청크에 있어 답변 정답률과 차이가 났다. 설정·원본·Top-5 결과는 유지했으며, [Top-1 보고서](reports/qa30_top1_20261102/REPORT.md)와 [Top-1/Top-5 비교 화면](reports/qa30_top1_20261102/report.html)에 근거·답변·검수 사유와 동일 입력의 출력 변동을 보관한다.

이 중 **본문 분류 10문항(TEXT01–TEXT10)**만 따로 집계하면 같은 가중치 순서의 정답은 Top-1 **8·8·8·7·6개**, Top-5 **9·9·9·8·8개 / 10개**다. 기존 실제 답변 100개를 재검수했으며 새 모델 호출은 없었다. Top-5 0%의 나머지 한 개는 오답, 25·50%는 부분정답이므로 완전 정답 수와 오류 정도를 구분한다. 기존 분류를 유지한 부분집합 분석이며, 검색을 본문으로만 제한하지는 않았다. 특히 입력 전압 문항은 표 청크의 수치를 사용한다. [본문 10문항 보고서](reports/qa10_text_20261102/REPORT.md)와 [비교 화면](reports/qa10_text_20261102/report.html)을 참고한다.

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
