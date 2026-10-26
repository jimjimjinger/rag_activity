# PM209 제품 매뉴얼 RAG

PM209가 제공하는 본문·제목·영역 좌표를 재사용하고, 그림 설명과 표 내용을 추가해 제품 매뉴얼 RAG를 준비합니다. 그림 description 23,807개와 Paddle 표 HTML 2,308개를 저장했고, 남은 12개에는 원본 이미지를 보고 작성한 검색용 HTML을 별도 보강본에 추가했습니다. 이 12개는 일부 내용을 줄여 적었거나 아이콘이 빠져 있어 정확도 재검수가 필요합니다. 2026-10-26에는 샘플 검색을 확인한 뒤 전체 209개·10,018페이지의 126,938개 청크를 GPU로 임베딩하고 Qdrant 저장·검증까지 완료했습니다. [전체 완료 기록](PM209/reports/RAG_FULL_RUN_20261026.md)에 결과가 있습니다. [샘플 결과와 한계](PM209/reports/RAG_SAMPLE_20261026.md), [실행 방법](PM209/RAG_PIPELINE.md)을 참고하세요.

기존 Dashboard 매뉴얼·Paddle 비교 결과·BGE-M3/FAISS RAG·Streamlit 실행 파일은 정리했습니다. 이전 파일은 Git 이력에서 확인할 수 있습니다.

## 현재 구성

```text
rag_activity/
├── README.md
├── PM209/
│   ├── README.md
│   ├── PM209_DATA_SPEC.md
│   ├── PM209_IMAGE_RETRIEVAL_COMPARISON.md
│   ├── requirements-images.txt
│   ├── requirements-rag.txt
│   ├── RAG_PIPELINE.md               청킹·임베딩·Qdrant 실행 안내
│   ├── src/                          그림 설명·표 저장·HTML 보완
│   ├── scripts/                      모델 실행·표 재인식·결과 복원
│   ├── tests/                        데이터 보존·복구 회귀 검사
│   ├── reports/                      샘플 이미지·검수·최종 검증
│   ├── results/                      별도 전달 압축본 안내, 압축 파일은 Git 제외
│   └── datasets/PM209/               실행용 데이터·생성 결과·벡터 DB, Git 제외
│       ├── images/                    페이지 JPG 10,018장
│       ├── data/                      원본 train·val·test JSONL
│       │   ├── vlm/                  그림 설명 보강본
│       │   ├── paddle_tables_html/   첫 표 추출·검수 이미지·실패 원시 결과
│       │   ├── paddle_tables_html_repaired/  실패 보완본·재인식 후보
│       │   └── paddle_tables_manual_20261026/  남은 12개 수기 검색용 보강본
│       ├── meta_data/                 공식 메타데이터 3개
│       ├── PM209.zip
│       └── download_manifest.json
├── .venv/                             기존 Python 환경, Git 제외
└── .runtime/                          기존 모델·실행 환경, Git 제외
```

[작업 방법](PM209/README.md), [데이터 명세](PM209/PM209_DATA_SPEC.md), [현재 설계와 GPU 작업](PM209/PM209_IMAGE_RETRIEVAL_COMPARISON.md)을 참고하세요.

## 문서 처리 방향

1. **파싱 보강:** 기존 그림 좌표로 이미지를 잘라 Qwen3.5-4B로 영어 설명을 생성하고, Table 영역에는 PaddleOCR의 OCR·구조 인식으로 HTML을 추가합니다.
2. **청킹:** 본문·그림·표를 공백 포함 500자, 최대 100자 겹침 기준으로 전체 처리했습니다. 표는 행 경계·확인 가능한 헤더 반복을 사용하며 구조가 불확실하면 표시합니다. 긴 표 문맥 26개는 길이 예외로 보존했습니다.
3. **임베딩·저장:** Qwen3-Embedding-0.6B와 Qdrant 로컬 모드로 전체 126,938개 벡터를 저장하고 검증했습니다. 별도 DB 서버는 필요하지 않습니다.
4. **검색:** 선택한 매뉴얼로 필터링합니다. 영어 QA 샘플 평가를 수행했으며 한국어 검색에는 누락·순위 문제가 남습니다.
5. **답변:** 후속 단계에서 한국어 답변과 근거 그림 표시를 연결할 계획입니다.

보강본은 원본과 같은 페이지 단위 JSONL이며, 그림 객체에 `description`, Table 객체에 `table_content`를 추가합니다. 기존 본문·좌표·QA는 유지합니다. QA 질문·정답은 보강 모델 입력에 넣지 않습니다. 추출 성공 수는 의미 정확도를 보장하지 않으며, 온도 기호·아이콘 등의 OCR 오류가 남습니다.

## 데이터와 메타데이터

데이터는 이전 `회사과제/pm205`에서 `PM209/datasets/PM209`로 옮겼습니다. 공식 ZIP에 메타데이터가 없어 [공식 저장소의 meta_data](https://github.com/AIM3-RUC/MPMQA/tree/master/meta_data)에서 별도로 추가했습니다.

`name2meta.json`은 209개 매뉴얼 ID를 브랜드·제품 종류에 연결합니다. 제품 선택과 필터링에 사용하며, 그림 설명 생성 자체의 필수 입력은 아닙니다. 다른 두 메타데이터 파일은 브랜드·제품별 개수 집계입니다.

Git에는 실행 코드·설정·문서·샘플 검수 결과와 전체 실행 요약을 저장합니다. 원본 ZIP·전체 페이지 JPG·보강 JSONL·벡터 DB·가상환경·모델과 약 545MiB의 [분할 압축본](PM209/results/README.md)은 로컬에 보관하며 Git에서 제외합니다. Git clone만으로 전체 생성 결과가 내려오지는 않습니다.

별도 전달받은 2026-10-22 분할 압축본과 manifest를 `PM209/results/20261022/`에 둔 경우, 저장소 루트에서 Python 3.10 이상으로 복원할 수 있습니다. 이 압축본은 당시 그림·표 결과이며, 이후 추가한 12개 수기 보강본과 2026-10-26 청킹·임베딩 결과는 포함하지 않습니다.

```powershell
python PM209/scripts/result_bundle.py restore
```

복원 후 `PM209/datasets/PM209/data/paddle_tables_html_repaired/review_index.html`을 열면 표 이미지와 결과를 비교할 수 있습니다. 이미 있는 파일은 내용이 같은 경우 재사용하며, 다른 경우 덮어쓰지 않습니다. 원본 전체 페이지 이미지로 재실행하려면 [공식 배포 안내](https://github.com/AIM3-RUC/MPMQA)에 따라 원본 데이터도 준비합니다.
