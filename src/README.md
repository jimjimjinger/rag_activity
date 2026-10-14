# src Python 코드 안내

전체 구성·플로우차트·실행 명령은 [프로젝트 README](../README.md)를 참고하세요.
문서 준비 순서는 `parse_pdf.py → build_corpus.py → rag.py --build`, 평소 질문 흐름은 `rag.py → llm_client.py`입니다.

| 파일 | 담당 작업 | 실행 시점 |
|---|---|---|
| `parse_pdf.py` | PDF 분석·Markdown·추출 이미지 저장 | PDF가 바뀌었을 때 |
| `build_corpus.py` | 기능 항목별 청킹, 출처·이미지 경로를 JSONL에 저장 | Markdown·청킹·이미지 연결이 바뀌었을 때 |
| `rag.py` | 임베딩·FAISS 검색, 답변과 원본 출처 안내 | 인덱스 구축 또는 질문할 때 |
| `llm_client.py` | 프롬프트 구성, Qwen 답변 스트리밍 | 답변을 생성할 때 |

## 1. parse_pdf.py — PDF 분석

**입력:** `--pdf` PDF 경로, `--output` 결과 폴더. 생략하면 현재 매뉴얼과 기본 폴더를 사용합니다.

**출력:** 페이지별 Markdown·Paddle 추출 이미지, `run_summary.json`.

1. 원본 PDF의 경로와 해시를 확인합니다.
2. `PPStructureV3(**settings)`로 문서 분석 파이프라인을 만듭니다.
3. `pipeline.predict(input=str(pdf_path))`로 PDF의 OCR·표·배치 분석을 실행합니다.
4. 반환된 페이지 번호와 전체 페이지 수를 확인하며 Markdown을 저장합니다.
5. 전체 페이지가 순서대로 처리됐는지 확인하고 완료 기록을 남깁니다.

| 주요 모델 | 역할 |
|---|---|
| PP-DocLayout-S | 본문·제목·표·그림 영역 검출 |
| PP-OCRv5_mobile_det | 글자 영역 검출 |
| en_PP-OCRv5_mobile_rec | 영어 글자 인식 |
| SLANet_plus | 표 구조 인식 설정 |

PDF 렌더링은 Paddle 내부에서 수행합니다.
Paddle이 그림 영역을 Markdown과 함께 이미지 파일로 저장할 수 있습니다.
수식·차트 수치 변환·도장 인식은 비활성화되어 있습니다. 매번 PDF 전체를 분석합니다.

## 2. build_corpus.py — 항목별 청크 생성

**입력:** 파싱 완료 기록, 해당 PDF, 페이지별 Markdown, 원본 대조 보정 파일.

**출력:** `data/rag/corpus.jsonl`. 한 줄이 청크 하나입니다.

### split_page(content)

현재 매뉴얼의 추출 형식에 맞춘 규칙 기반 함수입니다.

1. HTML 또는 Markdown 표가 있으면 페이지 전체를 유지합니다.
2. 명시적인 `## 번호 제목`을 찾고, 그 지점부터 같은 수준의 `## 제목`들을 분리 후보로 삼습니다. 번호는 3이나 5에서 시작해도 됩니다.
3. 후보가 두 개 미만이면 페이지 전체를 유지합니다.
4. 번호가 붙어 깨진 제목, 번호의 누락·중복, 제목만 있고 설명이 없는 경우에는 페이지 전체를 유지합니다.
5. 첫 항목 이전의 메뉴 경로·상위 제목·화면 소개를 공통 문맥으로 사용합니다.
6. 각 항목의 제목·본문 앞에 공통 문맥을 붙입니다.

본문의 일반 번호 목록이나 더 깊은 Markdown 제목은 새로운 청크로 나누지 않습니다.
예를 들어 한 기능 아래의 `1. 선택 → 2. 입력 → 3. 실행`은 함께 남습니다.
같은 수준의 제목으로 추출된 절차까지 의미적으로 구별하는 기능은 없으므로 다른 문서에서는 규칙을 점검해야 합니다.

20페이지는 다음 세 청크로 분리됩니다.

| ID | 항목 | 함께 포함되는 공통 문맥 |
|---|---|---|
| p20-s1 | 1 Business Unit and Country | Spec Monitoring 메뉴 경로·화면 소개 |
| p20-s2 | 2 Issue Filter | 동일 |
| p20-s3 | 3 Page Details | 동일 |

9페이지의 `3Suggested Questions`는 원본 대조 후 `## 3 Suggested Questions`와 본문으로 보정되어 3개 청크로 분리됩니다.
37~39페이지의 표는 페이지 전체를 유지합니다. 현재 결과는 **42페이지 / 73개 청크**입니다.
의미를 판단하는 별도 AI 모델은 사용하지 않습니다.

### apply_corrections(content, page, corrections)

`data/rag/markdown_corrections.json`의 보정을 메모리에서 적용한 뒤 `split_page()`에 전달합니다.
Paddle의 원래 Markdown 파일을 덮어쓰지 않습니다. 번호·제목의 줄바꿈 복구와 원본에서 확인된 누락 문장만 보완합니다.
각 보정에는 변경 전·후 문자열과 검토 당시 Markdown 해시가 있습니다. 다른 버전의 추출 결과에 보정을 적용하지 않도록 검사합니다.
보정 파일의 PDF 해시도 검사하므로 새 PDF나 재파싱 결과에서는 기록을 다시 검토해야 합니다.

| 원본 페이지 | 보정 후 항목별 청크 수 | 확인한 문제 |
|---|---|---|
| 9 | 3 | 3번 제목·본문이 붙음 |
| 11 | 3 | Page Type 설명 누락 |
| 15 | 2 | 3·4번 제목 경계, 공통 소개 누락 |
| 20 | 3 | Page Details의 3번 누락 |
| 22 | 2 | Page Type 설명 누락 |
| 23 | 4 | 3번 제목 형식, Download 설명 누락 |
| 24 | 2 | 5·6번 설명 누락, 번호가 5부터 시작 |
| 31 | 4 | 3·4번 제목 형식 |
| 32 | 2 | 5·6번 설명 누락, 번호가 5부터 시작 |

32페이지 `Add item`의 본문은 원본 자체가 DICT List/Change Log 안내입니다. 임의로 추가 기능 설명을 만들어 넣지 않았습니다.

### build()

원본 해시와 전체 페이지 분석 완료를 검사한 뒤 각 페이지에 `apply_corrections() → split_page()`를 적용합니다.
`page_images()`는 해당 Markdown이 참조한 Paddle 추출 이미지의 경로와 해시를 청크에 붙입니다.
현재 19개 이미지가 연결됩니다. 같은 페이지의 항목들은 동일한 이미지 목록을 공유하며 이미지가 없는 페이지는 빈 목록입니다.
임의의 과거 이미지 파일을 폴더에서 찾아 붙이지 않으며, 페이지 밖 경로나 외부 URL은 연결하지 않습니다.
모든 페이지를 성공적으로 읽은 후 임시 파일을 최종 코퍼스로 교체합니다.

| 청크 필드 | 의미 |
|---|---|
| `id` | 페이지·항목 순서 기반 ID. 예: p20-s2 |
| `page` | 원본 PDF 페이지 번호, 1부터 시작 |
| `section` | 기능 제목. 페이지 전체라면 Page |
| `chunk_type` | section 또는 page |
| `content` | 공통 문맥과 항목 본문 또는 페이지 전체 |
| `source_markdown`, `source_pdf` | 프로젝트 기준 상대 경로 |
| `source_sha256` | 원본 PDF 변경 확인용 해시 |
| `origin` | PP-StructureV3 Markdown, 보정한 경우 PDF-verified corrections 표시 |
| `correction_source` | 보정 기록 경로. 보정하지 않은 페이지는 null |
| `images` | Paddle 추출 이미지별 `path`(상대 경로), `sha256`. 임베딩하지 않는 metadata |

표를 BeautifulSoup으로 다시 해석하거나 pdfplumber 결과와 비교하지 않습니다.
출처와 상위 메뉴의 정보가 청크에 남도록 하는 것이 핵심입니다.

## 3. rag.py — LangChain FAISS로 임베딩·검색

**입력:** 코퍼스 또는 저장된 인덱스, 사용자 질문.

**출력:** 인덱스 파일 또는 터미널의 검색 근거·원본 PDF 경로·페이지·답변.

### Document와 임베딩

`load_documents()`가 코퍼스의 각 행을 다음 객체로 만듭니다.

```python
Document(
    id=chunk["id"],
    page_content=chunk["content"],
    metadata={...},  # 페이지, 항목, 출처 등
)
```

`Document`는 파이썬 객체입니다. `page_content`가 임베딩 입력이고 metadata는 출처 표시 등에 사용합니다.
`embeddings()`는 `OllamaEmbeddings(model="bge-m3:567m", ...)` 연결 객체를 만듭니다.
실제 문서·질문 벡터 계산은 Ollama에서 실행하는 BGE-M3가 합니다.

### build_index() — 인덱스 생성·저장

```python
store = FAISS.from_documents(documents, embeddings(), normalize_L2=True)
INDEX.write_bytes(store.serialize_to_bytes())
```

LangChain FAISS 연동이 본문 임베딩을 요청하고, 문서 벡터를 정규화해 인덱스를 만듭니다.
`data/rag/index.pkl` 하나에 벡터 인덱스·Document·벡터 ID 연결을 함께 저장합니다.
Windows의 한글 경로에서 FAISS 기본 파일 저장 함수가 오류를 내므로, LangChain의 바이트 직렬화 기능을 사용합니다.
인덱스 변경 검사, 점수 환산, 기존 벡터 재사용은 구현하지 않습니다. 코퍼스·임베딩 모델을 바꾸면 다시 구축합니다.

### load_retriever() / retrieve() — 불러오기·검색

```python
store = FAISS.deserialize_from_bytes(
    INDEX.read_bytes(), embeddings(), normalize_L2=True,
    allow_dangerous_deserialization=True,
)
retriever = store.as_retriever(search_kwargs={"k": top_k})
documents = retriever.invoke(question)
```

`deserialize_from_bytes()`는 이 PC에서 직접 생성한 인덱스만 불러옵니다. 저장 파일은 Git에서 제외하며 다른 PC에서는 다시 만듭니다.
불러올 때도 `normalize_L2=True`를 지정해야 질문 벡터가 정규화됩니다.

`invoke()`가 BGE-M3 질문 임베딩 → 정규화 → FAISS 검색 → 상위 k개 Document 반환을 수행합니다.
기본 k는 3이며 `--top-k`로 바꿉니다. 정규화된 벡터의 L2 거리 순위는 코사인 유사도 순위와 같습니다.
직접 작성한 검색기 클래스나 유사도 계산 코드는 없습니다.
연동 클래스는 `langchain_community.vectorstores.FAISS`이며 실제 벡터 검색은 FAISS가 합니다.

### 근거와 실행

| 함수 | 역할 |
|---|---|
| `build_context()` | 검색 본문 앞에 페이지 출처를 붙여 연결하고 Qwen 입력 길이에 맞춤 |
| `ask()` | 검색 근거·원본 출처 표시 → Qwen 답변 호출 |
| `main()` | 인덱스 구축, 단일 질문, 대화형 입력 선택 |

`--search-only`는 Qwen을 호출하지 않습니다.
검색 결과에는 원본 PDF 경로와 근거 페이지를 표시합니다. 질문·답변 기록이나 HTML 파일은 저장하지 않습니다.
Qwen에는 본문만 전달하며 Paddle 추출 이미지는 출처 metadata로 연결합니다.
질문마다 독립적으로 처리하며 대화 기억, 리랭커, 자동 페이지 확장은 추가하지 않았습니다.

## 4. llm_client.py — 프롬프트와 Qwen

`PROMPT`는 LangChain `ChatPromptTemplate`입니다.

- 시스템 메시지: 한국어, 문서 근거, 원문 메뉴명, 페이지 출처, 근거 부족 시 한계 설명.
- 사용자 메시지: 검색된 본문을 `{context}`, 현재 질문을 `{question}`으로 전달.
- 문서 안의 지시는 자료로 취급하고 매뉴얼에 없는 현재 점수나 클릭 경로를 만들지 않도록 지시.

`chat(question, context)`는 다음 체인을 실행합니다.

```python
chain = PROMPT | llm | StrOutputParser()
for text in chain.stream({"question": question, "context": context}):
    print(text, end="", flush=True)
```

여기서 `llm`은 `ChatOllama(model="qwen3:4b-instruct", ...)`입니다.
CPU 4스레드, 문맥 4,096토큰, 출력 최대 256토큰, temperature 0.2를 사용합니다.
답변을 스트리밍하고 완성된 문자열도 반환합니다. 빈 답변은 오류로 안내합니다.

통신 주소는 같은 PC의 `http://127.0.0.1:11435`입니다.
로컬 요청은 시스템 프록시를 거치지 않으며 LangSmith 추적도 비활성화합니다.
이 파일은 PDF를 읽거나 검색하지 않습니다.

## 분석 비교 및 환경

`tests/`에는 기존 PP-StructureV3와 PaddleOCR-VL의 분석 결과를 따로 저장합니다.
수동 보정 전 Markdown을 비교하고, 같은 `split_page()` 규칙을 적용한 청킹 결과도 별도로 만듭니다.
실험 결과는 현재 RAG의 코퍼스·인덱스에 반영하지 않습니다. 실행 방법은 [비교 폴더 안내](../tests/README.md)에 있습니다.

RAG에는 `requirements-rag-lock.txt`, Paddle에는 `requirements-parser-lock.txt`를 사용합니다.
설치와 실행 명령은 [프로젝트 README](../README.md)에 있습니다.
