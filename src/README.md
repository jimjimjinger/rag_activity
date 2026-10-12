# src Python 코드 안내

전체 프레임워크와 흐름도, 실행 명령은 [프로젝트 README](../README.md)를 참고하세요. 현재 코드는 **Paddle Markdown을 그대로 페이지별 청크로 사용하는 최소 구성**입니다.

## 파일과 실행 시점

| 순서 | 파일 | 역할 |
|---|---|---|
| 문서 준비 1 | `parse_pdf.py` | PDF를 Paddle에 직접 전달 → Markdown 저장 |
| 문서 준비 2 | `build_corpus.py` | Markdown을 페이지당 1개 청크로 저장 |
| 문서 준비 3 | `rag.py --build` | BGE-M3로 임베딩하고 인덱스 저장 |
| 평소 질문 | `rag.py` → `llm_client.py` | 질문 검색 → Qwen 답변 |

아래 데이터 경로는 RAG 프로젝트 루트를 기준으로 합니다.

## 1. parse_pdf.py — PDF 직접 분석

**입력:** `--pdf`로 지정한 PDF, `--output` 저장 경로. 생략하면 현재 매뉴얼과 기본 파싱 폴더를 사용합니다.
**출력:** `outputs/parsing/pp_structurev3/`의 페이지별 Markdown과 `run_summary.json`.

`main()`은 다음 순서로 처리합니다.

1. 원본 PDF 경로와 해시를 확인하고 실행 상태를 기록합니다.
2. `PPStructureV3(**settings)`로 분석 파이프라인을 생성합니다.
3. **`pipeline.predict(input=str(pdf_path))`에 PDF 경로를 직접 전달**합니다.
4. 반환된 `page_index`(0부터 시작)와 `page_count`로 페이지 번호·전체 페이지 수를 확인합니다.
5. `page_01/page_01.md`처럼 명시적인 파일명으로 페이지별 Markdown을 저장합니다. 상세 분석 JSON은 저장하지 않습니다.
6. 모든 페이지의 결과가 순서대로 반환되었는지 확인한 뒤 완료 상태를 기록합니다. 실패한 실행은 코퍼스 갱신에 사용할 수 없습니다.

PDF 렌더링은 Paddle 내부에 맡깁니다. 우리 코드가 페이지 PNG를 만들거나 이미지 해상도를 조절하지 않습니다. 단, Paddle의 Markdown 저장 함수가 그림 영역 이미지를 함께 저장할 수 있습니다. 실행 기록은 누락·다른 원본 사용 방지를 위한 것으로, 내용의 정확도를 검수하는 기능은 아닙니다.

| 주요 모델 | 역할 |
|---|---|
| `PP-DocLayout-S` | 본문·제목·표·그림 영역 검출 |
| `PP-OCRv5_mobile_det` | 글자 영역 검출 |
| `en_PP-OCRv5_mobile_rec` | 영어 글자 인식 |
| `SLANet_plus` | 표 구조 인식 설정 |

표 분류·셀 검출 등의 보조 모델도 내부에서 사용합니다. CPU 4스레드, MKL-DNN 비활성화, OCR 검출 입력 최대 1,280픽셀이며 수식·차트 수치 변환·도장 인식은 비활성화합니다.

**매번 PDF 전체를 분석합니다.** `--pages`, `--width`, `--force`, `--prepare-only`와 페이지별 캐시 재사용은 제거했습니다. `--pdf`와 `--output`만 사용합니다. 기존 파싱 결과는 자동으로 다시 만들지 않으므로, 코드 변경 후에도 저장된 Markdown과 검색 인덱스는 그대로 사용할 수 있습니다.

## 2. build_corpus.py — Markdown을 청크로 저장

**입력:** `run_summary.json`, 해당 원본 PDF, `page_XX/page_XX.md`.
**출력:** `data/rag/corpus.jsonl`.

주요 함수는 `build()` 하나입니다.

1. PDF 해시가 파싱 당시 원본과 같은지 확인합니다.
2. 전체 페이지 처리가 완료되었는지 확인합니다.
3. 각 페이지의 Markdown을 읽고 앞뒤 공백만 제거합니다.
4. **페이지당 하나의 청크**를 만들고 페이지·출처를 붙입니다.
5. 모든 페이지를 읽은 뒤 코퍼스를 저장합니다.

| 청크 필드 | 뜻 |
|---|---|
| `id` | 예: `p11-markdown` |
| `page` | 원본 PDF 페이지 |
| `content` | Paddle Markdown 본문. HTML 표가 포함될 수 있음 |
| `origin` | `PP-StructureV3 Markdown` |
| `source_markdown` | 내용을 읽은 Markdown 파일 |
| `source_pdf`, `source_sha256` | 원본 PDF와 식별 해시 |

**표를 셀로 풀거나 원본 표와 비교하지 않습니다.** BeautifulSoup·pdfplumber·수동 검수 설명·별도 OCR 대체·메뉴 연결 정보가 없습니다. 포함 페이지와 청크 수를 별도 JSON으로 기록하지 않습니다.

현재 42페이지에서 42개 청크를 만듭니다. Markdown 수정 후에는 이 파일을 다시 실행하고 인덱스도 갱신해야 합니다. 내용이 비었거나 파일이 누락되면 코퍼스를 교체하기 전에 오류를 냅니다.

## 3. rag.py — 기본 벡터 검색과 답변 연결

**입력:** `corpus.jsonl`, `index.json`, 질문.
**출력:** 인덱스 파일 또는 화면의 검색 근거·답변.

| 함수 | 역할 |
|---|---|
| `api()` | 로컬 Ollama API 요청 |
| `embed()` | BGE-M3로 텍스트를 벡터로 변환 |
| `build_index()` | 청크·벡터·식별 정보를 `index.json`에 저장 |
| `cosine()` | 질문과 문서 벡터의 코사인 유사도 계산 |
| `select_hits()` | 유사도 내림차순으로 상위 3개 청크 선택 |
| `retrieve()` | 원본·코퍼스·모델 변경 확인, 질문 임베딩, 검색 |
| `build_context()` | 검색된 Markdown에 페이지 출처를 붙여 연결 |
| `ask()` | 검색 → 근거 구성 → Qwen 요청 및 화면 출력 |
| `main()` | `--build`, 단일 질문, 대화형 실행 선택 |

`build_index()`는 같은 모델로 임베딩한 동일 본문을 재사용합니다. 코퍼스·PDF·임베딩 모델이 바뀌면 `retrieve()`가 오래된 인덱스 사용을 차단합니다.

`build_context()`는 검색한 Markdown에 페이지 출처를 붙여 Qwen에 전달할 근거를 구성합니다.

이전 MMR, 메뉴 연결 확장, 표의 부모 문맥, 이동 경로 판별·표 필터는 제거했습니다. 유사도는 정답 확률이 아닙니다.

`--search-only`는 Qwen을 호출하지 않고 검색 근거만 표시합니다. 답변 저장용 `--output` 옵션과 결과 JSON 저장, 처리 시간 측정은 없습니다. 질문마다 독립적으로 처리합니다. `ask()`는 일반 질문에서 답변 문자열을 반환하고 검색만 실행하면 반환값이 없습니다.

## 4. llm_client.py — Qwen 호출

`chat(question, context, system_extra="")`가 질문과 근거를 `http://127.0.0.1:11435/api/chat`으로 보냅니다. URL은 같은 PC의 Ollama 서버입니다.

- 모델: `qwen3:4b-instruct`
- 설정: 문맥 4,096토큰, 최대 출력 256토큰, CPU 4스레드, temperature 0.2
- 출력: 생성되는 답변을 바로 보여주는 스트리밍
- 반환: 답변 문자열. 시간·토큰 속도 측정이나 파일 저장은 하지 않음

한국어·문서 근거·페이지 표시를 지시합니다. `rag.py`는 매뉴얼과 실시간 데이터의 차이, 이미지 링크의 한계 등을 추가로 안내합니다. 이 파일은 PDF를 읽거나 검색하지 않으며 단독 질문 입력 루프도 없습니다.

모델 최초 다운로드·재설치는 별도 Python 파일 없이 Ollama의 `pull` 명령으로 수행합니다. 서버 주소와 실행 파일 경로를 포함한 명령은 [프로젝트 README](../README.md)의 환경과 관리 명령을 참고하세요. Paddle 분석 모델은 PP-StructureV3가 필요할 때 준비합니다.

## 의존성과 기존 자료

작성한 Python 코드에서는 pdfplumber·BeautifulSoup·pypdfium2를 직접 사용하지 않습니다. 단, BeautifulSoup과 pypdfium2는 PaddleX 내부 의존성이므로 `requirements-parser-lock.txt`와 설치 환경에 남겨 둡니다.

이전 수동 검수 설명은 `outputs/legacy/page_annotations.json`에 보관하며 현재 코드는 읽지 않습니다. 기존 분석·검수·테스트 파일이 남아 있어도 현재 검색 원문은 페이지별 Paddle Markdown뿐입니다.

단순화는 정확도 향상을 보장하지 않습니다. 이미지 링크만 남은 내용, OCR 누락·오류, 여러 페이지를 연결해야 하는 질문에서 답변이 부족할 수 있습니다.
