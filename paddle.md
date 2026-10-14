# Paddle 계열의 역할과 발전 과정

문서 분석과 RAG에 필요한 핵심만 정리했습니다. 공식 자료 확인일: 2026-10-10.

## 1. 이름별 역할

| 이름 | 무엇인가? | 쉬운 설명 |
|---|---|---|
| **PaddlePaddle** | 딥러닝 프레임워크 | 모델을 학습하고 실행하는 기반. PyTorch와 같은 범주 |
| **PaddleOCR** | OCR·문서 분석 도구 모음 | PP-OCR, PP-Structure, PaddleOCR-VL 등을 제공하는 프로젝트 |
| **PP-OCR** | 글자 검출·인식 모델 계열 | 글자가 어디 있는지 찾고, 이미지의 글자를 텍스트로 변환 |
| **PP-Structure** | 문서 구조 분석 파이프라인 계열 | OCR에 배치·표 분석 등을 결합해 문서를 구조화 |
| **PaddleOCR-VL** | 시각·언어 모델 기반 문서 분석 계열 | 이미지 특징과 언어 처리를 결합해 글자·표·수식 등 문서 요소를 인식 |

PaddlePaddle은 기반 프레임워크이고, 나머지는 문서 분석 도구·모델 계열입니다. 이름 전체가 하나의 모델의 버전 순서를 뜻하지는 않습니다. [PaddlePaddle 공식 저장소](https://github.com/PaddlePaddle/Paddle), [PaddleOCR 공식 저장소](https://github.com/PaddlePaddle/PaddleOCR)

## 2. 어떻게 발전했나?

핵심 방향은 **글자를 읽는 OCR에서 표·배치까지 다루는 문서 분석으로 기능이 확장된 것**입니다. 기존 OCR 계열도 함께 개선되고 있습니다.

| 시기·단계 | 주요 변화 |
|---|---|
| 초기 PP-OCR → PP-OCRv2 | 가벼운 글자 검출·인식 시스템의 정확도와 효율 개선 |
| 2022년 PP-OCRv3 / 2023년 PP-OCRv4 | 다국어 인식 개선, 모바일·서버 환경에 맞춘 OCR 발전 |
| PP-Structure → 2022년 PP-StructureV2 | 글자뿐 아니라 문서 배치·표 구조를 분석하고 문서 레이아웃 복원 기능 확장 |
| 2025년 PP-OCRv5 / PP-StructureV3 | 다국어 OCR 개선과 문서 계층·읽기 순서 등을 다루는 구조 분석 강화 |
| 2025년 PaddleOCR-VL | 시각·언어 모델을 활용하는 문서 분석 방식 추가 |
| 2026년 VL-1.5 → VL-1.6 | 촬영·왜곡 등 복잡한 문서 환경과 표 등 요소 인식 개선 |
| 2026년 PP-OCRv6 | OCR 계열도 계속 발전. 다국어 인식 개선 및 tiny·small·medium 구성 제공 |

초기 OCR 개선은 [PP-OCRv2 기술 보고서](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/PP-OCRv2.pdf), 2022~2023년 이력은 [공식 2.x 문서](https://www.paddleocr.ai/v2.10.0/en/index.html), 2025년 구성은 [PaddleOCR 3.0 기술 보고서](https://arxiv.org/abs/2507.05595)와 [PaddleOCR-VL 논문](https://arxiv.org/abs/2510.14528), 2026년 변경은 [공식 릴리스 기록](https://github.com/PaddlePaddle/PaddleOCR/releases)을 기준으로 요약했습니다.

**PP-StructureV3는 여러 전문 모델을 연결하는 방식이고, PaddleOCR-VL은 시각·언어 모델을 활용하는 별도 계열입니다.** VL이 나왔다고 PP-Structure가 없어지거나, 모든 프로젝트에서 교체해야 하는 것은 아닙니다. PaddleOCR 패키지 버전과 PP-OCR 모델 버전도 서로 다릅니다.

## 3. 현재 프로젝트에서 사용하는 구성

현재 [src/parse_pdf.py](src/parse_pdf.py)는 **PP-StructureV3**를 사용합니다. 하나의 모델이 모든 작업을 처리하는 것이 아니라, 아래 모델과 보조 모델을 연결합니다.

| 코드에서 지정한 모델 | 역할 |
|---|---|
| `PP-DocLayout-S` | 제목·본문·표·그림 등의 영역과 배치 검출 |
| `PP-OCRv5_mobile_det` | 글자가 있는 영역 검출 |
| `en_PP-OCRv5_mobile_rec` | 검출된 영역의 영어 글자 인식 |
| `SLANet_plus` | 표 구조 인식. 코드에서는 유선표 구조 인식 모델로 지정 |

CPU에서 실행하도록 경량 모델을 지정했고, 표 인식은 사용합니다. 수식·차트 수치·도장 인식과 문서 방향·왜곡 보정은 현재 설정에서 꺼 두었습니다. 이 목록은 **우리 코드의 설정**이며 PP-StructureV3 전체 기능 목록은 아닙니다.

처리는 대략 **문서 영역 구분 → 글자·표 분석 → 읽기 순서에 맞춰 결과 구성**으로 이해하면 됩니다. PDF 페이지의 이미지 처리는 Paddle 내부에 맡기고, 우리 코드는 결과를 페이지별 Markdown으로 저장합니다.

## 4. RAG에서 맡는 구간

```text
PDF → PP-StructureV3 → 페이지별 Markdown → 청크 생성 → BGE-M3 임베딩·저장

질문 → BGE-M3 질문 임베딩 → 관련 청크 검색 → Qwen 답변
```

**Paddle은 검색할 문서 내용을 추출하는 역할**, **BGE-M3는 검색용 벡터를 만드는 역할**, **Qwen은 검색 근거로 답변하는 역할**입니다. 질문할 때 Paddle을 다시 실행하지 않습니다.

그림 속 글자는 OCR로 추출될 수 있지만, 그림의 의미 전체가 설명문으로 바뀌는 것은 아닙니다. 현재 시스템에서 Markdown의 이미지 링크만 남은 부분은 Qwen이 이미지를 직접 보는 방식으로 처리하지 않습니다.

## 5. PyMuPDF와는 무엇이 다른가?

**PyMuPDF의 기본 텍스트 추출은 PDF에 저장된 글자 정보를 꺼내는 방식이고, PaddleOCR은 이미지에 보이는 글자를 모델로 인식하는 방식입니다.** 비교 대상은 기반 프레임워크인 PaddlePaddle보다 문서 분석 도구인 PaddleOCR입니다.

| 구분 | PyMuPDF | PaddleOCR / PP-StructureV3 |
|---|---|---|
| 주된 역할 | PDF 읽기·편집·렌더링·내용 추출 | AI 모델로 글자·문서 구조 분석 |
| 일반 텍스트 PDF | 저장된 글자와 좌표를 직접 추출 | 페이지 이미지를 분석해 글자를 인식 |
| 스캔·스크린샷 속 글자 | 기본 텍스트 추출로는 읽지 못하며 OCR 처리가 필요 | OCR 모델로 인식 가능 |
| 표·배치 | 좌표·선 등을 활용한 표 추출 기능 제공 | 배치·표 인식 모델로 문서 구조 분석 |
| 처리 부담 | 기본 텍스트 추출은 빠르고 가벼움 | 모델을 불러오고 추론하는 시간이 필요 |

PyMuPDF에도 **Tesseract 기반 OCR 기능과 표 추출 기능이 있습니다.** 위 비교에서 기본 텍스트 추출과 OCR 사용을 구분해야 합니다. [PyMuPDF 텍스트·표 추출 안내](https://pymupdf.readthedocs.io/en/latest/recipes-text.html), [OCR 안내](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html). PP-StructureV3는 배치·표·읽기 순서 등을 분석하는 파이프라인입니다. [공식 설명](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html)

### 대시보드 매뉴얼을 예로 들면

- **PPT에서 직접 입력한 `India`를 PDF로 저장:** PDF에 글자 정보가 남아 있다면 PyMuPDF로 바로 추출할 수 있습니다.
- **`India`가 보이는 대시보드 화면을 캡처해 PPT에 붙인 뒤 PDF로 저장:** 별도 텍스트층이 없다면 사진 속 글자이므로 OCR로 인식해야 합니다. PaddleOCR은 이런 영역의 글자를 읽는 데 사용할 수 있습니다.

텍스트가 잘 저장된 PDF로 RAG를 만들 때는 PyMuPDF의 직접 추출이 더 간단할 수 있습니다. **우리 프로젝트는 스크린샷·표·배치를 모델로 분석하는 과정을 공부하기 위해 PaddleOCR을 사용합니다.** PaddleOCR이 모든 PDF에서 더 정확하다는 뜻은 아니며, OCR 과정에서 글자를 잘못 읽을 수도 있습니다.

## 6. 한 문장으로 설명하기

> Paddle 계열은 글자 인식 중심의 PP-OCR에서 표·배치까지 분석하는 PP-Structure와 시각·언어 모델 기반 PaddleOCR-VL로 기능을 확장해 왔으며, 우리 시스템은 PP-StructureV3의 문서 분석 결과를 RAG의 검색 자료로 사용합니다.

실습에서는 각 모델의 역할을 설명하고, 같은 PDF의 보정 전 Structure·VL 결과를 대조해 누락·읽기 순서·표 구조의 차이를 확인합니다.
기존 CPU 경량 Structure, GPU용 Structure 고정밀 후보, GPU용 VL을 별도 저장해 비교합니다. GPU 후보는 큰 배치 모델·서버 OCR·SLANeXt 표 모델을 사용하며 최적 구성으로 검증된 상태는 아닙니다.
비교 절차와 아직 검증하지 않은 부분은 [실험 안내](tests/README.md)에 정리했습니다. 문서 분석 결과와 검색용 청킹 결과를 구분해서 평가합니다.
