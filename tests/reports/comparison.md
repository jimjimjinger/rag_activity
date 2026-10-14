# 문서 분석 결과 비교

세 구성의 전체 분석 완료.

원본 대조 해석은 [분석 보고서](analysis_review.md), 현재 실행 설정은 [실험 안내](../README.md)를 참고하세요.

같은 [원본 PDF](../../documents/B2B%20Dashboard%20Manual_v3.pdf)의 보정 전 결과입니다. 수동 보정은 적용하지 않았습니다.

| 기호 | 구성 | 장치 | 상태 | 완료 페이지 | 보정 전 청크 수 |
|---|---|---|---|---|---|
| L | Structure 경량: 기존 결과 | cpu | complete | 42/42 | 59 |
| S | Structure GPU 후보 | gpu:0 | complete | 42/42 | 44 |
| V | PaddleOCR-VL-1.6 | gpu:0 | complete | 42/42 | 55 |

L은 기존 CPU 경량 결과이며 S·V는 같은 GPU에서 완료한 8GB 프로필 결과입니다.
8GB 프로필은 페이지 폭 1,200픽셀, 배치 1로 실행합니다. 기존 결과와 입력 조건이 다르므로 구성 전체의 비교입니다.
현재 표시된 렌더링 폭 L / S / V: 2000 / 1200 / 1200 px.
각 실행의 입력·생성·메모리 조건은 결과 폴더의 run_summary.json에 기록합니다.
L의 과거 시간과 새 GPU 실행 시간을 직접적인 모델 속도 비교로 사용하지 않습니다.
S·V의 실행 시간도 모델·엔진·예열 조건과 함께 읽습니다. 초기 로드·다운로드 시간은 페이지별 시간과 별도입니다.

각 corpus.jsonl은 같은 split_page() 규칙을 적용했습니다. 제목 표기 형식도 청크 수에 영향을 줍니다.
글자 수·청크 수·처리 시간은 정확도 점수가 아닙니다. PDF와 대조한 누락·오인식·표 구조·읽기 순서를 기록하세요.

| 페이지 | L 경량 | S GPU 후보 | V VL | 글자 수 L / S / V | 청크 수 L / S / V | 페이지 시간(초) L / S / V |
|---|---|---|---|---|---|---|
| 1 | [보기](../results/structure_cpu/page_01/page_01.md) | [보기](../results/structure_gpu_8gb/page_01/page_01.md) | [보기](../results/vl_gpu_8gb/page_01/page_01.md) | 60 / 57 / 66 | 1 / 1 / 1 | 10.56 / 7.05 / 2.9 |
| 2 | [보기](../results/structure_cpu/page_02/page_02.md) | [보기](../results/structure_gpu_8gb/page_02/page_02.md) | [보기](../results/vl_gpu_8gb/page_02/page_02.md) | 211 / 326 / 269 | 1 / 1 / 1 | 12.67 / 6.98 / 3.3 |
| 3 | [보기](../results/structure_cpu/page_03/page_03.md) | [보기](../results/structure_gpu_8gb/page_03/page_03.md) | [보기](../results/vl_gpu_8gb/page_03/page_03.md) | 27 / 111 / 109 | 1 / 1 / 1 | 7.22 / 0.52 / 1.18 |
| 4 | [보기](../results/structure_cpu/page_04/page_04.md) | [보기](../results/structure_gpu_8gb/page_04/page_04.md) | [보기](../results/vl_gpu_8gb/page_04/page_04.md) | 914 / 906 / 905 | 4 / 1 / 4 | 25.52 / 0.79 / 6.14 |
| 5 | [보기](../results/structure_cpu/page_05/page_05.md) | [보기](../results/structure_gpu_8gb/page_05/page_05.md) | [보기](../results/vl_gpu_8gb/page_05/page_05.md) | 104 / 2380 / 8564 | 1 / 1 / 1 | 77.45 / 2.63 / 17.52 |
| 6 | [보기](../results/structure_cpu/page_06/page_06.md) | [보기](../results/structure_gpu_8gb/page_06/page_06.md) | [보기](../results/vl_gpu_8gb/page_06/page_06.md) | 267 / 863 / 908 | 1 / 1 / 2 | 39.3 / 0.98 / 6.05 |
| 7 | [보기](../results/structure_cpu/page_07/page_07.md) | [보기](../results/structure_gpu_8gb/page_07/page_07.md) | [보기](../results/vl_gpu_8gb/page_07/page_07.md) | 78 / 81 / 84 | 1 / 1 / 1 | 8.94 / 0.34 / 0.99 |
| 8 | [보기](../results/structure_cpu/page_08/page_08.md) | [보기](../results/structure_gpu_8gb/page_08/page_08.md) | [보기](../results/vl_gpu_8gb/page_08/page_08.md) | 114 / 175 / 642 | 1 / 1 / 2 | 41.67 / 1.07 / 4.12 |
| 9 | [보기](../results/structure_cpu/page_09/page_09.md) | [보기](../results/structure_gpu_8gb/page_09/page_09.md) | [보기](../results/vl_gpu_8gb/page_09/page_09.md) | 703 / 699 / 820 | 1 / 1 / 2 | 30.77 / 1.15 / 4.68 |
| 10 | [보기](../results/structure_cpu/page_10/page_10.md) | [보기](../results/structure_gpu_8gb/page_10/page_10.md) | [보기](../results/vl_gpu_8gb/page_10/page_10.md) | 716 / 2033 / 7107 | 3 / 1 / 1 | 80.44 / 2.45 / 14.48 |
| 11 | [보기](../results/structure_cpu/page_11/page_11.md) | [보기](../results/structure_gpu_8gb/page_11/page_11.md) | [보기](../results/vl_gpu_8gb/page_11/page_11.md) | 662 / 1484 / 5439 | 1 / 1 / 1 | 60.08 / 1.85 / 11.49 |
| 12 | [보기](../results/structure_cpu/page_12/page_12.md) | [보기](../results/structure_gpu_8gb/page_12/page_12.md) | [보기](../results/vl_gpu_8gb/page_12/page_12.md) | 369 / 515 / 663 | 1 / 1 / 1 | 42.53 / 0.63 / 3.49 |
| 13 | [보기](../results/structure_cpu/page_13/page_13.md) | [보기](../results/structure_gpu_8gb/page_13/page_13.md) | [보기](../results/vl_gpu_8gb/page_13/page_13.md) | 832 / 1105 / 11084 | 3 / 1 / 1 | 55.73 / 0.9 / 20.78 |
| 14 | [보기](../results/structure_cpu/page_14/page_14.md) | [보기](../results/structure_gpu_8gb/page_14/page_14.md) | [보기](../results/vl_gpu_8gb/page_14/page_14.md) | 561 / 747 / 1227 | 2 / 1 / 2 | 26.84 / 0.67 / 5.53 |
| 15 | [보기](../results/structure_cpu/page_15/page_15.md) | [보기](../results/structure_gpu_8gb/page_15/page_15.md) | [보기](../results/vl_gpu_8gb/page_15/page_15.md) | 341 / 580 / 1055 | 1 / 1 / 2 | 44.8 / 0.74 / 4.55 |
| 16 | [보기](../results/structure_cpu/page_16/page_16.md) | [보기](../results/structure_gpu_8gb/page_16/page_16.md) | [보기](../results/vl_gpu_8gb/page_16/page_16.md) | 244 / 368 / 1650 | 1 / 1 / 1 | 33.14 / 0.52 / 7.17 |
| 17 | [보기](../results/structure_cpu/page_17/page_17.md) | [보기](../results/structure_gpu_8gb/page_17/page_17.md) | [보기](../results/vl_gpu_8gb/page_17/page_17.md) | 495 / 4455 / 672 | 1 / 1 / 1 | 54.88 / 1.66 / 3.62 |
| 18 | [보기](../results/structure_cpu/page_18/page_18.md) | [보기](../results/structure_gpu_8gb/page_18/page_18.md) | [보기](../results/vl_gpu_8gb/page_18/page_18.md) | 384 / 523 / 2153 | 1 / 1 / 1 | 22.61 / 0.63 / 7.0 |
| 19 | [보기](../results/structure_cpu/page_19/page_19.md) | [보기](../results/structure_gpu_8gb/page_19/page_19.md) | [보기](../results/vl_gpu_8gb/page_19/page_19.md) | 476 / 3405 / 15813 | 1 / 1 / 1 | 62.94 / 1.77 / 20.96 |
| 20 | [보기](../results/structure_cpu/page_20/page_20.md) | [보기](../results/structure_gpu_8gb/page_20/page_20.md) | [보기](../results/vl_gpu_8gb/page_20/page_20.md) | 701 / 812 / 1206 | 3 / 1 / 3 | 45.97 / 0.93 / 5.49 |
| 21 | [보기](../results/structure_cpu/page_21/page_21.md) | [보기](../results/structure_gpu_8gb/page_21/page_21.md) | [보기](../results/vl_gpu_8gb/page_21/page_21.md) | 674 / 686 / 1036 | 3 / 1 / 1 | 64.83 / 1.44 / 5.31 |
| 22 | [보기](../results/structure_cpu/page_22/page_22.md) | [보기](../results/structure_gpu_8gb/page_22/page_22.md) | [보기](../results/vl_gpu_8gb/page_22/page_22.md) | 494 / 728 / 2650 | 1 / 1 / 1 | 54.52 / 0.92 / 9.58 |
| 23 | [보기](../results/structure_cpu/page_23/page_23.md) | [보기](../results/structure_gpu_8gb/page_23/page_23.md) | [보기](../results/vl_gpu_8gb/page_23/page_23.md) | 618 / 692 / 7632 | 1 / 1 / 1 | 122.2 / 1.7 / 16.83 |
| 24 | [보기](../results/structure_cpu/page_24/page_24.md) | [보기](../results/structure_gpu_8gb/page_24/page_24.md) | [보기](../results/vl_gpu_8gb/page_24/page_24.md) | 332 / 493 / 9469 | 1 / 1 / 1 | 71.8 / 1.61 / 17.24 |
| 25 | [보기](../results/structure_cpu/page_25/page_25.md) | [보기](../results/structure_gpu_8gb/page_25/page_25.md) | [보기](../results/vl_gpu_8gb/page_25/page_25.md) | 415 / 438 / 551 | 1 / 1 / 1 | 59.67 / 0.94 / 3.4 |
| 26 | [보기](../results/structure_cpu/page_26/page_26.md) | [보기](../results/structure_gpu_8gb/page_26/page_26.md) | [보기](../results/vl_gpu_8gb/page_26/page_26.md) | 338 / 424 / 550 | 1 / 1 / 1 | 95.12 / 1.04 / 3.4 |
| 27 | [보기](../results/structure_cpu/page_27/page_27.md) | [보기](../results/structure_gpu_8gb/page_27/page_27.md) | [보기](../results/vl_gpu_8gb/page_27/page_27.md) | 414 / 429 / 548 | 1 / 1 / 1 | 60.91 / 1.3 / 3.21 |
| 28 | [보기](../results/structure_cpu/page_28/page_28.md) | [보기](../results/structure_gpu_8gb/page_28/page_28.md) | [보기](../results/vl_gpu_8gb/page_28/page_28.md) | 117 / 615 / 1400 | 1 / 1 / 1 | 26.41 / 0.69 / 6.89 |
| 29 | [보기](../results/structure_cpu/page_29/page_29.md) | [보기](../results/structure_gpu_8gb/page_29/page_29.md) | [보기](../results/vl_gpu_8gb/page_29/page_29.md) | 464 / 602 / 598 | 2 / 1 / 2 | 31.83 / 0.72 / 4.3 |
| 30 | [보기](../results/structure_cpu/page_30/page_30.md) | [보기](../results/structure_gpu_8gb/page_30/page_30.md) | [보기](../results/vl_gpu_8gb/page_30/page_30.md) | 676 / 691 / 6725 | 3 / 1 / 1 | 57.25 / 1.07 / 13.48 |
| 31 | [보기](../results/structure_cpu/page_31/page_31.md) | [보기](../results/structure_gpu_8gb/page_31/page_31.md) | [보기](../results/vl_gpu_8gb/page_31/page_31.md) | 656 / 2500 / 11493 | 1 / 1 / 1 | 75.77 / 1.97 / 20.09 |
| 32 | [보기](../results/structure_cpu/page_32/page_32.md) | [보기](../results/structure_gpu_8gb/page_32/page_32.md) | [보기](../results/vl_gpu_8gb/page_32/page_32.md) | 338 / 502 / 9381 | 1 / 1 / 1 | 72.39 / 1.62 / 17.37 |
| 33 | [보기](../results/structure_cpu/page_33/page_33.md) | [보기](../results/structure_gpu_8gb/page_33/page_33.md) | [보기](../results/vl_gpu_8gb/page_33/page_33.md) | 237 / 374 / 530 | 1 / 1 / 1 | 26.89 / 0.84 / 3.26 |
| 34 | [보기](../results/structure_cpu/page_34/page_34.md) | [보기](../results/structure_gpu_8gb/page_34/page_34.md) | [보기](../results/vl_gpu_8gb/page_34/page_34.md) | 33 / 82 / 79 | 1 / 1 / 1 | 6.59 / 0.31 / 1.32 |
| 35 | [보기](../results/structure_cpu/page_35/page_35.md) | [보기](../results/structure_gpu_8gb/page_35/page_35.md) | [보기](../results/vl_gpu_8gb/page_35/page_35.md) | 790 / 780 / 775 | 3 / 3 / 3 | 22.06 / 0.59 / 5.51 |
| 36 | [보기](../results/structure_cpu/page_36/page_36.md) | [보기](../results/structure_gpu_8gb/page_36/page_36.md) | [보기](../results/vl_gpu_8gb/page_36/page_36.md) | 317 / 1311 / 2744 | 1 / 1 / 1 | 26.2 / 0.89 / 6.89 |
| 37 | [보기](../results/structure_cpu/page_37/page_37.md) | [보기](../results/structure_gpu_8gb/page_37/page_37.md) | [보기](../results/vl_gpu_8gb/page_37/page_37.md) | 1199 / 1391 / 3705 | 1 / 1 / 1 | 40.2 / 1.92 / 7.73 |
| 38 | [보기](../results/structure_cpu/page_38/page_38.md) | [보기](../results/structure_gpu_8gb/page_38/page_38.md) | [보기](../results/vl_gpu_8gb/page_38/page_38.md) | 1513 / 1623 / 4375 | 1 / 1 / 1 | 40.66 / 1.44 / 8.06 |
| 39 | [보기](../results/structure_cpu/page_39/page_39.md) | [보기](../results/structure_gpu_8gb/page_39/page_39.md) | [보기](../results/vl_gpu_8gb/page_39/page_39.md) | 1654 / 1705 / 4859 | 1 / 1 / 1 | 46.52 / 1.37 / 8.81 |
| 40 | [보기](../results/structure_cpu/page_40/page_40.md) | [보기](../results/structure_gpu_8gb/page_40/page_40.md) | [보기](../results/vl_gpu_8gb/page_40/page_40.md) | 179 / 196 / 282 | 1 / 1 / 1 | 31.31 / 1.01 / 2.84 |
| 41 | [보기](../results/structure_cpu/page_41/page_41.md) | [보기](../results/structure_gpu_8gb/page_41/page_41.md) | [보기](../results/vl_gpu_8gb/page_41/page_41.md) | 65 / 1028 / 2406 | 1 / 1 / 1 | 12.16 / 0.64 / 49.75 |
| 42 | [보기](../results/structure_cpu/page_42/page_42.md) | [보기](../results/structure_gpu_8gb/page_42/page_42.md) | [보기](../results/vl_gpu_8gb/page_42/page_42.md) | 75 / 70 / 71 | 1 / 1 / 1 | 6.19 / 0.29 / 1.02 |

## 우선 눈으로 비교할 페이지

- 1: 한글 메모와 날짜. 영어용·중영일용 인식기의 지원 언어 차이도 함께 기록
- 9: Suggested Questions 제목과 본문 경계
- 11·22: Page Type 설명 문장 누락 여부
- 15: Strategic Task와 Applied Area 항목의 분리
- 20: Page Details의 번호와 읽기 순서
- 23·31: Search·Download 제목과 설명
- 24·32: 5·6번 항목의 설명 누락 여부
- 37~39: 표의 헤더·행·열·값 연결

32페이지 Add item처럼 원본의 표현이 혼동을 주는 부분도 PDF와 대조해 모델 오류와 구분합니다.
모델 설정과 실행 순서는 [실험 안내](../README.md)를 참고하세요.
