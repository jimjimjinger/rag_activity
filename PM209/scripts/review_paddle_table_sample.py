"""Check sample preservation and compare HTML cells to image-transcribed references."""

import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "PM209/reports/paddle_table_sample_20261022"


class Cells(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.cell = None
        self.table_count = 0

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table_count += 1
        elif tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            attrs = dict(attrs)
            self.cell = {"text": "", "rowspan": int(attrs.get("rowspan", 1)),
                         "colspan": int(attrs.get("colspan", 1))}
            self.rows[-1].append(self.cell)
        elif tag == "br" and self.cell is not None:
            self.cell["text"] += " "

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell["text"] += data


def norm(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def main():
    references = json.loads((OUT / "reference.json").read_text(encoding="utf-8"))
    samples = json.loads((OUT / "samples.json").read_text(encoding="utf-8"))
    metrics = json.loads((OUT / "run_metrics.json").read_text(encoding="utf-8"))
    manual_path = OUT / "manual_review.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    checks = {}
    sections = []
    for sample in samples:
        kind = sample["kind"]
        html_path = OUT / f"{kind}.html"
        if not html_path.exists():
            checks[kind] = {"html_output_success": False}
            continue
        raw = html_path.read_text(encoding="utf-8")
        parser = Cells()
        parser.feed(raw)
        ref = references[kind]
        spans = [{"row": r, "cell": c, "rowspan": cell["rowspan"], "colspan": cell["colspan"]}
                 for r, row in enumerate(parser.rows) for c, cell in enumerate(row)
                 if cell["rowspan"] > 1 or cell["colspan"] > 1]
        grid = {}
        collisions = []
        for r, row in enumerate(parser.rows):
            c = 0
            for cell in row:
                while (r, c) in grid:
                    c += 1
                for rr in range(r, r + cell["rowspan"]):
                    for cc in range(c, c + cell["colspan"]):
                        if (rr, cc) in grid:
                            collisions.append([rr, cc])
                        grid[rr, cc] = cell["text"]
                c += cell["colspan"]
        width = max((c for r, c in grid), default=-1) + 1
        check = dict(html_output_success=True, rows=parser.rows,
                     actual_row_count=len(parser.rows), actual_column_count=width,
                     expected_row_count=ref["expected_rows"], expected_column_count=ref["expected_columns"],
                     actual_spans=spans, expected_spans=ref["expected_spans"],
                     spans_match=spans == ref["expected_spans"], grid_collisions=collisions,
                     row_count_matches=len(parser.rows) == ref["expected_rows"],
                     column_count_matches=width == ref["expected_columns"],
                     html_closed="</table>" in raw.lower(), table_count=parser.table_count)
        if "rows" in ref:
            mismatches = []
            matched = 0
            for r, row in enumerate(ref["rows"]):
                for c, value in enumerate(row):
                    actual = parser.rows[r][c]["text"] if r < len(parser.rows) and c < len(parser.rows[r]) else None
                    if actual is not None and norm(value) == norm(actual):
                        matched += 1
                    else:
                        mismatches.append({"row": r, "cell": c, "expected": value, "actual": actual})
            check.update(reference_cells=sum(map(len, ref["rows"])), exact_cells=matched,
                         cell_mismatches=mismatches)
        else:
            anchors = []
            for anchor in ref["numeric_and_unit_anchors"]:
                row = parser.rows[anchor["row"]] if anchor["row"] < len(parser.rows) else []
                actual = row[1]["text"] if len(row) > 1 else ""
                anchors.append({**anchor, "same_row_found": norm(anchor["text"]) in norm(actual)})
            check["numeric_unit_anchors"] = anchors
            check["last_row_solution_present"] = norm(ref["last_row_solution"]) in norm(parser.rows[-1][-1]["text"])
            check["manual_full_text_and_icons_review_required"] = True
        checks[kind] = check
        manual_note = html.escape(manual.get(kind, {}).get("conclusion_ko", "수동 검수 전"))
        timing = next(t["inference_seconds"] for t in metrics["tables"] if t["kind"] == kind)
        sections.append(f'<section id="{kind}"><h2>{kind} · {timing:.2f}초</h2><p>{manual_note}</p><p>{html.escape(sample["region"]["id"])}</p>'
                        f'<div class="pair"><div><h3>원본 표 영역</h3><img src="{kind}.png"></div>'
                        f'<div><h3>Paddle 반환 HTML</h3>{re.sub(r"</?(?:html|body)>", "", raw)}</div></div>'
                        f'<p><a href="{kind}.html">원본 HTML 파일</a> · <a href="{kind}.raw.json">파이프라인 원시 결과</a></p></section>')

    # Compare complete saved sample pages to source pages after removing only our added field.
    wanted = {s["page_id"] for s in samples}
    source_pages = {}
    for split in {s["split"] for s in samples}:
        with (ROOT / "PM209/datasets/PM209/data/vlm" / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                page = json.loads(line)
                if page["id"] in wanted:
                    source_pages[page["id"]] = page
    additions = []
    with (OUT / "pages.sample.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            for region in page["bounding_boxes"]:
                if "table_content" in region:
                    assert region["structure"] == "Table"
                    additions.append(region["id"])
                    del region["table_content"]
            assert page == source_pages[page["id"]], "Source page fields changed"
    assert set(additions) == {s["region"]["id"] for s in samples}
    checks["preservation"] = {"full_sample_pages_match_except_table_content": True,
                              "added_table_ids": additions,
                              "original_and_vlm_hashes": json.loads((OUT / "preservation.json").read_text(encoding="utf-8"))}
    (OUT / "validation.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    doc = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PM209 Paddle 표 3개 검수</title>
<style>body{font:16px/1.5 system-ui;margin:32px;color:#18212b;background:#f6f7f9}section{background:white;padding:24px;margin:24px 0;border-radius:10px}section p{overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}.pair>div{min-width:0;overflow:auto}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font:14px/1.5 Arial}td,th{border:1px solid #777;padding:7px;vertical-align:top}h1{margin-bottom:8px}a{color:#075dc3}@media(max-width:1000px){.pair{grid-template-columns:1fr}}</style>
<h1>PM209 · PaddleOCR TableRecognitionPipelineV2</h1><p>표 3개 샘플. 왼쪽은 기존 shape로 자른 원본, 오른쪽은 파이프라인이 반환한 HTML입니다. HTML 저장 성공은 인식 정확도를 뜻하지 않습니다. 표시용 테두리만 추가했으며 HTML 파일과 table_content는 반환 문자열 그대로입니다.</p>
<p><a href="validation.json">셀·병합·보존 검증</a> · <a href="run_metrics.json">처리 시간·GPU 기록</a> · <a href="tables.sample.jsonl">Table 샘플</a> · <a href="pages.sample.jsonl">페이지 보존 샘플</a></p>'''
    details = '''<section><h2>모델·설정·보존</h2><p>PaddleOCR 3.3.2 / PaddleX 3.3.11 / 공식 Windows RTX 50용 PaddlePaddle GPU 3.0.0.dev20250717, CUDA 12.9, cuDNN 9.9. OCR: PP-OCRv5_server_det + PP-OCRv5_server_rec. 표 분류: PP-LCNet_x1_0_table_cls. 구조: SLANeXt_wired / SLANeXt_wireless. 셀 검출: RT-DETR-L_wired_table_cell_det / RT-DETR-L_wireless_table_cell_det.</p><p>gpu:0, 표 한 개씩·OCR 배치 1, 검출 긴 변 최대 1280, 레이아웃 검출·문서/표/텍스트 방향 분류·왜곡 보정 끄기, TensorRT/HPI 끄기. Paddle auto_growth 할당 한도 6144MiB. 관측 장치 전체 VRAM 최대 2810MiB, 온도 60°C. 종료 후 VRAM 0MiB. 모델 준비 약 200초는 표별 처리 시간에서 제외했습니다.</p><p>원본과 그림 보강본 train/val/test 6개 파일의 SHA-256이 전후 일치합니다. 그림 description 23,807개와 기존 ID·좌표·본문·QA를 유지했습니다. 별도 페이지 샘플에서는 세 Table에 table_content만 추가했습니다. 모델에는 이미지 배열만 전달했습니다.</p><p>초기 한글 모델 경로 로딩 오류는 %LOCALAPPDATA%/PM209-PaddleTables/paddlex 영문 캐시로 해결했습니다. 표 추론 오류는 0건입니다. 세 표 모두 마지막 행까지 확인했고 끝부분 잘림은 관찰되지 않았습니다. HTML은 반환 문자열 그대로이며 오류를 수정하거나 Qwen으로 다시 작성하지 않았습니다.</p><p><a href="manual_review.json">행별 수동 검수와 실행 명령</a> · <a href="settings.json">실행 설정</a> · <a href="pipeline.yaml">전체 파이프라인 설정</a> · <a href="environment.json">GPU 환경</a> · <a href="requirements.lock.txt">설치 패키지 잠금 목록</a></p><h3>실행 명령 (저장소 루트)</h3><pre>.runtime\\venv-paddle-tables\\Scripts\\python.exe -X utf8 -u -B PM209/scripts/run_paddle_table_sample.py\n.runtime\\venv-paddle-tables\\Scripts\\python.exe -X utf8 -B PM209/scripts/review_paddle_table_sample.py</pre><p>기존 HTML이 있으면 추출 실행기는 덮어쓰기를 거부합니다. 전체 표 처리는 실행하지 않았습니다.</p></section>'''
    (OUT / "review.html").write_text(doc + "\n".join(sections) + details + "</html>", encoding="utf-8")
    print(json.dumps({k: {a: b for a, b in v.items() if a not in ("rows", "original_and_vlm_hashes")} for k, v in checks.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
