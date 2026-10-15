"""저장된 Structure·VL 화면 캡처 결과를 평가하며 OCR이나 LLM은 실행하지 않는다."""
import hashlib
from html import unescape
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/image_checks.json"
OUTPUT = ROOT / "tests/reports"
MODEL_NAMES = {"structure_gpu_8gb": "Structure", "vl_gpu_8gb": "VL"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    text = re.sub(r"<[^>]*>", " ", text)
    text = unicodedata.normalize("NFKC", unescape(text))
    text = text.replace("**", "")
    for dash in ("–", "—", "−", "‐", "‑"):
        text = text.replace(dash, "-")
    return re.sub(r"\s+", " ", text).strip().casefold()


def contains(text, check):
    expected = normalize(check["expected"])
    # 숫자 경계를 검사해 585263114033처럼 붙은 숫자열 속의 2631은 일치로 인정하지 않는다.
    boundary = r"\d" if check["kind"] in ("date", "number") else r"\w"
    return bool(re.search(rf"(?<!{boundary}){re.escape(expected)}(?!{boundary})", text))


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    box = [max(a[0], b[0]), max(a[1], b[1]),
           min(a[2], b[2]), min(a[3], b[3])]
    return box if area(box) else None


def union_coverage(target, boxes):
    clipped = [part for box in boxes if (part := intersection(target, box))]
    xs = sorted({x for box in clipped for x in (box[0], box[2])})
    total = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in clipped
                           if b[0] < right and b[2] > left)
        end = None
        height = 0
        for low, high in intervals:
            if end is None or low > end:
                height += high - low
                end = high
            elif high > end:
                height += high - end
                end = high
        total += (right - left) * height
    return total / area(target)


def capture_blocks(data, target, markdown):
    blocks = []
    for block in data["parsing_res_list"]:
        overlap = intersection(target, block["block_bbox"])
        if overlap and (area(overlap) / area(target) >= 0.5
                        or area(overlap) / area(block["block_bbox"]) >= 0.5):
            content = normalize(block["block_content"])
            blocks.append({
                "block_id": block["block_id"], "label": block["block_label"],
                "bbox": block["block_bbox"], "content": block["block_content"],
                "markdown_exposed": bool(content and content in markdown),
            })
    return blocks


def image_files(folder, target, markdown):
    images = []
    references = set(re.findall(r'src=["\']([^"\']+)', markdown))
    references.update(re.findall(r"!\[[^\]]*\]\(([^\s)]+)\)", markdown))
    for path in sorted((folder / "imgs").glob("*")):
        match = re.search(r"_box_(\d+)_(\d+)_(\d+)_(\d+)\.", path.name)
        if not match:
            continue
        box = list(map(int, match.groups()))
        if intersection(target, box):
            from PIL import Image
            with Image.open(path) as img:
                dimensions = list(img.size)
            if dimensions != [box[2] - box[0], box[3] - box[1]]:
                raise ValueError(f"Image dimensions differ from filename box: {path}")
            images.append({
                "path": path.relative_to(ROOT).as_posix(), "bbox": box,
                "sha256": sha256(path), "dimensions": dimensions,
                "markdown_linked": path.relative_to(folder).as_posix() in references,
            })
    return images


def evaluate():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pdf = ROOT / fixture["source_pdf"]
    if sha256(pdf) != fixture["source_sha256"]:
        raise ValueError("Source PDF differs from the reviewed PDF")
    report = {
        "evaluation_date": fixture["evaluation_date"],
        "source_pdf": fixture["source_pdf"], "source_sha256": fixture["source_sha256"],
        "fixture_sha256": sha256(FIXTURE), "script_sha256": sha256(Path(__file__)),
        "review_method": fixture["review_method"],
        "metric_definition": fixture["metric_definition"],
        "preservation_threshold": fixture["preservation_threshold"],
        "source_preview_hashes": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted((OUTPUT / "image_evaluation/source_previews").glob("*.png"))
        },
        "models": {},
    }
    for model in fixture["models"]:
        base = ROOT / "tests/results" / model
        manifest = json.loads((base / "run_summary.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != fixture["source_sha256"]:
            raise ValueError(f"{model}: source PDF mismatch")
        records = {item["page"]: item for item in manifest["results"]}
        result = {"name": MODEL_NAMES[model],
                  "manifest_sha256": sha256(base / "run_summary.json"),
                  "captures": []}
        for capture in fixture["captures"]:
            page = capture["page"]
            folder = base / f"page_{page:02d}"
            json_path = folder / f"page_{page:02d}.json"
            md_path = folder / f"page_{page:02d}.md"
            if sha256(md_path) != records[page]["markdown_sha256"]:
                raise ValueError(f"{model}: Markdown changed on p{page}")
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if [data["width"], data["height"]] != [1200, 675]:
                raise ValueError("Unexpected coordinate dimensions")
            raw_md = md_path.read_text(encoding="utf-8")
            markdown = normalize(raw_md)
            blocks = capture_blocks(data, capture["bbox"], markdown)
            images = image_files(folder, capture["bbox"], raw_md)
            coverage = union_coverage(capture["bbox"], [i["bbox"] for i in images])
            linked_coverage = union_coverage(
                capture["bbox"], [i["bbox"] for i in images if i["markdown_linked"]])
            checks = []
            for check in capture["checks"]:
                raw_hits = [b["block_id"] for b in blocks
                            if contains(normalize(b["content"]), check)]
                md_hits = [b["block_id"] for b in blocks
                           if b["markdown_exposed"] and contains(normalize(b["content"]), check)]
                checks.append({**check, "json_present": bool(raw_hits),
                               "markdown_present": bool(md_hits),
                               "json_block_ids": raw_hits, "markdown_block_ids": md_hits})
            result["captures"].append({
                "id": capture["id"], "page": page, "name": capture["name"],
                "bbox": capture["bbox"], "preview": capture["preview"],
                "json_path": json_path.relative_to(ROOT).as_posix(),
                "json_sha256": sha256(json_path),
                "markdown_path": md_path.relative_to(ROOT).as_posix(),
                "markdown_sha256": sha256(md_path),
                "model_settings": data["model_settings"],
                "image_coverage": round(coverage, 6),
                "linked_image_coverage": round(linked_coverage, 6),
                "image_preserved": coverage >= fixture["preservation_threshold"],
                "image_linked": linked_coverage >= fixture["preservation_threshold"],
                "images": images, "blocks": blocks, "checks": checks,
            })
        captures = result["captures"]
        all_checks = [c for capture in captures for c in capture["checks"]]
        result["totals"] = {
            "captures": len(captures), "items": len(all_checks),
            "image_preserved": sum(c["image_preserved"] for c in captures),
            "image_linked": sum(c["image_linked"] for c in captures),
            "json_present": sum(c["json_present"] for c in all_checks),
            "markdown_present": sum(c["markdown_present"] for c in all_checks),
        }
        report["models"][model] = result
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "image_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for result in report["models"].values():
        print(result["name"], result["totals"])
        for c in result["captures"]:
            print(c["id"], "coverage", c["image_coverage"], c["linked_image_coverage"],
                  "JSON", sum(x["json_present"] for x in c["checks"]),
                  "MD", sum(x["markdown_present"] for x in c["checks"]))
    return report


def write_report(report):
    models = list(report["models"].values())
    structure, vl = models
    s, v = structure["totals"], vl["totals"]
    lines = [
        "# Structure와 VL 화면 캡처 정량 평가",
        "",
        "평가일: 2026-10-15. 경량 Structure는 지민의 요청에 따라 제외했다. "
        "저장된 8GB GPU 결과만 읽었으며 모델 재실행·추가 OCR·LLM 평가·RAG 재빌드는 하지 않았다.",
        "",
        "## 결론",
        "",
        f"선택한 7개 캡처에서 추출 이미지 파일로 보존한 수는 Structure {s['image_preserved']}/7, "
        f"VL {v['image_preserved']}/7이다. 캡처 안의 지정 문자열 35개 중 JSON 보존은 "
        f"Structure {s['json_present']}/35, VL {v['json_present']}/35이며, "
        f"최종 Markdown 텍스트로 전달된 수는 Structure {s['markdown_present']}/35, "
        f"VL {v['markdown_present']}/35이다.",
        "",
        "**캡처를 이미지로 보존하는 성과와 내부 정보를 텍스트로 전달하는 성과가 달랐다.** "
        "현재 VL은 image 영역의 OCR을 비활성화했으므로 이 수치를 VL 모델 자체의 이미지 이해 정확도로 해석하지 않는다. "
        "본문·정형 표의 결과와 합쳐 단일 정확도로 평균 내지 않는다.",
        "",
        "## 평가 대상과 정답 작성",
        "",
        "원본 PDF를 CPU에서 페이지 폭 1,600px로 렌더링하고, 선택한 캡처를 3,600px 페이지 렌더에서 확대해 직접 확인했다. "
        "6페이지의 캡처 7개에 대해 판독 가능한 문자열을 5개씩 전사했다. "
        "20페이지 노란 메모에 가려진 항목, 로고·번호 원형·워터마크는 평가 항목에서 제외했다. "
        "37~39페이지 정형 표 셀은 이 평가에 포함하지 않는다. 캡처 속 표의 지정 문자열은 여기에서만 집계한다.",
        "",
        "**이미 출력을 살펴본 뒤 만든 회고적 진단 표본이다.** 무작위 표본이나 전체 이미지 정답지가 아니며, "
        "사전 등록·블라인드 테스트 또는 독립 검증 세트로 주장하지 않는다. 문자열 35개는 해당 캡처의 전체 내용이 아니다.",
        "",
        "정답과 좌표: [image_checks.json](../fixtures/image_checks.json). "
        "모든 입력 SHA-256, 캡처별 블록·이미지 경로·항목별 통과 여부: [실행 원자료](image_evaluation.json).",
        "",
        "| 캡처 | 페이지 | 대상 | 원본 확인 |",
        "|---|---:|---|---|",
    ]
    for c in structure["captures"]:
        page_preview = f"image_evaluation/source_previews/page_{c['page']:02d}.png"
        lines.append(f"| {c['id']} | {c['page']} | {c['name']} | [전체 페이지]({page_preview}) / "
                     f"[확대]({Path(c['preview']).relative_to('tests/reports').as_posix()}) |")
    lines += [
        "",
        "## 1. 추출 이미지 보존과 Markdown 연결",
        "",
        "원본 캡처 범위를 1,200×675 출력 좌표의 사각형으로 지정했다. "
        "저장된 이미지 파일의 이름에 기록된 crop 좌표와 실제 픽셀 크기가 일치하는지 검사하고, "
        "해당 이미지들이 덮는 원본 캡처 면적의 합집합을 계산했다. 분할·중복 이미지는 겹치는 면적을 중복 집계하지 않는다.",
        "",
        "보존 기준은 원본 캡처 면적의 **95% 이상**이다. 이는 외곽 여백 차이를 허용하기 위해 정한 이번 실습의 기준이며 "
        "표준 벤치마크 임계값이 아니다. 보존으로 집계한 출력 이미지들도 직접 열어 원본과 대조했다. "
        "픽셀 단위 화질 점수, 모든 작은 글자의 가독성, 중요 부분별 가중치는 측정하지 않았다.",
        "",
        "- 파일 보존: 저장된 추출 이미지 전체를 대상으로 한다.",
        "- Markdown 연결 보존: 최종 Markdown에서 실제 참조하는 추출 이미지들만 대상으로 한다.",
        "- 원본 PDF 뷰어는 별도다. 추출 이미지 연결이 실패해도 Streamlit의 원본 PDF에서는 캡처를 볼 수 있다.",
        "",
        "## 2. 캡처 내부 지정 문자열 보존",
        "",
        "페이지 전체에서 단어를 찾지 않고, 캡처와 겹치는 parsing_res_list 블록만 사용했다. "
        "블록 면적의 50% 이상 또는 캡처 면적의 50% 이상이 겹치면 해당 블록을 포함한다. "
        "Structure가 두 Jira 폼을 한 블록으로 합친 경우 그 블록이 두 캡처의 검사에 포함된다. "
        "따라서 이 검사는 캡처 영역의 문자열 존재를 확인하며, 정확한 버튼 위치나 표의 행·열 귀속은 검증하지 않는다.",
        "",
        "HTML 태그·대소문자·연속 공백·전각 문자·하이픈 형태를 정규화하고 지정 문자열을 정확히 비교했다. "
        "문자열 내부의 철자·숫자·밑줄·구두점은 보존한다. 텍스트는 단어 경계를 확인한다. "
        "날짜·숫자는 숫자 경계를 확인하므로 2631이 585263114033처럼 붙은 숫자열 안에 있으면 통과시키지 않는다.",
        "",
        "JSON 보존율 = 지정 문자열이 캡처 블록 내용에 있는 수 / 35. "
        "Markdown 텍스트 보존율 = 해당 블록 내용이 Markdown 텍스트에도 포함되고 지정 문자열이 남은 수 / 35. "
        "이미지 링크·파일 속 눈으로 읽을 수 있는 글자·오른쪽 기능 설명에 반복된 단어만으로는 통과시키지 않는다.",
        "",
        "불일치는 **오인식 또는 미출력**을 뜻한다. 이를 모두 인식 오류나 환각으로 세지 않는다. "
        "캡처 전체 문자 오류율, 출력 전체의 환각률, 필드-값 연결 정확도, 표 구조 TEDS, 이미지 의미 설명은 평가하지 않았다.",
        "",
        "## 총계",
        "",
        "| 항목 | Structure | VL |",
        "|---|---:|---:|",
    ]
    for label, key, denominator in [
        ("추출 이미지 파일 보존", "image_preserved", 7),
        ("Markdown 연결 이미지 보존", "image_linked", 7),
        ("JSON 지정 문자열 보존", "json_present", 35),
        ("Markdown 지정 문자열 보존", "markdown_present", 35),
    ]:
        lines.append(f"| {label} | {s[key]}/{denominator} ({s[key]/denominator:.1%}) | "
                     f"{v[key]}/{denominator} ({v[key]/denominator:.1%}) |")
    lines += [
        "",
        "## 캡처별 결과",
        "",
        "| 캡처 | S 파일 면적 / 연결 면적 | V 파일 면적 / 연결 면적 | JSON S / V | Markdown S / V |",
        "|---|---|---|---|---|",
    ]
    for sc, vc in zip(structure["captures"], vl["captures"]):
        sj = sum(x["json_present"] for x in sc["checks"])
        vj = sum(x["json_present"] for x in vc["checks"])
        sm = sum(x["markdown_present"] for x in sc["checks"])
        vm = sum(x["markdown_present"] for x in vc["checks"])
        lines.append(
            f"| {sc['id']} | {sc['image_coverage']:.1%} / {sc['linked_image_coverage']:.1%} | "
            f"{vc['image_coverage']:.1%} / {vc['linked_image_coverage']:.1%} | {sj}/5 / {vj}/5 | {sm}/5 / {vm}/5 |")
    lines += [
        "",
        "## 해석과 실제 오류 사례",
        "",
        "- Structure 9페이지는 캡처의 지정 문자열 5개를 JSON에 남겼지만 Markdown에는 이미지 링크로 전달했다. "
        "텍스트 전용 RAG는 이 이미지 속 글자를 읽지 않는다.",
        "- Structure 31페이지의 완전한 캡처 파일은 저장돼 있지만 Markdown에서 참조하지 않는다. "
        "이미지 파일 보존과 RAG 화면에 연결된 이미지 보존이 다른 사례다.",
        "- VL JSON의 use_ocr_for_image_block=False를 확인했다. 9·14·20페이지의 image 블록은 "
        "캡처를 보존하지만 내부 OCR 텍스트를 출력하지 않아 지정 문자열 점수는 0이다. "
        "이 설정을 켜서 재실행한 실험은 수행하지 않았다.",
        "- VL 22페이지는 Title Length와 상·하한 조건을 텍스트로 전달했지만, "
        "Raw Data의 title_tag·title_len을 [1514-549, 1514-149]로 잘못 출력했다.",
        "- VL 23페이지의 원본 날짜 2026-09-01은 2020-08-01로, "
        "31페이지에서는 여러 날짜가 2020-00-01로 출력됐다. 표 헤더도 손상됐다. "
        "많은 표 텍스트를 생성해도 지정 문자열 보존이 높다는 뜻은 아니다.",
        "- 일부 날짜나 버튼 단어가 캡처 어디에든 존재하면 문자열 보존 검사는 통과할 수 있다. "
        "예를 들어 14페이지 Create는 버튼 위치까지 확인하는 지표가 아니다.",
        "",
        "## 비교 조건과 다음 평가",
        "",
        "두 저장 결과는 같은 PDF, 페이지 폭 1,200px, 배치 1의 8GB 설정이다. "
        "레이아웃 분류·이미지 OCR 정책·출력 방식은 다르므로 현재 파이프라인 구성의 비교다. "
        "모델 자체의 캡처 OCR 능력을 비교하려면 같은 캡처 영역·입력 조건·출력 과제를 고정한 별도 실험이 필요하다.",
        "",
        "## 재현",
        "",
        "저장소 루트 PowerShell에서 실행한다. 기존 .venv의 Python과 Pillow만 사용하며 GPU 모델을 로드하지 않는다.",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -X utf8 -B .\\tests\\scripts\\evaluate_images.py",
        "```",
        "",
    ]
    lines += ["## 지정 문자열 35개 상세", "",
              "| 캡처 | 지정 문자열 | S JSON | V JSON | S Markdown | V Markdown |",
              "|---|---|---|---|---|---|"]
    mark = lambda passed: "일치" if passed else "불일치·미출력"
    for sc, vc in zip(structure["captures"], vl["captures"]):
        for a, b in zip(sc["checks"], vc["checks"]):
            expected = a["expected"].replace("|", "\\|")
            lines.append(
                f"| {sc['id']} | {expected} | {mark(a['json_present'])} | {mark(b['json_present'])} | "
                f"{mark(a['markdown_present'])} | {mark(b['markdown_present'])} |")
    (OUTPUT / "image_evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    write_report(evaluate())
