"""보정 전 경량 Structure, 고정밀 후보 Structure, VL의 분석 결과를 비교한다."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "tests/reports"
sys.path.insert(0, str(ROOT / "src"))
from build_corpus import split_page


def compare():
    folders = {"L": ("Structure 경량: 기존 결과", ROOT / "tests/results/structure_cpu"),
               "S": ("Structure GPU 후보", ROOT / "tests/results/structure_gpu_8gb"),
               "V": ("PaddleOCR-VL-1.6", ROOT / "tests/results/vl_gpu_8gb")}
    manifests, statistics = {}, {}
    source_hash = None
    total_pages = 42
    for key, (name, folder) in folders.items():
        path = folder / "run_summary.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        manifests[key] = manifest
        if manifest:
            if source_hash is None:
                source_hash = manifest["source_sha256"]
                total_pages = manifest["total_pages"]
            elif manifest["source_sha256"] != source_hash:
                raise ValueError(f"{name} uses another PDF")
        completed = {r["page"]: r for r in manifest.get("results", [])}
        chunks, pages = [], {}
        for markdown in sorted(folder.glob("page_*/page_*.md")):
            page = int(markdown.stem.split("_")[1])
            if page not in completed:
                continue
            record = completed[page]
            if key != "L":
                if (not markdown.with_suffix(".json").exists()
                        or hashlib.sha256(markdown.read_bytes()).hexdigest() != record["markdown_sha256"]):
                    raise ValueError(f"{name}: saved page {page} changed or its JSON is missing")
            content = markdown.read_text(encoding="utf-8").strip()
            parts = split_page(content)
            pages[page] = {"characters": len(content), "chunks": len(parts),
                           "seconds": record.get("seconds", "—"),
                           "path": "../" + markdown.relative_to(ROOT / "tests").as_posix()}
            for number, part in enumerate(parts, 1):
                chunks.append({"id": f"p{page}-s{number}", "page": page, **part,
                               "source_markdown": markdown.relative_to(ROOT).as_posix(),
                               "manual_corrections_applied": False})
        if folder.exists():
            (folder / "corpus.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
        statistics[key] = pages

    if source_hash != hashlib.sha256((ROOT / "documents/B2B Dashboard Manual_v3.pdf").read_bytes()).hexdigest():
        raise ValueError("The source PDF changed after the recorded analyses")
    full = all(manifests[k].get("status") == "complete"
               and set(statistics[k]) == set(range(1, total_pages + 1)) for k in folders)
    lines = ["# 문서 분석 결과 비교", "",
             "세 구성의 전체 분석 완료." if full else "전체 비교는 미완료입니다. 아직 없는 결과를 성능 수치로 해석하지 않습니다.", "",
             "원본 대조 해석은 [분석 보고서](analysis_review.md), 현재 실행 설정은 [실험 안내](../README.md)를 참고하세요.", "",
             "같은 [원본 PDF](../../documents/B2B%20Dashboard%20Manual_v3.pdf)의 보정 전 결과입니다. 수동 보정은 적용하지 않았습니다.", "",
             "| 기호 | 구성 | 장치 | 상태 | 완료 페이지 | 보정 전 청크 수 |",
             "|---|---|---|---|---|---|"]
    for key, (name, folder) in folders.items():
        manifest, pages = manifests[key], statistics[key]
        lines.append(f"| {key} | {name} | {manifest.get('device', '—')} | {manifest.get('status', 'not_started')} | {len(pages)}/{total_pages} | {sum(p['chunks'] for p in pages.values())} |")
    lines += ["", "L은 기존 CPU 경량 결과이며 S·V는 같은 GPU에서 완료한 8GB 프로필 결과입니다.",
              "8GB 프로필은 페이지 폭 1,200픽셀, 배치 1로 실행합니다. 기존 결과와 입력 조건이 다르므로 구성 전체의 비교입니다.",
              "현재 표시된 렌더링 폭 L / S / V: " + " / ".join(str(manifests[k].get("render_width", "기존 기록 확인")) for k in folders) + " px.",
              "각 실행의 입력·생성·메모리 조건은 결과 폴더의 run_summary.json에 기록합니다.",
              "L의 과거 시간과 새 GPU 실행 시간을 직접적인 모델 속도 비교로 사용하지 않습니다.",
              "S·V의 실행 시간도 모델·엔진·예열 조건과 함께 읽습니다. 초기 로드·다운로드 시간은 페이지별 시간과 별도입니다.", "",
              "각 corpus.jsonl은 같은 split_page() 규칙을 적용했습니다. 제목 표기 형식도 청크 수에 영향을 줍니다.",
              "글자 수·청크 수·처리 시간은 정확도 점수가 아닙니다. PDF와 대조한 누락·오인식·표 구조·읽기 순서를 기록하세요.", "",
              "| 페이지 | L 경량 | S GPU 후보 | V VL | 글자 수 L / S / V | 청크 수 L / S / V | 페이지 시간(초) L / S / V |",
              "|---|---|---|---|---|---|---|"]
    for page in range(1, total_pages + 1):
        values = [statistics[k].get(page) for k in folders]
        links = [f"[보기]({v['path']})" if v else "아직 없음" for v in values]
        cells = [" / ".join(str(v[field]) if v else "—" for v in values)
                 for field in ("characters", "chunks", "seconds")]
        lines.append(f"| {page} | {' | '.join(links + cells)} |")
    lines += ["", "## 우선 눈으로 비교할 페이지", "",
              "- 1: 한글 메모와 날짜. 영어용·중영일용 인식기의 지원 언어 차이도 함께 기록",
              "- 9: Suggested Questions 제목과 본문 경계", "- 11·22: Page Type 설명 문장 누락 여부",
              "- 15: Strategic Task와 Applied Area 항목의 분리", "- 20: Page Details의 번호와 읽기 순서",
              "- 23·31: Search·Download 제목과 설명", "- 24·32: 5·6번 항목의 설명 누락 여부",
              "- 37~39: 표의 헤더·행·열·값 연결", "",
              "32페이지 Add item처럼 원본의 표현이 혼동을 주는 부분도 PDF와 대조해 모델 오류와 구분합니다.",
              "모델 설정과 실행 순서는 [실험 안내](../README.md)를 참고하세요."]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for key, (name, _) in folders.items():
        print(f"{key} {name}: {len(statistics[key])}/{total_pages} pages")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    compare()
