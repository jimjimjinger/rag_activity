"""문서 분석 출력이나 RAG 데이터를 보정하지 않고 원본 PDF로 확인한 사례를 점검한다."""
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import statistics
import unicodedata

ROOT = Path(__file__).resolve().parents[2]


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = unescape(re.sub(r"<[^>]+>", " ", text))
    text = text.replace("’", "'").replace("‘", "'").replace("**", "")
    return re.sub(r"\s+", " ", text).strip().casefold()


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.rows, self.row, self.cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.rows = []
        elif tag == "tr" and self.rows is not None:
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(normalize("".join(self.cell)))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
        elif tag == "table" and self.rows is not None:
            self.tables.append(self.rows)
            self.rows = None


def evaluate():
    reference = json.loads((ROOT / "tests/fixtures/reference_checks.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ROOT / reference["source_pdf"]).read_bytes()).hexdigest()
    if digest != reference["source_sha256"]:
        raise ValueError("Reference checks belong to a different PDF.")
    report = {"source_sha256": digest, "scope": reference["review_method"], "models": {}}
    for key, name in (("L", "structure_cpu"), ("S", "structure_gpu_8gb"), ("V", "vl_gpu_8gb")):
        folder = ROOT / "tests/results" / name
        manifest_path = folder / "run_summary.json"
        if not manifest_path.exists():
            report["models"][key] = {"status": "not_started"}
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_sha256"] != digest:
            raise ValueError(f"{key}: PDF hash mismatch")
        records = {item["page"]: item for item in manifest["results"]}
        pages = {}
        for page, record in records.items():
            path = folder / f"page_{page:02d}/page_{page:02d}.md"
            if "markdown_sha256" in record and hashlib.sha256(path.read_bytes()).hexdigest() != record["markdown_sha256"]:
                raise ValueError(f"{key}: Markdown changed on page {page}")
            pages[page] = path.read_text(encoding="utf-8")
        result = {"folder": folder.relative_to(ROOT).as_posix(), "memory_profile": manifest.get("memory_profile", "standard"),
                  "status": manifest["status"], "completed_pages": len(pages), "text": [], "headings": [], "tables": []}
        times = [item["seconds"] for item in records.values() if "seconds" in item]
        if times:
            result["timing"] = {"page_seconds_sum": round(sum(times), 2), "median_seconds": round(statistics.median(times), 2),
                                "mean_seconds": round(statistics.mean(times), 2), "initialization_seconds": manifest.get("initialization_seconds")}
        for check in reference["text_checks"]:
            content = pages.get(check["page"])
            passed = normalize(check["text"]) in normalize(content) if content is not None else None
            result["text"].append({**check, "passed": passed})
        for check in reference["heading_checks"]:
            content = pages.get(check["page"])
            pattern = rf"^#{{1,6}}\s+{check['number']}[.)]?\s+{re.escape(check['title'])}\s*$"
            passed = bool(re.search(pattern, content.replace("**", ""), re.M | re.I)) if content is not None else None
            result["headings"].append({**check, "passed": passed})
        for number, expected in reference["tables"].items():
            content = pages.get(int(number))
            if content is None:
                result["tables"].append({"page": int(number), "available": False})
                continue
            parser = Tables()
            parser.feed(content)
            actual = next((table for table in parser.tables if table and "monitoring item" in " ".join(table[0])), [])
            mismatches, correct = [], 0
            for row, cells in enumerate(expected):
                for column, text in enumerate(cells):
                    got = actual[row][column] if row < len(actual) and column < len(actual[row]) else None
                    if got == normalize(text):
                        correct += 1
                    else:
                        mismatches.append({"row": row + 1, "column": column + 1, "expected": text, "actual": got})
            result["tables"].append({"page": int(number), "available": True,
                                     "expected_shape": [len(expected), len(expected[0])],
                                     "actual_row_lengths": [len(row) for row in actual],
                                     "correct_cells": correct, "expected_cells": sum(map(len, expected)), "mismatches": mismatches})
        report["models"][key] = result
    destination = ROOT / "tests/reports/analysis_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, result in report["models"].items():
        if "text" not in result:
            print(key, result["status"])
            continue
        print(key, result["status"], "pages", result["completed_pages"])
        for category in ("text", "headings"):
            checks = [check for check in result[category] if check["passed"] is not None]
            print(category, sum(check["passed"] for check in checks), "/", len(checks))
        print("table cells", sum(table.get("correct_cells", 0) for table in result["tables"]), "/", sum(table.get("expected_cells", 0) for table in result["tables"]))


if __name__ == "__main__":
    evaluate()
