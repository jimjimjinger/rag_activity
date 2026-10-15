"""명확한 기능 항목은 청크로 나누고, 경계가 불명확한 Markdown 페이지는 통째로 유지한다."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "outputs/parsing/pp_structurev3"
DESTINATION = ROOT / "data/rag/corpus.jsonl"
CORRECTIONS = ROOT / "data/rag/markdown_corrections.json"


class ImageReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            source = dict(attrs).get("src")
            if source:
                self.paths.append(source)


def page_images(markdown, content):
    """Paddle이 생성하고 Markdown에서 참조한 이미지만 연결한다."""
    parser = ImageReferences()
    parser.feed(content)
    references = parser.paths + re.findall(r"!\[[^\]]*\]\(<([^>]+)>\)", content)
    references += re.findall(r"!\[[^\]]*\]\(([^\s)]+)\)", content)
    candidates = []
    for name in references:
        if re.match(r"^[a-zA-Z][\w+.-]*:", name) or name.startswith("//"):
            continue
        path = (markdown.parent / name).resolve()
        path.relative_to(markdown.parent.resolve())
        candidates.append(path)
    images, seen = [], set()
    for path in candidates:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        if relative in seen:
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        images.append({"path": relative, "sha256": actual})
        seen.add(relative)
    return images


def apply_corrections(content, page, corrections):
    """검토한 Markdown 버전이 일치할 때만 원본 PDF로 확인한 보정을 적용한다."""
    entry = corrections.get(str(page))
    if entry is None:
        return content
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if digest != entry["markdown_sha256"]:
        raise ValueError(f"Page {page} Markdown changed; review markdown_corrections.json first.")
    for edit in entry["replacements"]:
        if content.count(edit["before"]) != 1:
            raise ValueError(f"Page {page} correction must match exactly once.")
        content = content.replace(edit["before"], edit["after"], 1)
    return content


def split_page(content):
    """페이지 문맥과 절차를 보존하면서 명확한 기능 제목을 기준으로 분리한다.

    이 매뉴얼에 맞춘 보수적인 규칙이며, 의미 분석이나 LLM으로 분리하지 않는다.
    표가 있거나 번호가 깨졌거나 설명이 빈 항목이 있으면 페이지 전체를 유지한다.
    """
    whole = [{"section": "Page", "content": content, "chunk_type": "page"}]
    if re.search(r"<table\b|^\s*\|", content, re.I | re.M):
        return whole
    headings = list(re.finditer(r"^## (?!#)(.+?)\s*$", content, re.M))
    first = next((i for i, h in enumerate(headings)
                  if re.match(r"[1-9]\d*[.)]?\s+\S", h[1])), None)
    if first is None or len(headings[first:]) < 2:
        return whole
    headings = headings[first:]
    # '3Suggested Questions'처럼 번호와 제목이 붙으면 OCR 경계가 불명확하다고 판단한다.
    if re.search(r"^(?:##\s+)?\d+[A-Za-z가-힣]", content[headings[0].start():], re.M):
        return whole
    numbers = [int(m[1]) for h in headings
               if (m := re.match(r"(\d+)[.)]?\s+\S", h[1]))]
    # 이어지는 페이지는 3번이나 5번부터 시작할 수 있으므로 원본 번호를 유지한다.
    if numbers != list(range(numbers[0], numbers[0] + len(numbers))) or any(h[1].isdigit() for h in headings):
        return whole
    common = content[:headings[0].start()].strip()
    chunks = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        body = content[heading.end():end].strip()
        if not body:
            return whole
        chunks.append({
            "section": heading[1].strip(),
            "content": "\n\n".join(part for part in
                                    (common, content[heading.start():end].strip()) if part),
            "chunk_type": "section",
        })
    return chunks


def build(parsed=PARSED, destination=DESTINATION, corrections_path=CORRECTIONS):
    parsed, destination = Path(parsed).resolve(), Path(destination).resolve()
    run = json.loads((parsed / "run_summary.json").read_text(encoding="utf-8"))
    source = ROOT / "documents" / Path(run["source_pdf"]).name
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != run["source_sha256"]:
        raise ValueError("The PDF changed; rerun parsing first.")
    pages = list(range(1, run["total_pages"] + 1))
    completed = {item["page"] for item in run.get("results", [])}
    if not pages or run["status"] != "complete" or run.get("pages", pages) != pages or completed != set(pages):
        raise ValueError("Complete parsing of every PDF page before building the corpus.")
    corrections = {}
    if corrections_path is not None and Path(corrections_path).exists():
        corrections_path = Path(corrections_path).resolve()
        reviewed = json.loads(corrections_path.read_text(encoding="utf-8"))
        if reviewed["source_sha256"] != digest:
            raise ValueError("Correction source PDF changed; review markdown_corrections.json first.")
        corrections = reviewed["pages"]
    chunks = []
    records = {item["page"]: item for item in run["results"]}
    for page in pages:
        markdown = parsed / f"page_{page:02d}/page_{page:02d}.md"
        raw = markdown.read_bytes()
        expected_hash = records[page].get("markdown_sha256")
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(f"Page {page} Markdown changed after parsing.")
        content = raw.decode("utf-8").strip()
        if not content:
            raise ValueError(f"Page {page} has empty Markdown; check the Paddle result.")
        content = apply_corrections(content, page, corrections)
        images = page_images(markdown, content)
        corrected = str(page) in corrections
        for number, part in enumerate(split_page(content), start=1):
            chunks.append({
                "id": f"p{page}-s{number}", "page": page, **part,
                "images": images,
                "origin": run["pipeline"] + " Markdown" + (" + PDF-verified corrections" if corrected else ""),
                "correction_source": corrections_path.relative_to(ROOT).as_posix() if corrected else None,
                "source_markdown": markdown.relative_to(ROOT).as_posix(),
                "source_pdf": source.relative_to(ROOT).as_posix(), "source_sha256": digest,
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(destination)
    print(f"Saved {len(chunks)} Markdown chunks: {destination}")
    return chunks


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed", type=Path, default=PARSED)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--no-corrections", action="store_true")
    args = parser.parse_args()
    if args.no_corrections and args.corrections:
        parser.error("Choose --corrections or --no-corrections")
    if args.parsed.resolve() != PARSED.resolve() and args.output.resolve() == DESTINATION.resolve():
        parser.error("Choose a separate --output for experimental parser results")
    corrections = args.corrections
    if not args.no_corrections and corrections is None and args.parsed.resolve() == PARSED.resolve():
        corrections = CORRECTIONS
    build(args.parsed, args.output, corrections)
