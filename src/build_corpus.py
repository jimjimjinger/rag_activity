"""Split clear feature sections; keep ambiguous Markdown pages together."""
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
    """Link only images already produced and referenced by Paddle."""
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
    """Apply only PDF-verified edits to the exact Markdown version reviewed."""
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
    """Split explicit feature headings, preserving page context and procedures.

    These are conservative rules for this manual, not semantic/LLM splitting.
    Tables, broken numbering and empty sections keep the whole page together.
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
    # E.g. '3Suggested Questions' is an uncertain OCR boundary.
    if re.search(r"^(?:##\s+)?\d+[A-Za-z가-힣]", content[headings[0].start():], re.M):
        return whole
    numbers = [int(m[1]) for h in headings
               if (m := re.match(r"(\d+)[.)]?\s+\S", h[1]))]
    # A continued page can start at item 3 or 5; retain its original numbers.
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


def build():
    run = json.loads((PARSED / "run_summary.json").read_text(encoding="utf-8"))
    source = ROOT / "documents" / Path(run["source_pdf"]).name
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != run["source_sha256"]:
        raise ValueError("The PDF changed; rerun parsing first.")
    pages = list(range(1, run["total_pages"] + 1))
    completed = {item["page"] for item in run.get("results", [])}
    if not pages or run["status"] != "complete" or run["pages"] != pages or completed != set(pages):
        raise ValueError("Complete parsing of every PDF page before building the corpus.")
    corrections = {}
    if CORRECTIONS.exists():
        reviewed = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
        if reviewed["source_sha256"] != digest:
            raise ValueError("Correction source PDF changed; review markdown_corrections.json first.")
        corrections = reviewed["pages"]
    chunks = []
    for page in pages:
        markdown = PARSED / f"page_{page:02d}/page_{page:02d}.md"
        content = markdown.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Page {page} has empty Markdown; check the Paddle result.")
        content = apply_corrections(content, page, corrections)
        images = page_images(markdown, content)
        corrected = str(page) in corrections
        for number, part in enumerate(split_page(content), start=1):
            chunks.append({
                "id": f"p{page}-s{number}", "page": page, **part,
                "images": images,
                "origin": "PP-StructureV3 Markdown" + (" + PDF-verified corrections" if corrected else ""),
                "correction_source": CORRECTIONS.relative_to(ROOT).as_posix() if corrected else None,
                "source_markdown": markdown.relative_to(ROOT).as_posix(),
                "source_pdf": source.relative_to(ROOT).as_posix(), "source_sha256": digest,
            })
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    temporary = DESTINATION.with_suffix(".tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(DESTINATION)
    print(f"Saved {len(chunks)} Markdown chunks: {DESTINATION}")
    return chunks


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    build()
