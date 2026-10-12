"""Build one searchable chunk per page from Paddle Markdown output."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PARSED = ROOT / "outputs/parsing/pp_structurev3"
DESTINATION = ROOT / "data/rag/corpus.jsonl"


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
    chunks = []
    for page in pages:
        markdown = PARSED / f"page_{page:02d}/page_{page:02d}.md"
        content = markdown.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Page {page} has empty Markdown; check the Paddle result.")
        chunks.append({
            "id": f"p{page}-markdown", "page": page, "content": content,
            "origin": "PP-StructureV3 Markdown", "source_markdown": str(markdown),
            "source_pdf": str(source), "source_sha256": digest,
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
