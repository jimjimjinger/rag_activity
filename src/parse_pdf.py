"""Pass a complete PDF directly to PP-StructureV3 and save each page."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "documents" / "B2B Dashboard Manual_v3.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "parsing" / "pp_structurev3")
    args = parser.parse_args()
    pdf_path = args.pdf.resolve(strict=True)
    if pdf_path.suffix.lower() != ".pdf":
        parser.error("--pdf must be a PDF file")
    runtime = Path(os.environ.get("LOCALAPPDATA", str(ROOT))) / "B2B-RAG"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(runtime / "paddle-cache"))
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("HF_HOME", str(runtime / "huggingface-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_pdf": str(pdf_path), "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "pipeline": "PP-StructureV3", "input_type": "pdf", "status": "processing",
        "total_pages": 0, "pages": [], "results": [],
    }

    def save_manifest():
        (output / "run_summary.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    try:
        from paddleocr import PPStructureV3
        settings = dict(
            device="cpu", cpu_threads=4, enable_mkldnn=False,
            layout_detection_model_name="PP-DocLayout-S",
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="en_PP-OCRv5_mobile_rec",
            text_recognition_batch_size=1,
            text_det_limit_side_len=1280, text_det_limit_type="max",
            wired_table_structure_recognition_model_name="SLANet_plus",
            use_region_detection=False,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, use_formula_recognition=False,
            use_chart_recognition=False, use_seal_recognition=False,
            use_table_recognition=True,
            # Table legends can be classified as footnotes: retain them.
            markdown_ignore_labels=["number", "header", "header_image", "footer", "footer_image"],
        )
        print("Loading local PP-StructureV3 models...", flush=True)
        pipeline = PPStructureV3(**settings)
        for result in pipeline.predict(input=str(pdf_path)):
            page = result["page_index"] + 1
            total = result["page_count"]
            if page != len(manifest["results"]) + 1 or not total or page > total:
                raise RuntimeError("Unexpected PDF page numbering from Paddle.")
            if manifest["total_pages"] not in (0, total):
                raise RuntimeError("Inconsistent PDF page count from Paddle.")
            manifest["total_pages"] = total
            manifest["pages"] = list(range(1, total + 1))
            page_dir = output / f"page_{page:02d}"
            page_dir.mkdir(exist_ok=True)
            # Explicit filenames avoid overwriting results from the same PDF.
            result.save_to_markdown(save_path=str(page_dir / f"page_{page:02d}.md"))
            manifest["results"].append({"page": page, "directory": page_dir.name})
            save_manifest()
            print(f"Saved page {page}/{total}", flush=True)
        if not manifest["results"] or len(manifest["results"]) != manifest["total_pages"]:
            raise RuntimeError("Paddle did not return every PDF page.")
        manifest["status"] = "complete"
        save_manifest()
        print(f"Saved results: {output}", flush=True)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        save_manifest()
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
