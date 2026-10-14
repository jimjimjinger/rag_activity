"""Run StructureV3 quality or PaddleOCR-VL on the same PDF for comparison."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "documents/B2B Dashboard Manual_v3.pdf"
OUTPUT = ROOT / "tests/paddleocr_vl"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=["vl", "structure"], default="vl")
    parser.add_argument("--device", choices=["cpu", "gpu:0"], default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pages", type=int, nargs="+", help="Analyze selected PDF page numbers; omit for all pages")
    parser.add_argument("--text-recognition-model", help="Override the Structure recognizer; use another output folder for another model")
    args = parser.parse_args()
    if args.pages and min(args.pages) < 1:
        parser.error("Page numbers start at 1")
    if args.text_recognition_model and args.pipeline != "structure":
        parser.error("--text-recognition-model is only for --pipeline structure")
    if args.pipeline == "vl":
        default_output = ROOT / "tests/paddleocr_vl_gpu" if args.device.startswith("gpu") else OUTPUT
    else:
        default_output = ROOT / ("tests/pp_structurev3_gpu" if args.device.startswith("gpu") else "tests/pp_structurev3_quality_cpu")
    output = (args.output or default_output).resolve()
    # Use the existing parser cache without running source-availability probes.
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "huggingface")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4" if args.device == "cpu" else "1")
    import numpy as np
    import paddle
    import pypdfium2 as pdfium
    import yaml
    import paddlex
    from paddleocr import PaddleOCRVL, PPStructureV3

    if args.device.startswith("gpu"):
        if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
            raise RuntimeError("GPU Paddle/CUDA is unavailable. Run scripts/setup-paddle-gpu.ps1 on the GPU computer.")
        paddle.set_device(args.device)
    framework_package = "paddlepaddle-gpu" if paddle.is_compiled_with_cuda() else "paddlepaddle"
    with pdfium.PdfDocument(str(PDF)) as pdf:
        total_pages = len(pdf)
    requested = sorted(set(args.pages or range(1, total_pages + 1)))
    if max(requested) > total_pages:
        parser.error(f"This PDF has only {total_pages} pages")
    settings = {"device": args.device, "cpu_threads": 4,
                "enable_mkldnn": False, "use_doc_orientation_classify": False,
                "use_doc_unwarping": False, "use_chart_recognition": False,
                "use_seal_recognition": False,
                "markdown_ignore_labels": ["number", "header", "header_image", "footer", "footer_image"]}
    if args.pipeline == "vl":
        pipeline_name, pipeline_class = "PaddleOCR-VL-1.6", PaddleOCRVL
        settings.update(pipeline_version="v1.6", use_queues=False)
    else:
        pipeline_name, pipeline_class = "PP-StructureV3", PPStructureV3
        settings.update(
            layout_detection_model_name="PP-DocLayout_plus-L",
            region_detection_model_name="PP-DocBlockLayout",
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name=args.text_recognition_model or "PP-OCRv5_server_rec",
            text_recognition_batch_size=1,
            text_det_limit_side_len=2000, text_det_limit_type="max",
            wired_table_structure_recognition_model_name="SLANeXt_wired",
            wireless_table_structure_recognition_model_name="SLANeXt_wireless",
            use_region_detection=True, use_textline_orientation=False,
            use_formula_recognition=False, use_table_recognition=True,
        )
    config_path = Path(paddlex.__file__).parent / f"configs/pipelines/{pipeline_name}.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def single_batch(value):
        if isinstance(value, dict):
            if "batch_size" in value:
                value["batch_size"] = 1
            for child in value.values():
                single_batch(child)
        elif isinstance(value, list):
            for child in value:
                single_batch(child)

    single_batch(config)
    digest = hashlib.sha256(PDF.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_summary.json"
    manifest = {"source_pdf": PDF.relative_to(ROOT).as_posix(), "source_sha256": digest,
                "pipeline": pipeline_name, "profile": args.pipeline,
                "device": args.device, "render_width": 2000, "batch_size": 1,
                "manual_corrections_applied": False, "settings": settings,
                "packages": {n: version(n) for n in (framework_package, "paddleocr", "paddlex")},
                "status": "initializing", "total_pages": total_pages,
                "requested_pages": requested, "results": []}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (previous["source_sha256"] != digest or previous["settings"] != settings
                or previous.get("profile", "vl") != args.pipeline):
            raise ValueError("Existing parser results use another PDF or configuration. Choose another output folder.")
        if previous.get("packages") != manifest["packages"]:
            raise ValueError("Parser package versions changed; use a separate output folder.")
        manifest["results"] = previous["results"]

    def save():
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_path)

    save()
    print(f"Loading {pipeline_name} ({args.device})...", flush=True)
    try:
        initialized = time.perf_counter()
        pipeline = pipeline_class(**settings, paddlex_config=config)
        manifest["initialization_seconds"] = round(time.perf_counter() - initialized, 2)
        # cpu_threads configures the layout predictor; set dynamic VLM threads too.
        if args.device == "cpu":
            paddle.framework.core.set_num_threads(settings["cpu_threads"])
            manifest["native_dynamic_threads"] = settings["cpu_threads"]
        else:
            manifest["gpu_name"] = paddle.device.cuda.get_device_name(0)
            manifest["cuda_build"] = paddle.version.cuda()
            manifest["gpu_total_memory_bytes"] = paddle.device.cuda.get_device_properties(0).total_memory
        manifest["status"] = "processing"
        save()
        with pdfium.PdfDocument(str(PDF)) as pdf:
            manifest["total_pages"] = len(pdf)
            completed = {r["page"]: r for r in manifest["results"]}
            for number in requested:
                i = number - 1
                destination = output / f"page_{number:02d}"
                markdown = destination / f"page_{number:02d}.md"
                if number in completed:
                    if (not markdown.with_suffix(".json").exists()
                            or hashlib.sha256(markdown.read_bytes()).hexdigest() != completed[number]["markdown_sha256"]):
                        raise ValueError(f"Saved page {number} changed; inspect it before resuming.")
                    continue
                print(f"Analyzing page {number}/{len(pdf)}...", flush=True)
                started = time.perf_counter()
                page = pdf[i]
                try:
                    bitmap = page.render(scale=2000 / page.get_width())
                    try:
                        image = bitmap.to_pil()
                        try:
                            rgb = np.array(image.convert("RGB"))
                        finally:
                            image.close()
                    finally:
                        bitmap.close()
                finally:
                    page.close()
                results = list(pipeline.predict(input=rgb[:, :, ::-1].copy()))
                if len(results) != 1:
                    raise ValueError(f"Expected one result for page {number}, got {len(results)}")
                destination.mkdir(parents=True, exist_ok=True)
                results[0].save_to_markdown(save_path=str(markdown))
                results[0].save_to_json(save_path=str(destination / f"page_{number:02d}.json"))
                elapsed = round(time.perf_counter() - started, 2)
                manifest["results"].append({"page": number, "seconds": elapsed,
                                            "markdown_sha256": hashlib.sha256(markdown.read_bytes()).hexdigest()})
                save()
                print(f"Saved page {number}/{len(pdf)} ({elapsed}s)", flush=True)
        manifest["status"] = "complete" if len(manifest["results"]) == total_pages else "partial"
        manifest.pop("error", None)
        save()
        print(f"Saved {len(manifest['results'])}/{total_pages} pages ({manifest['status']}): {output}", flush=True)
    except KeyboardInterrupt:
        manifest["status"] = "stopped_by_user"
        save()
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
