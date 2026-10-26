"""Recognize exactly three preselected PM209 Table crops; never modify the dataset."""

import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "PM209/datasets/PM209"
OUT = ROOT / "PM209/reports/paddle_table_sample_20261022"
RUNTIME = ROOT / ".runtime/paddle-tables-sample"

# Keep the figure-generation environment and model service untouched.
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
os.environ["FLAGS_gpu_memory_limit_mb"] = "6144"
# Paddle's Windows C++ model loader cannot open this workspace's Korean path.
# Model weights use an ASCII-only per-user cache; data and outputs stay in the repo.
os.environ["PADDLE_PDX_CACHE_HOME"] = str(Path(os.environ["LOCALAPPDATA"]) / "PM209-PaddleTables/paddlex")
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
os.environ["OMP_NUM_THREADS"] = "1"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protected_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for folder in (DATASET / "data", DATASET / "data/vlm")
            for p in (folder / f"{s}.jsonl" for s in ("train", "val", "test"))}


def crop_region(sample):
    from PIL import Image
    image_path = (DATASET / sample["image_filename"]).resolve()
    if not image_path.is_relative_to(DATASET.resolve()):
        raise ValueError("Image path is outside the dataset")
    shape = sample["region"]["shape"]
    x, y, w, h = (float(shape[k]) for k in ("x", "y", "width", "height"))
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
        raise ValueError("Invalid Table shape")
    filename = str(image_path)
    if os.name == "nt":
        filename = "\\\\?\\" + filename
    with Image.open(filename) as page_image:
        box = [max(0, math.floor(x)), max(0, math.floor(y)),
               min(page_image.width, math.ceil(x + w)),
               min(page_image.height, math.ceil(y + h))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Empty Table crop")
        image = page_image.crop(box).convert("RGB")
    if box != sample["crop_box"]:
        raise ValueError("Crop differs from the inspected sample")
    return image


class GPUMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.rows = []
        self.phase = "initialization"
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        with (OUT / "gpu.jsonl").open("w", encoding="utf-8") as stream:
            while not self.stop.is_set():
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,utilization.gpu",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
                    values = [int(v.strip()) for v in result.stdout.strip().splitlines()[0].split(",")]
                    row = dict(at=time.time(), phase=self.phase, memory_mib=values[0],
                               temperature_c=values[1], utilization_percent=values[2])
                    self.rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                except Exception as exc:
                    stream.write(json.dumps({"error": str(exc), "phase": self.phase}) + "\n")
                    stream.flush()
                self.stop.wait(0.5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=12)


def main():
    import numpy as np
    import paddle
    from paddleocr import TableRecognitionPipelineV2

    samples = json.loads((OUT / "samples.json").read_text(encoding="utf-8"))
    assert len(samples) == 3 and {s["kind"] for s in samples} == {"simple", "merged", "dense"}
    assert len({s["region"]["id"] for s in samples}) == 3
    assert all(s["region"]["structure"] == "Table" for s in samples)
    if any((OUT / f'{s["kind"]}.html').exists() for s in samples):
        raise FileExistsError("Existing sample outputs are preserved; do not rerun over them")

    # Read full pages for preservation only. Only image arrays reach predict().
    pages = {}
    for split in {s["split"] for s in samples}:
        wanted = {s["page_id"] for s in samples if s["split"] == split}
        with (DATASET / "data/vlm" / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                page = json.loads(line)
                if page["id"] in wanted:
                    pages[(split, page["id"])] = page
    for sample in samples:
        page = pages[(sample["split"], sample["page_id"])]
        assert page["image_filename"] == sample["image_filename"]
        assert next(b for b in page["bounding_boxes"] if b["id"] == sample["region"]["id"]) == sample["region"]

    before = protected_hashes()
    write_json(OUT / "protected_before.json", before)
    assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() >= 1
    paddle.set_device("gpu:0")
    # Force an actual GPU kernel before claiming GPU execution.
    tensor = paddle.ones([8, 8])
    assert float(paddle.matmul(tensor, tensor).mean().item()) == 8.0
    write_json(OUT / "environment.json", {
        "versions": {p: importlib.metadata.version(p) for p in ("paddlepaddle-gpu", "paddleocr", "paddlex", "numpy")},
        "cuda": paddle.version.cuda(), "cudnn": paddle.version.cudnn(),
        "device": paddle.get_device(), "gpu_name": paddle.device.cuda.get_device_name(),
        "gpu_kernel_check": "passed", "allocator": "auto_growth", "allocator_limit_mb": 6144,
        "qa_input": False, "sample_count": 3,
    })
    settings = dict(
        device="gpu:0", use_layout_detection=False,
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_ocr_model=True, text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec", text_recognition_batch_size=1,
        text_det_limit_side_len=1280, text_det_limit_type="max",
        enable_hpi=False, use_tensorrt=False, enable_mkldnn=False, cpu_threads=4,
    )
    predict_settings = dict(use_table_orientation_classify=False,
                            use_e2e_wired_table_rec_model=False,
                            use_e2e_wireless_table_rec_model=False,
                            use_ocr_results_with_table_cells=True)
    write_json(OUT / "settings.json", {"init": settings, "predict": predict_settings})
    monitor = GPUMonitor()
    monitor.thread.start()
    metrics = {"tables": [], "pipeline": "TableRecognitionPipelineV2"}
    extracted = []
    try:
        start = time.perf_counter()
        pipeline = TableRecognitionPipelineV2(**settings)
        metrics["initialization_seconds"] = time.perf_counter() - start
        pipeline.export_paddlex_config_to_yaml(str(OUT / "pipeline.yaml"))
        for sample in samples:
            kind = sample["kind"]
            monitor.phase = kind
            row = {"kind": kind, "id": sample["region"]["id"]}
            start = time.perf_counter()
            try:
                image = crop_region(sample)
                # PNGs are for human inspection. Inference uses the in-memory BGR array.
                image.save(OUT / f"{kind}.png")
                pixels = np.asarray(image)[:, :, ::-1].copy()
                results = pipeline.predict(pixels, **predict_settings)
                paddle.device.cuda.synchronize()
                row["inference_seconds"] = time.perf_counter() - start
                if len(results) != 1:
                    raise ValueError(f"Expected one crop result, got {len(results)}")
                result = results[0]
                result.save_to_json(str(OUT / f"{kind}.raw.json"))
                tables = result["table_res_list"]
                if len(tables) != 1:
                    raise ValueError(f"Expected one Table, got {len(tables)}")
                html = tables[0]["pred_html"]
                if not isinstance(html, str) or "<table" not in html.lower() or "</table>" not in html.lower():
                    raise ValueError("No complete table HTML returned")
                # Store the exact pipeline string; no rewriting or text generation.
                (OUT / f"{kind}.html").write_text(html, encoding="utf-8")
                enriched = {**copy.deepcopy(sample["region"]), "table_content": html}
                extracted.append({"split": sample["split"], "page_id": sample["page_id"],
                                  "image_filename": sample["image_filename"], "table": enriched})
                row.update(html_output_success=True, html_characters=len(html), error=None,
                           semantic_accuracy="pending_manual_image_comparison")
            except Exception:
                row.update(html_output_success=False, error=traceback.format_exc())
            row["total_seconds"] = time.perf_counter() - start
            readings = [r for r in monitor.rows if r["phase"] == kind]
            row["observed_gpu_peak_mib"] = max((r["memory_mib"] for r in readings), default=None)
            metrics["tables"].append(row)
            write_json(OUT / "run_metrics.json", metrics)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        with (OUT / "tables.sample.jsonl").open("x", encoding="utf-8") as stream:
            for result in extracted:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        with (OUT / "pages.sample.jsonl").open("x", encoding="utf-8") as stream:
            for sample in samples:
                page = copy.deepcopy(pages[(sample["split"], sample["page_id"])] )
                for result in extracted:
                    if result["page_id"] == page["id"]:
                        for region in page["bounding_boxes"]:
                            if region["id"] == result["table"]["id"]:
                                region["table_content"] = result["table"]["table_content"]
                stream.write(json.dumps(page, ensure_ascii=False) + "\n")
    finally:
        monitor.close()
        after = protected_hashes()
        write_json(OUT / "preservation.json", {"before": before, "after": after,
                   "all_six_files_unchanged": before == after})
        metrics["observed_gpu_peak_mib"] = max((r["memory_mib"] for r in monitor.rows), default=None)
        metrics["observed_gpu_peak_temperature_c"] = max((r["temperature_c"] for r in monitor.rows), default=None)
        write_json(OUT / "run_metrics.json", metrics)
        assert before == after, "Protected dataset files changed"


if __name__ == "__main__":
    main()
