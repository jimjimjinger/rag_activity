"""Paddle TableRecognitionPipelineV2 helpers; dataset fields are never corrected."""

from collections import Counter
import copy
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "PM209/datasets/PM209"
OUTPUT = DATASET / "data/paddle_tables_html"
SAMPLE = ROOT / "PM209/reports/paddle_table_sample_20261022"
SPLITS = ("train", "val", "test")


def now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_json(path, data):
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def key_for(split, page_id, region_id):
    raw = json.dumps([split, page_id, region_id], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def settings():
    return read_json(SAMPLE / "settings.json")


def config_hash():
    config = {"settings": settings(), "versions": read_json(SAMPLE / "environment.json")["versions"]}
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def configure_environment():
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
    os.environ["FLAGS_gpu_memory_limit_mb"] = "6144"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(Path(os.environ["LOCALAPPDATA"]) / "PM209-PaddleTables/paddlex")
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "BOS"
    os.environ["OMP_NUM_THREADS"] = "1"


def hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for folder in (DATASET / "data", DATASET / "data/vlm")
            for p in (folder / f"{s}.jsonl" for s in SPLITS)}


def inventory():
    from describe_images import page_pairs, FIGURE_TYPES
    targets, counts = [], {}
    for split in SPLITS:
        count = Counter(pages=0, regions=0, qa=0, descriptions=0, tables=0)
        for _, page in page_pairs(DATASET / "data" / f"{split}.jsonl", DATASET / "data/vlm" / f"{split}.jsonl"):
            count.update(pages=1, regions=len(page["bounding_boxes"]), qa=len(page.get("qa_data", [])))
            for region in page["bounding_boxes"]:
                if region["structure"] in FIGURE_TYPES and region.get("description"):
                    count["descriptions"] += 1
                if region["structure"] == "Table":
                    if "table_content" in region:
                        raise ValueError("The protected figure baseline already has table_content")
                    count["tables"] += 1
                    targets.append(dict(split=split, page_id=page["id"], image_filename=page["image_filename"],
                                        region=region, key=key_for(split, page["id"], region["id"])))
        counts[split] = dict(count)
    if len({t["key"] for t in targets}) != len(targets):
        raise ValueError("Duplicate Table identity")
    return targets, counts


def image_path(target):
    path = (DATASET / target["image_filename"]).resolve()
    if not path.is_relative_to(DATASET.resolve()):
        raise ValueError("Image is outside dataset")
    return Path("\\\\?\\" + str(path)) if os.name == "nt" else path


def crop(target):
    from PIL import Image
    s = target["region"]["shape"]
    x, y, w, h = (float(s[k]) for k in ("x", "y", "width", "height"))
    if not all(math.isfinite(v) for v in (x, y, w, h)) or min(w, h) <= 0:
        raise ValueError("Invalid Table coordinates")
    with Image.open(image_path(target)) as image:
        box = [max(0, math.floor(x)), max(0, math.floor(y)),
               min(image.width, math.ceil(x + w)), min(image.height, math.ceil(y + h))]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Table crop is empty")
        return image.crop(box).convert("RGB"), box


class TableHTML(HTMLParser):
    """Check output framing, not semantic accuracy or a preferred table width."""
    def __init__(self):
        super().__init__()
        self.stack, self.text = [], []
        self.rows = self.cells = self.tables = self.empty = 0
        self.cell_text = None
        self.spans = []

    def handle_starttag(self, tag, attrs):
        if tag in ("table", "thead", "tbody", "tfoot", "tr", "td", "th"):
            self.stack.append(tag)
        if tag == "table":
            self.tables += 1
        if tag == "tr":
            self.rows += 1
        if tag in ("td", "th"):
            self.cells += 1
            self.cell_text = []
            attrs = dict(attrs)
            for name in ("rowspan", "colspan"):
                if name in attrs:
                    value = int(attrs[name])
                    if value < 1:
                        raise ValueError("Non-positive span")
                    if value > 1:
                        self.spans.append([name, value])

    def handle_endtag(self, tag):
        if tag in ("table", "thead", "tbody", "tfoot", "tr", "td", "th"):
            if not self.stack or self.stack.pop() != tag:
                raise ValueError("Unbalanced table HTML")
        if tag in ("td", "th"):
            if not "".join(self.cell_text or []).strip():
                self.empty += 1
            self.cell_text = None

    def handle_data(self, data):
        self.text.append(data)
        if self.cell_text is not None:
            self.cell_text.append(data)


def inspect_html(content, raw=None):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty HTML")
    parsed = TableHTML()
    parsed.feed(content)
    parsed.close()
    if parsed.stack or parsed.tables != 1 or not parsed.rows or not parsed.cells:
        raise ValueError("No complete single table in output")
    text = " ".join(parsed.text)
    flags = []
    if re.search(r"°℃|℃C|°°|\d\s*C\b", text):
        flags.append("suspect_temperature_notation")
    if re.search(r"°|℃|\b(?:k?W|k?V|mA|Hz|kHz|MHz|mm|cm|kg|metres|yards)\b|%", text, re.I):
        flags.append("units_need_review")
    if re.search(r"\b(?:press|button|remote control|mode)\b|On/Off", text, re.I):
        flags.append("button_or_mode_icon_check")
    if parsed.empty:
        flags.append("empty_cells_check")
    if parsed.spans:
        flags.append("merged_cells_check")
    scores = []
    if raw:
        scores = raw.get("overall_ocr_res", {}).get("rec_scores", [])
    if scores and min(scores) < 0.90:
        flags.append("ocr_confidence_below_0_90")
    return dict(rows=parsed.rows, cells=parsed.cells, empty_cells=parsed.empty,
                spans=parsed.spans, flags=flags, min_ocr_confidence=min(scores) if scores else None,
                manual_review="unreviewed", semantic_accuracy_verified=False,
                note="Flags are review priorities; they neither prove an error nor detect all missed icons or text.")


def source_stamp(target, image_hash):
    return dict(split=target["split"], page_id=target["page_id"], image_filename=target["image_filename"],
                region=target["region"], image_sha256=image_hash, config_hash=config_hash())


def asset_dir(target, output=OUTPUT):
    return output / "assets" / target["split"] / target["key"]


def load_record(target, stamp, output=OUTPUT):
    folder = asset_dir(target, output)
    path = folder / "complete.json"
    if not path.exists():
        return None
    record = read_json(path)
    if record["source"] != stamp:
        raise ValueError(f"Completed Table input/settings changed: {target['region']['id']}")
    for name, sha in record["artifact_sha256"].items():
        if not (folder / name).exists() or digest(folder / name) != sha:
            raise ValueError(f"Completed Table artifact is missing/changed: {folder / name}")
    inspect_html(record["table_content"])
    if (folder / "table.html").read_text(encoding="utf-8") != record["table_content"]:
        raise ValueError("HTML and recovery record differ")
    return record


def save_record(target, stamp, image, box, content, raw, seconds, provenance, output=OUTPUT):
    from io import BytesIO
    quality = inspect_html(content, raw)
    folder = asset_dir(target, output)
    if (folder / "complete.json").exists():
        raise FileExistsError("Refusing to overwrite a completed Table")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    atomic_bytes(folder / "image.png", image_bytes.getvalue())
    atomic_bytes(folder / "table.html", content.encode("utf-8"))
    atomic_json(folder / "raw.json", raw)
    if provenance.startswith("sample:"):
        quality.update(manual_review="reviewed_with_known_issues", semantic_accuracy_verified=False)
        quality["flags"].append("known_sample_ocr_errors")
    record = dict(source=stamp, table_content=content, crop_box=box, crop_size=list(image.size),
                  seconds=seconds, created_at=now(), provenance=provenance, quality=quality,
                  artifact_sha256={n: digest(folder / n) for n in ("image.png", "table.html", "raw.json")})
    # Commit marker is written last; incomplete files are not counted as success.
    atomic_json(folder / "complete.json", record)
    return record


def enrich_page(page, split, records):
    result = copy.deepcopy(page)
    for region in result["bounding_boxes"]:
        key = key_for(split, page["id"], region["id"])
        if key in records:
            record = records[key]
            if region["structure"] != "Table" or record["source"]["region"] != region:
                raise ValueError("Recovered Table does not match source region")
            region["table_content"] = record["table_content"]
    return result


def validate_page(before, after):
    if len(before["bounding_boxes"]) != len(after["bounding_boxes"]):
        raise ValueError("Region count changed")
    clean = copy.deepcopy(after)
    added = 0
    for old, new in zip(before["bounding_boxes"], clean["bounding_boxes"]):
        if "table_content" in new:
            if old["structure"] != "Table":
                raise ValueError("table_content was added to a non-Table region")
            inspect_html(new.pop("table_content"))
            added += 1
    if clean != before:
        raise ValueError("Page, description, ID, coordinates, text or QA changed")
    return added


def checkpoint(records, output=OUTPUT):
    counts = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        tmp = output / f"{split}.jsonl.tmp"
        saved = 0
        with (DATASET / "data/vlm" / f"{split}.jsonl").open(encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as dest:
            for line in source:
                before = json.loads(line)
                after = enrich_page(before, split, records)
                saved += validate_page(before, after)
                dest.write(json.dumps(after, ensure_ascii=False) + "\n")
            dest.flush()
            os.fsync(dest.fileno())
        os.replace(tmp, output / f"{split}.jsonl")
        counts[split] = saved
    return counts


def validate_existing(records, output=OUTPUT):
    """Accept an older checkpoint, but refuse to overwrite changed saved contents."""
    from itertools import zip_longest
    for split in SPLITS:
        path = output / f"{split}.jsonl"
        if not path.exists():
            continue
        with (DATASET / "data/vlm" / f"{split}.jsonl").open(encoding="utf-8") as source, path.open(encoding="utf-8") as saved:
            for a, b in zip_longest(source, saved):
                if a is None or b is None:
                    raise ValueError("Existing checkpoint page count differs")
                old, new = json.loads(a), json.loads(b)
                validate_page(old, new)
                for region in new["bounding_boxes"]:
                    if "table_content" in region:
                        key = key_for(split, old["id"], region["id"])
                        if key not in records or region["table_content"] != records[key]["table_content"]:
                            raise ValueError("Existing HTML differs from completed recovery records")


def validate_output(records, before_hashes, output=OUTPUT):
    from itertools import zip_longest
    stats = {}
    for split in SPLITS:
        count = Counter(pages=0, regions=0, qa=0, descriptions=0, tables=0, table_content=0)
        with (DATASET / "data/vlm" / f"{split}.jsonl").open(encoding="utf-8") as source, (output / f"{split}.jsonl").open(encoding="utf-8") as dest:
            for a, b in zip_longest(source, dest):
                if a is None or b is None:
                    raise ValueError("Page count changed")
                old, new = json.loads(a), json.loads(b)
                added = validate_page(old, new)
                if new != enrich_page(old, split, records):
                    raise ValueError("Final JSONL differs from committed table records")
                count.update(pages=1, regions=len(new["bounding_boxes"]), qa=len(new.get("qa_data", [])), table_content=added)
                for region in new["bounding_boxes"]:
                    count["tables"] += region["structure"] == "Table"
                    count["descriptions"] += bool(region.get("description"))
        stats[split] = dict(count)
    after_hashes = hashes()
    if after_hashes != before_hashes:
        raise ValueError("Protected source file hashes changed")
    return dict(validated_at=now(), counts=stats, protected_files_unchanged=True,
                protected_sha256=after_hashes, added_tables=sum(v["table_content"] for v in stats.values()),
                semantic_accuracy_verified=False)
