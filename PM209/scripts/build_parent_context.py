"""Build a detached parent map for all existing PM209 chunks, without model calls."""
import argparse
from collections import Counter
import datetime as dt
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lc_qdrant_rag as app
import parent_context as parent


def chunk_offsets(path):
    offsets = {}
    count = 0
    previous = None
    with path.open("rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            chunk = json.loads(line)
            key = (chunk["manual_id"], chunk["page_id"])
            if key != previous:
                if key in offsets:
                    raise ValueError("Chunks are not contiguous by page")
                offsets[key] = [start, stream.tell()]
            else:
                offsets[key][1] = stream.tell()
            previous = key
            count += 1
    return offsets, count


def build(destination):
    started = time.perf_counter()
    source = app.SOURCE / "chunks.jsonl"
    paths = [source] + [app.legacy.DEFAULT_INPUT / (s + ".jsonl") for s in ("train", "val", "test")]
    identity = {str(p.resolve()): parent.file_sha(p) for p in paths}
    module_hash = parent.file_sha(parent.__file__)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        old = app.legacy.read_json(manifest_path)
        if old["input_sha256"] != identity or old["module_sha256"] != module_hash:
            raise ValueError("Parent map inputs/code changed; choose a new --output directory")
        if old["status"] == "complete":
            for key, entry in old["pages"].items():
                if parent.file_sha(destination / "pages" / (key + ".json")) != entry["sha256"]:
                    raise ValueError("Stored parent page changed")
            print("Reused verified parent map", flush=True)
            return old
    destination.mkdir(parents=True, exist_ok=True)
    state = {"status": "building", "version": parent.VERSION, "input_sha256": identity,
             "module_sha256": module_hash, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
             "rules": "Page sections, indented tables, single-headed consecutive procedures; original-region fallback",
             "qa_used": False, "embedding_calls": 0, "pages": {}}
    app.write_json(manifest_path, state)
    offsets, expected_chunks = chunk_offsets(source)
    counts = Counter()
    manuals = set()
    with source.open("rb") as chunk_file:
        for split in ("train", "val", "test"):
            for page in app.legacy.read_lines(app.legacy.DEFAULT_INPUT / (split + ".jsonl")):
                manual = page["image_filename"].replace("\\", "/").split("/")[1]
                bounds = offsets.pop((manual, page["id"]), None)
                if bounds is None:
                    raise ValueError("Source page missing from existing chunks")
                chunk_file.seek(bounds[0])
                chunks = [json.loads(line) for line in chunk_file.read(bounds[1]-bounds[0]).splitlines()]
                value = parent.build_page(page, chunks)
                key = parent.page_key(manual, page["id"])
                path = destination / "pages" / (key + ".json")
                app.write_json(path, value)
                state["pages"][key] = {"sha256": parent.file_sha(path), "manual_id": manual,
                                        "page_id": page["id"], "chunks": len(chunks)}
                counts.update(p["kind"] for p in value["parents"].values())
                counts["chunks"] += len(chunks)
                counts["title_corrections"] += len(value["title_corrections"])
                counts["pages"] += 1
                manuals.add(manual)
                if counts["pages"] % 1000 == 0:
                    print(f"Mapped {counts['pages']} pages / {counts['chunks']} chunks", flush=True)
    if offsets or counts["chunks"] != expected_chunks:
        raise ValueError("Parent map chunk coverage differs from the index")
    if any(parent.file_sha(p) != expected for p, expected in identity.items()):
        raise ValueError("An input changed during parent-map construction")
    state.update(status="complete", counts=dict(counts), manuals=len(manuals),
                 elapsed_seconds=time.perf_counter()-started)
    app.write_json(manifest_path, state)
    print(json.dumps({k: state[k] for k in ("status", "counts", "manuals", "elapsed_seconds")}), flush=True)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=parent.DEFAULT_STORE)
    build(parser.parse_args().output)
