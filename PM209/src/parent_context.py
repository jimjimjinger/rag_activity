"""Link existing PM209 chunks to page sections and expand retrieved evidence.

The map reads document regions and existing chunks. It never reads QA fields or
creates embeddings. Search ranking remains owned by the existing retriever.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import answering
from chunking import FIGURES, inside_fraction, layout, normalize, ocr_text, rect, x_overlap

PM = Path(__file__).resolve().parents[1]
DEFAULT_STORE = PM / "datasets/PM209/rag/parent_context_v2"
VERSION = "pm209-parent-context-v2"
MAX_SOURCES = 32
STEP = re.compile(r"(?:^|\s)(\d{1,2})[.)]\s+")


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def page_key(manual, page_id):
    return hashlib.sha256((manual + "\0" + page_id).encode()).hexdigest()


def numbered_procedure(regions, titles):
    """Recognize a single headed, consecutively numbered page procedure."""
    if len(titles) != 1:
        return False
    numbers = []
    for region in regions:
        if region["structure"] == "Text":
            numbers.extend(int(m.group(1)) for m in STEP.finditer(ocr_text(region)))
    ordered = sorted(numbers)
    return (len(ordered) >= 3 and len(ordered) == len(set(ordered))
            and ordered == list(range(1, ordered[-1] + 1))
            and rect(titles[0])[3] <= min(rect(r)[1] for r in regions
                                        if r["structure"] == "Text"))


def section_links(regions):
    lanes, width, height = layout(regions)
    objects = [r for r in regions if r["structure"] in FIGURES | {"Table"}]
    containers = {}
    for region in regions:
        area = region["shape"]["width"] * region["shape"]["height"]
        enclosing = [obj for obj in objects if obj["id"] != region["id"]
                     and obj["shape"]["width"] * obj["shape"]["height"] > 1.05 * area
                     and inside_fraction(region, obj) >= .9]
        if enclosing:
            containers[region["id"]] = min(enclosing, key=lambda obj:
                obj["shape"]["width"] * obj["shape"]["height"])["id"]
    titles = [r for r in regions if r["structure"] == "Title" and ocr_text(r)
              and r["id"] not in containers]
    procedure = numbered_procedure(regions, titles)
    assigned = {}
    reasons = {}
    for region in regions:
        rid = region["id"]
        box = rect(region)
        if region in titles:
            assigned[rid] = rid
            reasons[rid] = "heading"
            continue
        if procedure and box[1] >= rect(titles[0])[3] - 2:
            assigned[rid] = titles[0]["id"]
            reasons[rid] = "consecutive_numbered_procedure"
            continue
        candidates = []
        for title in titles:
            heading = rect(title)
            if heading[3] > box[1] + 2:
                continue
            spans_columns = region["structure"] in FIGURES | {"Table"} and lanes[rid] == "wide"
            if not spans_columns and lanes[title["id"]] not in (lanes[rid], "wide"):
                continue
            # A short heading also governs indented tables beneath it.
            if heading[0] <= box[0] + .04 * width:
                candidates.append(title)
        candidates.sort(key=lambda r: rect(r)[3], reverse=True)
        if candidates and (len(candidates) == 1 or
                           rect(candidates[0])[3] - rect(candidates[1])[3] > 3):
            assigned[rid] = candidates[0]["id"]
            reasons[rid] = "nearest_heading_in_reading_column"
        else:
            assigned[rid] = rid
            reasons[rid] = "original_region_only"

    # On unheaded areas, associate unambiguous contained/nearby text with an object.
    for rid, container_id in containers.items():
        # Nested figures inherit the containing diagram/table's section.
        while container_id in containers:
            container_id = containers[container_id]
        assigned[rid] = assigned[container_id]
        reasons[rid] = "contained_object_label"
    for region in regions:
        rid = region["id"]
        if region["structure"] != "Text" or reasons[rid] != "original_region_only":
            continue
        a = rect(region)
        matches = []
        for obj in objects:
            b = rect(obj)
            horizontal = x_overlap(a, b) / max(1, min(a[2]-a[0], b[2]-b[0]))
            gap = max(0, b[1]-a[3], a[1]-b[3])
            if horizontal >= .7 and gap <= min(40, .025 * height):
                matches.append(obj)
        if len(matches) == 1:
            assigned[rid] = assigned[matches[0]["id"]]
            reasons[rid] = "unambiguous_object_caption"
    return assigned, reasons, lanes, procedure


def build_page(page, chunks):
    """Whitelist document content, with no copying of the page-level QA object."""
    if not chunks:
        raise ValueError("Cannot build a page without indexed chunks")
    manual = chunks[0]["manual_id"]
    regions = page["bounding_boxes"]
    by_id = {r["id"]: r for r in regions}
    assigned, reasons, lanes, procedure = section_links(regions)
    parents = {}
    children = {}
    projected = {}
    corrections = []
    for chunk in chunks:
        if chunk["page_id"] != page["id"] or chunk["manual_id"] != manual:
            raise ValueError("Cross-page/manual chunk in parent builder")
        memberships = list(dict.fromkeys(assigned[rid] for rid in chunk["content_region_ids"]))
        allowed = answering.SOURCE_FIELDS + ("chunking_version", "split", "title",
                  "char_count", "overlap_char_count", "text_sha256", "token_count",
                  "deduplicated_region_ids")
        child = {k: copy.deepcopy(chunk[k]) for k in allowed if k in chunk}
        if len(memberships) == 1:
            heading = by_id[memberships[0]]
            if heading["structure"] == "Title":
                new_title = ocr_text(heading)
                old_title = chunk.get("title", "")
                body = chunk["text"]
                if old_title and body.startswith(old_title + "\n"):
                    body = body[len(old_title) + 1:]
                if chunk["chunk_type"] != "title":
                    child["text"] = new_title + "\n" + body
                    child["title"] = new_title
                    child["context_region_ids"] = [heading["id"]]
                    kept = set(child["content_region_ids"] + [heading["id"]])
                    child["source_shapes"] = [{"id": rid, "shape": by_id[rid]["shape"]}
                                              for rid in by_id if rid in kept]
                    child["retrieval_text_sha256"] = chunk.get("text_sha256")
                    child["text_sha256"] = hashlib.sha256(child["text"].encode()).hexdigest()
                    child["char_count"] = len(child["text"])
                    if old_title != new_title:
                        corrections.append({"chunk_id": chunk["chunk_id"],
                                            "old": old_title, "new": new_title})
        projected[chunk["chunk_id"]] = child
        children[chunk["chunk_id"]] = memberships
        for parent_id in memberships:
            anchor = by_id[parent_id]
            parent = parents.setdefault(parent_id, {
                "parent_id": parent_id,
                "title": ocr_text(anchor) if anchor["structure"] == "Title" else "",
                "kind": "procedure" if procedure else
                        "section" if anchor["structure"] == "Title" else "original_region",
                "child_ids": [], "region_ids": [], "link_reasons": []})
            parent["child_ids"].append(chunk["chunk_id"])
            for rid in chunk["content_region_ids"]:
                if assigned[rid] == parent_id:
                    parent["region_ids"].append(rid)
                    parent["link_reasons"].append(reasons[rid])

    def reading_order(child_id):
        chunk = projected[child_id]
        content = [by_id[r] for r in chunk["content_region_ids"]]
        numbers = [int(m.group(1)) for r in content if r["structure"] == "Text"
                   for m in STEP.finditer(ocr_text(r))]
        lane = min({"left": 0, "single": 0, "wide": 0, "right": 1}[lanes[r["id"]]]
                   for r in content)
        y = min(rect(r)[1] for r in content)
        first_row = min(chunk.get("table_rows") or [0])
        first_char = min((s.get("start", 0) for s in chunk.get("source_spans", [])), default=0)
        return (min(numbers) if procedure and numbers else 100, lane, y, first_row, first_char)

    for parent in parents.values():
        parent["child_ids"].sort(key=reading_order)
        parent["region_ids"] = list(dict.fromkeys(parent["region_ids"]))
        parent["link_reasons"] = sorted(set(parent["link_reasons"]))
    return {"version": VERSION, "manual_id": manual, "page_id": page["id"],
            "image_filename": page["image_filename"], "parents": parents,
            "child_to_parents": children, "chunks": projected,
            "title_corrections": corrections}


class ParentContextStore:
    def __init__(self, directory=DEFAULT_STORE):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete" or self.manifest["version"] != VERSION:
            raise ValueError("Build and validate the parent context store first")
        if self.manifest["module_sha256"] != file_sha(__file__):
            raise ValueError("Parent context code changed; rebuild the map")

    @lru_cache(maxsize=64)
    def page(self, manual, page_id):
        key = page_key(manual, page_id)
        path = self.directory / "pages" / (key + ".json")
        expected = self.manifest["pages"].get(key)
        if expected is None or file_sha(path) != expected["sha256"]:
            raise ValueError("Missing or altered parent page: " + page_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value["manual_id"] != manual or value["page_id"] != page_id:
            raise ValueError("Parent page identity mismatch")
        return value

    def prepare_context(self, question, hits, manual):
        seeds = []
        pools = []
        seen_parents = set()
        trace = {"version": VERSION, "store": str(self.directory),
                 "seed_chunk_ids": [h["chunk_id"] for h in hits], "parents": [],
                 "title_corrections": [], "duplicates_removed": [], "added_chunk_ids": []}
        for hit in hits:
            if hit["manual_id"] != manual:
                raise ValueError("Cross-manual parent lookup")
            page = self.page(manual, hit["page_id"])
            child_id = hit["chunk_id"]
            if child_id not in page["chunks"]:
                raise ValueError("Retrieved chunk is absent from its parent map")
            seed = copy.deepcopy(page["chunks"][child_id])
            seed["score"] = hit.get("score")
            seeds.append(seed)
            for parent_id in page["child_to_parents"][child_id]:
                key = (hit["page_id"], parent_id)
                if key in seen_parents:
                    continue
                seen_parents.add(key)
                parent = page["parents"][parent_id]
                trace["parents"].append({k: parent[k] for k in
                                         ("parent_id", "title", "kind", "link_reasons")}
                                        | {"page_id": hit["page_id"]})
                pool = [page["chunks"][cid] for cid in parent["child_ids"]]
                # Complete prose, numbered steps and table rows before extra captions.
                pool.sort(key=lambda c: c["chunk_type"] in ("figure", "title"))
                pools.append(pool)
                trace["title_corrections"].extend(c for c in page["title_corrections"]
                                                   if c["chunk_id"] in parent["child_ids"])
        sources, excluded = answering.prepare_context(question, seeds, manual)
        selected = [{k: v for k, v in s.items() if k != "source_id"} for s in sources]
        seen = {h["chunk_id"] for h in hits}
        texts = {normalize(s["text"]) for s in selected}
        # Round robin prevents the first large parent from monopolizing the budget.
        for offset in range(max((len(p) for p in pools), default=0)):
            for pool in pools:
                if offset >= len(pool):
                    continue
                chunk = pool[offset]
                cid = chunk["chunk_id"]
                if cid in seen:
                    continue
                seen.add(cid)
                text = normalize(chunk["text"])
                if text in texts:
                    trace["duplicates_removed"].append(cid)
                    continue
                if len(selected) >= MAX_SOURCES:
                    excluded.append({"chunk_id": cid, "reason": "source_count_budget"})
                    continue
                candidate = copy.deepcopy(chunk)
                candidate["score"] = None
                accepted, rejected = answering.prepare_context(question, selected + [candidate], manual)
                if rejected:
                    excluded.extend(rejected)
                    continue
                selected.append(candidate)
                texts.add(text)
                trace["added_chunk_ids"].append(cid)
        sources, final_excluded = answering.prepare_context(question, selected, manual)
        excluded.extend(final_excluded)
        trace.update(seed_count=len(hits), source_count=len(sources),
                     added_count=len(trace["added_chunk_ids"]),
                     context_chars=sum(len(s["text"]) for s in sources),
                     prompt_token_upper_bound=answering.prompt_upper_bound(
                         answering.messages_for(question, sources, retry=True)))
        return sources, excluded, trace
