"""PM209 character-based chunks. No QA fields are read by this module.

Offsets refer to normalized field text, never to HTML bytes or raw OCR JSON.
Geometry is a conservative heuristic, not a new document layout model.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import defaultdict
from lxml import html

VERSION = "pm209-char500-v1"
LIMIT = 500
OVERLAP = 100
FIGURES = {"Product Image", "illustration", "graphic"}


def normalize(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ocr_text(region):
    return normalize(" ".join(str(w.get("word", "")) for w in region.get("ocr_info", [])))


def manual_id(page, metadata):
    parts = page["image_filename"].replace("\\", "/").split("/")
    if len(parts) < 4 or parts[0] != "images" or parts[2] != "images" or parts[1] not in metadata:
        raise ValueError(f"Unrecognized image/manual mapping: {page['image_filename']}")
    return parts[1]


def sentence_spans(text):
    """Break only after punctuation + whitespace, not decimals or numbered steps."""
    starts = [0]
    for match in re.finditer(r"[.!?。！？][\"'’”)]*\s+", text):
        before = text[:match.start() + 1]
        if re.search(r"(?:\b(?:Mr|Mrs|Dr|No|Fig|vs|e\.g|i\.e)|\b\d+)\.$", before, re.I):
            continue
        starts.append(match.end())
    starts.append(len(text))
    return [(a, b) for a, b in zip(starts, starts[1:]) if text[a:b].strip()]


def split_text(text, prefix="", limit=LIMIT, overlap=OVERLAP):
    """Yield lossless normalized source ranges, always advancing new content.

    An unbreakable identifier or oversized context is kept with an explicit
    exception flag. The embedding stage rejects exceptions above its token cap.
    """
    text = text.strip()
    prefix = prefix.strip()
    if not text:
        return []
    head = prefix + "\n" if prefix else ""
    capacity = limit - len(head)
    if capacity < 40:
        return [{"text": head + text, "start": 0, "end": len(text), "overlap": 0,
                 "flags": ["oversized_context_review_required"]}]
    sentences = sentence_spans(text)
    ends = [b for a, b in sentences]
    pieces, cursor, next_start = [], 0, 0
    while cursor < len(text):
        start = next_start
        # Do not let overlap force a whole new sentence to be split.
        next_end = next((e for e in ends if e > cursor), len(text))
        if next_end - cursor <= capacity and next_end - start > capacity:
            start = cursor
        options = [e for e in ends if cursor < e <= start + capacity]
        long_sentence = not options
        flags = []
        if options:
            end = max(options)
        else:
            stop = min(len(text), start + capacity)
            clauses = list(re.finditer(r"[,;:，；]\s+", text[cursor:stop]))
            spaces = list(re.finditer(r"\s+", text[cursor:stop]))
            breaks = clauses if clauses and clauses[-1].end() >= (stop - cursor) // 2 else spaces
            end = cursor + breaks[-1].end() if breaks else stop
            if not breaks and stop < len(text) and not text[stop].isspace():
                # Never silently cut a part number or long uninterrupted token.
                following = re.search(r"\s", text[stop:])
                end = stop + following.start() if following else len(text)
                flags.append("unbreakable_token_length_exception")
        if end <= cursor:
            raise AssertionError("Chunker did not advance")
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        pieces.append({"text": head + text[a:b], "start": a, "end": b,
                       "overlap": max(0, cursor - a), "flags": flags})
        cursor = end
        next_start = cursor
        if cursor >= len(text) or overlap == 0:
            continue
        complete = [(a, b) for a, b in sentences if a >= start and b == end and b-a <= overlap]
        if complete:
            next_start = complete[-1][0]
        elif long_sentence:
            tail = max(start, end - overlap)
            boundary = re.search(r"\s+", text[tail:end])
            if boundary:
                next_start = tail + boundary.end()
    return pieces


def rect(region):
    s = region["shape"]
    return s["x"], s["y"], s["x"] + s["width"], s["y"] + s["height"]


def x_overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0]))


def inside_fraction(region, table):
    a, b = rect(region), rect(table)
    area = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    return x_overlap(a, b) * max(0, min(a[3], b[3])-max(a[1], b[1])) / area if area else 0


def table_units(markup):
    """Return row strings, cell references, header context and parser flags.

    rowspan/colspan are expanded using references to the same original cell.
    Unmarked first rows are NOT assumed to be true column headers.
    """
    flags = []
    try:
        parser = html.HTMLParser(no_network=True)
        root = html.fragment_fromstring(markup, create_parent="div", parser=parser)
        tables = root.xpath(".//table")
        if len(tables) != 1:
            raise ValueError("Expected exactly one non-nested table")
        table = tables[0]
        if any(e.level_name in {"ERROR", "FATAL"} for e in parser.error_log):
            flags.append("html_parser_repaired_review_required")
        if not re.search(r"</table\s*>", markup, re.I):
            flags.append("html_closing_tag_missing")
        rows = table.xpath("./tr|./thead/tr|./tbody/tr|./tfoot/tr")
        if not rows:
            raise ValueError("No table rows")
        grid, cells, header_rows = {}, [], []
        for ri, row in enumerate(rows):
            elements = row.xpath("./td|./th")
            if not elements:
                continue
            is_head = row.getparent().tag == "thead" or all(e.tag == "th" for e in elements)
            if is_head and (not header_rows and ri == 0 or header_rows and ri == header_rows[-1]+1):
                header_rows.append(ri)
            ci = 0
            for el in elements:
                while (ri, ci) in grid:
                    ci += 1
                rs, cs = int(el.get("rowspan", "1")), int(el.get("colspan", "1"))
                if rs <= 0 or cs <= 0 or rs > 1000 or cs > 1000 or ri+rs > len(rows):
                    raise ValueError("Invalid or out-of-range span")
                cell = {"row": ri, "column": ci, "rowspan": rs, "colspan": cs,
                        "text": normalize(el.text_content())}
                cells.append(cell)
                for rr in range(ri, ri+rs):
                    for cc in range(ci, ci+cs):
                        if (rr, cc) in grid:
                            raise ValueError("Overlapping spans")
                        grid[rr, cc] = cell
                ci += cs
        width = max(c for r, c in grid)+1
        if any((ri, ci) not in grid for ri in range(len(rows)) for ci in range(width)):
            raise ValueError("Ragged row/colspan grid; preserve literal rows without asserting column alignment")
        shared_headings = []
        for ri in header_rows:
            cell = grid.get((ri, 0))
            if cell and cell['colspan'] == width and cell['text']:
                shared_headings.append(cell['text'])
        labels = []
        for ci in range(width):
            values = []
            for ri in header_rows:
                cell = grid.get((ri, ci))
                if cell and cell["text"] and cell["text"] not in values and cell['text'] not in shared_headings:
                    values.append(cell["text"])
            labels.append(" > ".join(values) or f"Column {ci+1}")
        if not header_rows:
            flags.append("table_header_unmarked_review_required")
        units = []
        for ri in range(len(rows)):
            if ri in header_rows:
                continue
            refs, fields = [], []
            for ci in range(width):
                cell = grid.get((ri, ci))
                if cell is None:
                    flags.append("ragged_table_review_required")
                    fields.append(f"{labels[ci]}: [missing cell]")
                    continue
                refs.append({k: v for k, v in cell.items() if k != "text"})
                value = cell["text"] or "[empty cell]"
                fields.append(f"{labels[ci]}: {value}")
            units.append({"row": ri, "text": " | ".join(fields), "cells": refs,
                          "fields": fields})
        # Keep titles and footnotes outside the table, as well as <caption>.
        captions = table.xpath("./caption")
        notes = []
        for element in root.xpath("./*"):
            if element is not table and not element.xpath(".//table"):
                notes.append(normalize(element.text_content()))
        if root.text and root.text.strip():
            notes.insert(0, normalize(root.text))
        if table.tail and table.tail.strip():
            notes.append(normalize(table.tail))
        context = "\n".join(filter(None, [normalize(c.text_content()) for c in captions] + shared_headings + notes))
        if not header_rows and len(units) > 1:
            # Keep the first row as explicitly unclassified context, not guessed
            # column names. This also covers ordinary key/value tables safely.
            first = units[0]['text']
            if len(first) <= 180:
                context = '\n'.join(filter(None, [context, 'First row (header unconfirmed): ' + first]))
        # Header-only tables still carry information.
        if not units:
            units = [{"row": 0, "text": " | ".join(labels), "cells": cells, "fields": labels}]
        return {"units": units, "context": context, "cells": cells, "flags": sorted(set(flags))}
    except (ValueError, TypeError, html.etree.ParserError) as exc:
        # Preserve readable content and the source reference; never invent a grid.
        root = html.fragment_fromstring(markup or "<div></div>", create_parent='div')
        units = []
        for ri, row in enumerate(root.xpath('.//tr')):
            fields = [f'Cell {ci+1}: ' + (normalize(el.text_content()) or '[empty cell]')
                      for ci, el in enumerate(row.xpath('./td|./th'))]
            if fields:
                units.append({'row': ri, 'text': f'Literal row {ri+1}: ' + ' | '.join(fields),
                              'cells': [], 'fields': fields})
        if not units:
            text = normalize(' '.join(root.itertext()))
            units = [{"row": None, "text": text, "cells": [], "fields": [text]}]
        outside_rows = normalize(' '.join(root.xpath('.//text()[not(ancestor::tr)]')))
        context = '\n'.join(filter(None, ['Column/merge alignment unresolved; literal cells follow.', outside_rows]))
        return {"units": units, "context": context, "cells": [],
                "flags": ["table_structure_unresolved", str(exc)]}


def table_parts(markup, title=""):
    parsed = table_units(markup)
    context = "\n".join(filter(None, [title, parsed["context"]]))
    output, batch = [], []

    def flush():
        if batch:
            body = "\n".join(u["text"] for u in batch)
            output.append({"text": (context + "\n" if context else "") + body,
                           "table_rows": [u["row"] for u in batch],
                           "table_cells": [c for u in batch for c in u["cells"]],
                           "overlap": 0, "flags": parsed["flags"]})
            batch.clear()

    for unit in parsed["units"]:
        available = LIMIT - (len(context)+1 if context else 0)
        if len(unit["text"]) > available:
            flush()
            # Split a long row by cells; repeat the first cell as the row label.
            fields = unit["fields"]
            for fi, field in enumerate(fields):
                label, separator, value = field.partition(": ")
                row_context = fields[0] if fi > 0 else ""
                prefix = "\n".join(filter(None, [context, row_context, label if separator else ""]))
                for part in split_text(value if separator else field, prefix, overlap=0):
                    part.update(table_rows=[unit["row"]], table_cells=unit["cells"])
                    part['cell_span'] = {'row': unit['row'], 'column': fi,
                                         'start': part['start'], 'end': part['end'],
                                         'field': 'normalized_cell_text'}
                    part["flags"] = sorted(set(part["flags"] + parsed["flags"]))
                    output.append(part)
        else:
            candidate = sum(len(u["text"])+1 for u in batch) + len(unit["text"])
            if candidate > available:
                flush()
            batch.append(unit)
    flush()
    return output


def layout(regions):
    """Detect a clear two-column gutter; otherwise merge only aligned boxes."""
    if not regions:
        return {}, 1, 1
    boxes = [rect(r) for r in regions]
    left, right = min(b[0] for b in boxes), max(b[2] for b in boxes)
    width = max(1, right-left)
    height = max(1, max(b[3] for b in boxes)-min(b[1] for b in boxes))
    mid = (left+right)/2
    texts = [r for r in regions if r["structure"] == "Text"]
    ls = [rect(r) for r in texts if rect(r)[2] < mid + .02*width]
    rs = [rect(r) for r in texts if rect(r)[0] > mid - .02*width]
    parallel = sum(any(min(a[3], b[3]) > max(a[1], b[1]) for b in rs) for a in ls)
    two = len(ls) >= 3 and len(rs) >= 3 and parallel >= 2
    lanes = {}
    for r in regions:
        b = rect(r)
        lanes[r["id"]] = ("left" if b[2] <= mid+.02*width else "right" if b[0] >= mid-.02*width else "wide") if two else "single"
    return lanes, width, height


def chunk_page(page, split, metadata, manual_reviews=None):
    manual_reviews = manual_reviews or set()
    mid = manual_id(page, metadata)
    regions = page["bounding_boxes"]
    by_id = {r["id"]: r for r in regions}
    lanes, width, height = layout(regions)
    titles = [r for r in regions if r["structure"] == "Title" and ocr_text(r)]
    assigned, used_titles = {}, set()
    for region in regions:
        a = rect(region)
        candidates = []
        for title in titles:
            b = rect(title)
            if title["id"] == region["id"] or b[3] > a[1]+2:
                continue
            if lanes[title["id"]] not in {lanes[region["id"]], "wide"}:
                continue
            if abs(a[0]-b[0]) <= .04*width or b[0] <= (a[0]+a[2])/2 <= b[2]:
                candidates.append(title)
        if candidates:
            candidates.sort(key=lambda r: rect(r)[3], reverse=True)
            # Competing headings at one level are ambiguous.
            if len(candidates) < 2 or abs(rect(candidates[0])[3]-rect(candidates[1])[3]) > 3:
                assigned[region["id"]] = candidates[0]

    chunks = []

    def emit(kind, source_regions, title, part, origin, spans=None, flags=None):
        if not part["text"].strip():
            return
        ids = [r["id"] for r in source_regions]
        context_ids = [title["id"]] if title else []
        used_titles.update(context_ids)
        quality = sorted(set((flags or []) + part.get("flags", [])))
        if len(part["text"]) > LIMIT and not any("exception" in f or "oversized" in f for f in quality):
            raise AssertionError("Unflagged overlength chunk")
        record = dict(chunking_version=VERSION, manual_id=mid, split=split,
                      page_id=page["id"], chunk_type=kind,
                      content_region_ids=ids, context_region_ids=context_ids,
                      image_filename=page["image_filename"],
                      source_shapes=[{"id": rid, "shape": by_id[rid]["shape"]} for rid in dict.fromkeys(ids+context_ids)],
                      title=ocr_text(title) if title else "", text=part["text"],
                      char_count=len(part["text"]), overlap_char_count=part.get("overlap", 0),
                      content_origin=origin, quality_flags=quality,
                      source_spans=spans or ([part['cell_span']] if 'cell_span' in part else []), table_rows=part.get("table_rows", []),
                      table_cells=part.get("table_cells", []))
        signature = json.dumps(record, ensure_ascii=False, sort_keys=True)
        record["chunk_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, signature))
        record["text_sha256"] = hashlib.sha256(record["text"].encode()).hexdigest()
        chunks.append(record)

    table_chunks = defaultdict(list)
    for region in regions:
        kind = region["structure"]
        title = assigned.get(region["id"])
        if kind == "Table":
            if not region.get("table_content", "").strip():
                continue
            start = len(chunks)
            manual = region["id"] in manual_reviews
            for part in table_parts(region["table_content"], ocr_text(title) if title else ""):
                emit("table", [region], title, part, "manual_condensed" if manual else "paddle_table",
                     flags=["manual_condensed_review_required"] if manual else [])
            table_chunks[region["id"]] = chunks[start:]
        elif kind in FIGURES:
            description = normalize(region.get("description", ""))
            for part in split_text(description, ocr_text(title) if title else ""):
                emit("figure", [region], title, part, "qwen_figure_description",
                     [{"region_id": region["id"], "field": "description_normalized", "start": part["start"], "end": part["end"]}])

    # Exact text preservation + geometric containment, never geometry alone.
    deduped = set()
    for region in regions:
        if region["structure"] != "Text" or not ocr_text(region):
            continue
        for tid, parts in table_chunks.items():
            if tid in manual_reviews or inside_fraction(region, by_id[tid]) < .9:
                continue
            matches = [p for p in parts if ocr_text(region) in normalize(p["text"])]
            if matches:
                for p in matches:
                    # Aliases are separate from body IDs: a header repetition must
                    # not masquerade as direct evidence in region evaluation.
                    p.setdefault("deduplicated_region_ids", []).append(region["id"])
                    p["source_shapes"].append({"id": region["id"], "shape": region["shape"]})
                deduped.add(region["id"])
                break

    groups = []
    for lane in sorted(set(lanes.values())):
        ordered = sorted([r for r in regions if lanes[r["id"]] == lane], key=lambda r: (rect(r)[1], rect(r)[0]))
        current = []
        for region in ordered:
            if region["structure"] != "Text" or region["id"] in deduped or not ocr_text(region):
                if current:
                    groups.append(current); current = []
                continue
            if current:
                previous = current[-1]
                a, b = rect(previous), rect(region)
                same_title = assigned.get(previous["id"]) == assigned.get(region["id"])
                gap = b[1]-a[3]
                aligned = abs(a[0]-b[0]) <= .03*width and x_overlap(a, b) >= .65*min(a[2]-a[0], b[2]-b[0])
                wide_barrier = any(lanes[r["id"]] == "wide" and a[1] < rect(r)[1] <= b[1] for r in regions) if lane in {"left", "right"} else False
                warning = bool(re.match(r"(?:WARNING|CAUTION|DANGER|NOTE)\b", ocr_text(region), re.I))
                if not same_title or not aligned or not -3 <= gap <= max(32, .025*height) or wide_barrier or warning:
                    groups.append(current); current = []
            current.append(region)
        if current:
            groups.append(current)
    for group in groups:
        title = assigned.get(group[0]["id"])
        body, offsets = "", []
        for region in group:
            if body:
                body += "\n"
            start = len(body)
            body += ocr_text(region)
            offsets.append((start, len(body), region))
        for part in split_text(body, ocr_text(title) if title else ""):
            direct, spans = [], []
            for a, b, region in offsets:
                lo, hi = max(a, part["start"]), min(b, part["end"])
                if lo < hi:
                    direct.append(region)
                    spans.append({"region_id": region["id"], "field": "ocr_normalized", "start": lo-a, "end": hi-a})
            emit("text", direct, title, part, "source_ocr", spans,
                 ["layout_geometry_heuristic"])
    for title in titles:
        if title["id"] not in used_titles:
            for part in split_text(ocr_text(title), overlap=0):
                emit("title", [title], None, part, "source_ocr",
                     [{"region_id": title["id"], "field": "ocr_normalized", "start": part["start"], "end": part["end"]}])
    return chunks
