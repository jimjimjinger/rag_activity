import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import parent_context as parent
import answering


def region(rid, kind, text, x, y, width=300, height=20):
    return {"id": rid, "structure": kind, "ocr_info": [{"word": text}],
            "shape": {"x": x, "y": y, "width": width, "height": height}}


def chunk(reg, text=None, title="", context=None):
    return {"chunk_id": "chunk-" + reg["id"], "manual_id": "manual", "page_id": "page",
            "chunk_type": "table" if reg["structure"] == "Table" else "text",
            "text": text or parent.ocr_text(reg), "title": title,
            "content_region_ids": [reg["id"]], "context_region_ids": context or [],
            "source_shapes": [{"id": reg["id"], "shape": reg["shape"]}],
            "source_spans": [], "quality_flags": [], "image_filename": "page.jpg"}


class ParentContextTests(unittest.TestCase):
    def test_indented_table_links_to_nearest_subheading(self):
        regions = [region("h1", "Title", "Overview", 170, 260),
                   region("h2", "Title", "Package Contents", 175, 475, 190),
                   region("intro", "Text", "Items included", 175, 518, 640),
                   region("table", "Table", "", 237, 563, 452, 100)]
        chunks = [chunk(regions[2], "Package Contents\nItems included", "Package Contents", ["h2"]),
                  chunk(regions[3], "Overview\nCamera | USB cable", "Overview", ["h1"])]
        page = {"id": "page", "image_filename": "page.jpg", "bounding_boxes": regions,
                "qa_data": {"answer": "PRIVATE_QA_SENTINEL"}}
        chunks[0]["qa_data"] = "PRIVATE_QA_SENTINEL"
        result = parent.build_page(page, chunks)
        self.assertEqual(result["child_to_parents"]["chunk-table"], ["h2"])
        self.assertEqual(result["chunks"]["chunk-table"]["text"], "Package Contents\nCamera | USB cable")
        self.assertNotIn("PRIVATE_QA_SENTINEL", json.dumps(result))
        self.assertEqual(chunks[1]["text"], "Overview\nCamera | USB cable")

    def test_consecutive_procedure_connects_both_columns(self):
        regions = [region("heading", "Title", "Installation", 10, 10)]
        regions += [region(str(i), "Text", f"{i}. Step {i}", 10 if i < 4 else 400,
                           50 + (i-1) % 3 * 70, 150) for i in range(1, 7)]
        assigned, reasons, _, procedure = parent.section_links(regions)
        self.assertTrue(procedure)
        self.assertTrue(all(assigned[str(i)] == "heading" for i in range(1, 7)))
        self.assertEqual(reasons["6"], "consecutive_numbered_procedure")

    def test_next_heading_stops_section_and_unheaded_region_isolated(self):
        regions = [region("a", "Title", "Safety", 10, 0),
                   region("b", "Text", "Wear glasses", 10, 40),
                   region("c", "Title", "Maintenance", 10, 80),
                   region("d", "Text", "Clean device", 10, 120)]
        assigned, _, _, _ = parent.section_links(regions)
        self.assertEqual(assigned["b"], "a")
        self.assertEqual(assigned["d"], "c")
        self.assertEqual(parent.section_links(regions[1:2])[0]["b"], "b")

    def test_labels_inside_diagram_share_parent_across_columns(self):
        regions = [region("heading", "Title", "Front", 10, 0, 80),
                   region("diagram", "graphic", "", 10, 40, 600, 300)]
        regions += [region(str(i), "Text", f"Label {i}", 10 if i < 3 else 450,
                           60 + i % 3 * 50, 100) for i in range(6)]
        assigned, reasons, _, _ = parent.section_links(regions)
        self.assertEqual(assigned["diagram"], "heading")
        self.assertTrue(all(assigned[str(i)] == "heading" for i in range(6)))
        self.assertEqual(reasons["5"], "contained_object_label")

    def test_expansion_preserves_seeds_budget_and_rejects_wrong_manual(self):
        heading = region("h", "Title", "Setup", 10, 0)
        regions = [heading] + [region(str(i), "Text", "x" * 450 + str(i), 10, 40+i*30)
                               for i in range(25)]
        chunks = [chunk(r) for r in regions[1:]]
        page = {"id": "page", "image_filename": "page.jpg", "bounding_boxes": regions}
        mapped = parent.build_page(page, chunks)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pages").mkdir()
            key = parent.page_key("manual", "page")
            path = root / "pages" / (key + ".json")
            path.write_text(json.dumps(mapped), encoding="utf8")
            manifest = {"status": "complete", "version": parent.VERSION,
                        "module_sha256": parent.file_sha(parent.__file__),
                        "pages": {key: {"sha256": parent.file_sha(path)}}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
            store = parent.ParentContextStore(root)
            hits = copy.deepcopy(chunks[:5])
            sources, excluded, trace = store.prepare_context("설정 방법은?", hits, "manual")
            self.assertEqual([s["chunk_id"] for s in sources[:5]], [s["chunk_id"] for s in hits])
            self.assertGreater(trace["added_count"], 0)
            self.assertTrue(excluded)
            self.assertLessEqual(trace["prompt_token_upper_bound"] + answering.OPTIONS["num_predict"],
                                 answering.OPTIONS["num_ctx"])
            self.assertEqual(hits, chunks[:5])
            with self.assertRaises(ValueError):
                store.prepare_context("question", hits, "another-manual")


if __name__ == "__main__":
    unittest.main()
