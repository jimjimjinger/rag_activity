"""Preservation/recovery regressions for the full Paddle table extraction writer."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import paddle_tables as app

HTML = '<table><tr><td rowspan="2">A</td><td>27°℃</td></tr><tr><td>-</td></tr></table>'
TEST_TMP = Path(__file__).resolve().parents[1] / ".test-tmp"
TEST_TMP.mkdir(exist_ok=True)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.page = dict(id="page", image_filename="images/a.jpg", qa_data=[{"question": "private", "answer": "keep"}],
                         bounding_boxes=[dict(id="figure", structure="illustration", shape={"x": 1}, description="Existing description"),
                                         dict(id="table", structure="Table", shape={"x": 2}),
                                         dict(id="text", structure="Text", text="original")])
        self.key = app.key_for("train", "page", "table")
        self.record = dict(source={"region": self.page["bounding_boxes"][1]}, table_content=HTML)

    def test_only_table_content_added_all_other_data_identical(self):
        old = copy.deepcopy(self.page)
        new = app.enrich_page(self.page, "train", {self.key: self.record})
        self.assertEqual(app.validate_page(self.page, new), 1)
        self.assertEqual(self.page, old)
        self.assertEqual(new["qa_data"], old["qa_data"])
        self.assertEqual(new["bounding_boxes"][0], old["bounding_boxes"][0])

    def test_reject_description_or_qa_change(self):
        for mutate in (lambda p: p["bounding_boxes"][0].update(description="changed"),
                       lambda p: p["qa_data"][0].update(answer="changed")):
            bad = copy.deepcopy(self.page)
            mutate(bad)
            with self.assertRaises(ValueError):
                app.validate_page(self.page, bad)

    def test_reject_wrong_region_identity(self):
        record = copy.deepcopy(self.record)
        record["source"]["region"]["shape"]["x"] = 3
        with self.assertRaises(ValueError):
            app.enrich_page(self.page, "train", {self.key: record})

    def test_html_allows_single_cell_and_flags_units_without_rewriting(self):
        self.assertEqual(app.inspect_html('<table><tr><td>1</td></tr></table>')["cells"], 1)
        result = app.inspect_html(HTML)
        self.assertIn("suspect_temperature_notation", result["flags"])
        self.assertFalse(result["semantic_accuracy_verified"])
        with self.assertRaises(ValueError):
            app.inspect_html('<table><tr><td>truncated')

    def test_resume_and_tamper_detection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            target = dict(split="train", page_id="page", image_filename="a.jpg", key=self.key,
                          region=self.page["bounding_boxes"][1])
            stamp = {"region": target["region"], "image_sha256": "example", "config_hash": "example"}
            output = Path(tmp)
            record = app.save_record(target, stamp, Image.new("RGB", (2, 2)), [0, 0, 2, 2], HTML, {}, 1, "test", output)
            self.assertEqual(app.load_record(target, stamp, output), record)
            with self.assertRaises(FileExistsError):
                app.save_record(target, stamp, Image.new("RGB", (2, 2)), [0, 0, 2, 2], HTML, {}, 1, "test", output)
            (app.asset_dir(target, output) / "table.html").write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                app.load_record(target, stamp, output)

    def test_checkpoint_recovery_keeps_every_page_and_rejects_modified_saved_html(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as tmp:
            dataset, output = Path(tmp) / "dataset", Path(tmp) / "output"
            (dataset / "data/vlm").mkdir(parents=True)
            for split in app.SPLITS:
                (dataset / "data/vlm" / f"{split}.jsonl").write_text(json.dumps(self.page) + "\n", encoding="utf-8")
            with patch.object(app, "DATASET", dataset):
                counts = app.checkpoint({self.key: self.record}, output)
                self.assertEqual(counts, {"train": 1, "val": 0, "test": 0})
                app.validate_existing({self.key: self.record}, output)
                page = app.read_json(output / "train.jsonl")
                page["bounding_boxes"][1]["table_content"] = '<table><tr><td>Different</td></tr></table>'
                (output / "train.jsonl").write_text(json.dumps(page) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    app.validate_existing({self.key: self.record}, output)


if __name__ == "__main__":
    unittest.main()
