"""실제 모델 없이 페이지 보존·중단·재개와 QA 입력 분리를 검증한다."""

import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("describe_images", ROOT / "src/describe_images.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class EnrichmentTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".test-tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name)
        (self.dataset / "data").mkdir()
        Image.new("RGB", (100, 80), "white").save(self.dataset / "page.jpg")
        self.pages = []
        for n, manual in enumerate(("manual_a", "manual_a", "manual_b")):
            self.pages.append({
                "id": f"{manual}_{n:05}", "image_filename": "page.jpg",
                "bounding_boxes": [
                    {"id": f"text_{n}", "structure": "Text", "shape": {},
                     "ocr_info": [{"word": "Original body", "bbox": {}}]},
                    *[{"id": f"figure_{n}_{i}", "structure": "illustration",
                       "shape": {"x": 0, "y": 0, "width": 40, "height": 30}}
                      for i in range(2)],
                    {"id": f"table_{n}", "structure": "Table", "shape": {}},
                ],
                "qa_data": [{"question": {"text": "EVALUATION_QUESTION"},
                             "answer": {"text": "EVALUATION_ANSWER", "relevant": [f"figure_{n}_0"]}}],
            })
        self.source = self.dataset / "data/train.jsonl"
        self.source.write_text("".join(json.dumps(p) + "\n" for p in self.pages), encoding="utf-8")
        self.original_bytes = self.source.read_bytes()
        self.output = self.dataset / "data/vlm"
        self.args = SimpleNamespace(dataset=self.dataset, manual=None, types=["illustration"], max_edge=64)

    def read_result(self):
        return [json.loads(line) for line in (self.output / "train.jsonl").read_text(encoding="utf-8").splitlines()]

    def assert_preserved(self, pages):
        self.assertEqual(len(pages), 3)
        cleaned = copy.deepcopy(pages)
        for page in cleaned:
            for region in page["bounding_boxes"]:
                region.pop("description", None)
        self.assertEqual(cleaned, self.pages)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)

    def test_limit_and_resume_preserve_all_pages_regions_and_qa(self):
        self.assertEqual(app.enrich_split(self.args, "train", self.output, lambda _: "First description", 1), 1)
        first = self.read_result()
        self.assert_preserved(first)
        self.assertEqual(first[0]["bounding_boxes"][1]["description"], "First description")
        self.assertNotIn("description", first[0]["bounding_boxes"][2])
        self.assertEqual(app.enrich_split(self.args, "train", self.output, lambda _: "Next description", 1), 1)
        second = self.read_result()
        self.assert_preserved(second)
        self.assertEqual(second[0]["bounding_boxes"][1]["description"], "First description")
        self.assertEqual(second[0]["bounding_boxes"][2]["description"], "Next description")

    def test_manual_filter_keeps_other_manuals(self):
        self.args.manual = "manual_b"
        app.enrich_split(self.args, "train", self.output, lambda _: "Selected manual", 1)
        result = self.read_result()
        self.assert_preserved(result)
        self.assertEqual(result[:2], self.pages[:2])
        self.assertEqual(result[2]["bounding_boxes"][1]["description"], "Selected manual")

    def test_progress_stop_checkpoints_completed_description(self):
        events = []

        def progress(event, split, region_id):
            events.append((event, split, region_id))
            if event == "generated":
                raise KeyboardInterrupt("clean stop")

        self.args.on_progress = progress
        with self.assertRaises(KeyboardInterrupt):
            app.enrich_split(self.args, "train", self.output, lambda _: "Saved before stop", 100)
        result = self.read_result()
        self.assert_preserved(result)
        self.assertEqual(result[0]["bounding_boxes"][1]["description"], "Saved before stop")
        self.assertEqual(events, [("generating", "train", "figure_0_0"),
                                  ("generated", "train", "figure_0_0")])

    def test_region_filter_keeps_other_regions_and_pages(self):
        self.args.region_ids = ["figure_2_1"]
        count = app.enrich_split(self.args, "train", self.output, lambda _: "Selected figure", 3)
        result = self.read_result()
        self.assert_preserved(result)
        self.assertEqual(count, 1)
        self.assertEqual(result[:2], self.pages[:2])
        self.assertNotIn("description", result[2]["bounding_boxes"][1])
        self.assertEqual(result[2]["bounding_boxes"][2]["description"], "Selected figure")

    def test_failure_or_ctrl_c_saves_completed_descriptions_and_remaining_pages(self):
        for failure in (RuntimeError("generation failed"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                output = self.output / type(failure).__name__
                calls = 0

                def generate(_):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise failure
                    return "Completed before interruption"

                with self.assertRaises(type(failure)):
                    app.enrich_split(self.args, "train", output, generate, None)
                result = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assert_preserved(result)
                self.assertEqual(result[0]["bounding_boxes"][1]["description"], "Completed before interruption")
                self.assertEqual(calls, 2)

    def test_modified_qa_or_missing_page_rejected_without_overwriting_previous(self):
        app.enrich_split(self.args, "train", self.output, lambda _: "Kept", 1)
        good = self.read_result()
        for bad in (good[:-1], copy.deepcopy(good)):
            if len(bad) == 3:
                bad[-1]["qa_data"] = []
            output_file = self.output / "train.jsonl"
            output_file.write_text("".join(json.dumps(p) + "\n" for p in bad), encoding="utf-8")
            previous = output_file.read_bytes()
            with self.assertRaises(ValueError):
                app.enrich_split(self.args, "train", self.output, lambda _: "Should not commit", 0)
            self.assertEqual(output_file.read_bytes(), previous)
            self.assertFalse(output_file.with_suffix(".jsonl.tmp").exists())

    def test_only_nonempty_description_field_is_accepted(self):
        self.assertEqual(app.validate_description('{"description":"  Visible + and - labels.  "}'), "Visible + and - labels.")
        for value in ('{"description":" "}', '{"description":"ok","visible_text":[]}', '[]', 'not JSON'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.validate_description(value)

    def test_truncated_response_retries_shorter_without_saving_partial_text(self):
        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(content='{"description":"unfinished', response_metadata={"done_reason": "length", "eval_count": 768}),
            SimpleNamespace(content='{"description":"A washer control panel. The Start button is visible."}', response_metadata={"done_reason": "stop"}),
        ]
        errors = []
        result = app.generate_description(model, b"image", errors.append)
        self.assertEqual(result, "A washer control panel. The Start button is visible.")
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(errors[0]["output_tokens"], 768)
        first, second = [call.args[0][0].content[0]["text"] for call in model.invoke.call_args_list]
        self.assertIn("2 to 4 short English sentences", first)
        self.assertIn("at most 60 words", second)
        self.assertNotIn("unfinished", second)

    def test_invalid_or_overlong_generated_response_has_bounded_retries(self):
        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(content="not JSON", response_metadata={}),
            SimpleNamespace(content=json.dumps({"description": "word " * 61}), response_metadata={}),
            SimpleNamespace(content='{"description":"still incomplete', response_metadata={"done_reason": "length"}),
        ]
        with self.assertRaises(app.DescriptionGenerationError) as failure:
            app.generate_description(model, b"image")
        self.assertEqual(model.invoke.call_count, 3)
        self.assertEqual(len(failure.exception.attempts), 3)

    def test_server_failure_is_not_treated_as_bad_figure(self):
        model = Mock()
        model.invoke.side_effect = RuntimeError("server unavailable")
        with self.assertRaisesRegex(RuntimeError, "server unavailable"):
            app.generate_description(model, b"image")
        self.assertEqual(model.invoke.call_count, 1)

    def test_ollama_repeat_error_retries_with_shorter_prompt(self):
        from ollama import ResponseError

        model = Mock()
        model.invoke.side_effect = [
            ResponseError("prediction aborted, token repeat limit reached"),
            SimpleNamespace(content='{"description":"An MP3 player screen. The menu shows playback controls."}',
                            response_metadata={"done_reason": "stop"}),
        ]
        errors = []
        description = app.generate_description(model, b"image", errors.append)
        self.assertEqual(description, "An MP3 player screen. The menu shows playback controls.")
        self.assertEqual(model.invoke.call_count, 2)
        self.assertEqual(errors[0]["error_type"], "ollama_token_repeat")
        self.assertIsNone(errors[0]["response_preview"])
        retry_prompt = model.invoke.call_args_list[1].args[0][0].content[0]["text"]
        self.assertIn("at most 60 words", retry_prompt)

    def test_repeated_ollama_repeat_errors_are_bounded_and_deferable(self):
        from ollama import ResponseError

        model = Mock()
        model.invoke.side_effect = ResponseError("prediction aborted, token repeat limit reached")
        with self.assertRaises(app.DescriptionGenerationError) as failure:
            app.generate_description(model, b"image")
        self.assertEqual(model.invoke.call_count, 3)
        self.assertEqual(len(failure.exception.attempts), 3)
        self.assertTrue(all(row["error_type"] == "ollama_token_repeat" for row in failure.exception.attempts))

    def test_other_ollama_errors_are_not_silently_deferred(self):
        from ollama import ResponseError

        for message, code in (("CUDA out of memory", 500), ("model not found", 404),
                              ("prediction aborted", -1)):
            with self.subTest(message=message):
                model = Mock()
                model.invoke.side_effect = ResponseError(message, code)
                with self.assertRaises(ResponseError):
                    app.generate_description(model, b"image")
                self.assertEqual(model.invoke.call_count, 1)

    def test_failed_figure_is_recorded_others_saved_and_missing_figure_can_resume(self):
        failures = []
        self.args.on_figure_failure = lambda split, region_id, error: failures.append((split, region_id))
        self.args.skip_region_ids = set()
        calls = 0

        def generate(_):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise app.DescriptionGenerationError([{"reason": "length"}] * 3)
            return "A valid description. A visible feature."

        count = app.enrich_split(self.args, "train", self.output, generate, None)
        self.assertEqual(count, 5)
        result = self.read_result()
        self.assert_preserved(result)
        self.assertNotIn("description", result[0]["bounding_boxes"][1])
        self.assertIn("description", result[2]["bounding_boxes"][2])
        self.assertEqual(failures, [("train", "figure_0_0")])
        self.args.skip_region_ids = set()
        self.assertEqual(app.enrich_split(self.args, "train", self.output, lambda _: "Recovered", None), 1)
        self.assertEqual(self.read_result()[0]["bounding_boxes"][1]["description"], "Recovered")

    def test_model_message_contains_only_fixed_prompt_and_image(self):
        messages = app.build_messages(b"image pixels")
        self.assertEqual(len(messages), 1)
        content = messages[0].content
        self.assertEqual(content[0], {"type": "text", "text": app.PROMPT})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotIn("EVALUATION_", json.dumps(content))

    def test_cannot_write_enrichment_over_source_data(self):
        self.args.output = self.dataset / "data"
        with self.assertRaises(ValueError):
            app.run(self.args)
        self.assertEqual(self.source.read_bytes(), self.original_bytes)


if __name__ == "__main__":
    unittest.main()
