"""실패 그림을 뒤로 미루는 전체 처리의 종료 조건을 실제 GPU 없이 검사한다."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("run_full", Path(__file__).resolve().parents[1] / "scripts/run_full.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FullRunTests(unittest.TestCase):
    def test_failed_figure_is_retried_after_other_splits(self):
        counts = {"train": dict(figures=2, saved=0), "val": dict(figures=1, saved=0), "test": dict(figures=1, saved=0)}
        calls, passes = [], []

        def run(args):
            split = args.split[0]
            calls.append(split)
            if len(calls) == 1:
                counts[split]["saved"] = 1
                args.skip_region_ids.add("failed")
            else:
                counts[split]["saved"] = counts[split]["figures"]

        with patch.object(runner.app, "run", side_effect=run):
            result = runner.process_all(SimpleNamespace(), copy.deepcopy(counts), lambda: copy.deepcopy(counts),
                                        lambda: None, passes.append, lambda: False)
        self.assertEqual(calls, ["train", "val", "test", "train"])
        self.assertEqual(passes, [1, 2])
        self.assertTrue(all(row["figures"] == row["saved"] for row in result.values()))

    def test_permanent_failure_finishes_two_passes_without_claiming_completion(self):
        counts = {split: dict(figures=1 if split == "train" else 0, saved=0) for split in runner.SPLITS}

        def run(args):
            args.skip_region_ids.add("permanent_failure")

        with patch.object(runner.app, "run", side_effect=run) as mock:
            result = runner.process_all(SimpleNamespace(), copy.deepcopy(counts), lambda: copy.deepcopy(counts),
                                        lambda: None, lambda _: None, lambda: False)
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(result["train"]["saved"], 0)
        self.assertEqual(result["train"]["figures"], 1)


if __name__ == "__main__":
    unittest.main()
