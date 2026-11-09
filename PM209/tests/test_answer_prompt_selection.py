"""Verify CLI prompt selection reaches model requests and preserves frozen budgets."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

PM = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PM / "scripts"), str(PM / "src")]
import langchain_rag as cli
import answering


class PromptSelectionTests(unittest.TestCase):
    def setUp(self):
        self.previous = answering.PROMPT_VERSION, answering.SYSTEM_PROMPT

    def tearDown(self):
        answering.PROMPT_VERSION, answering.SYSTEM_PROMPT = self.previous

    def test_selection_reaches_model_and_generation_metadata(self):
        directory = PM / "reports/qa30_focused_prompt_20261108"
        for name, filename, version in (
                ("baseline", "prompt-before.txt", "pm209-grounded-ko-v1"),
                ("focused", "prompt-after.txt", "pm209-focused-ko-v2"),
                ("baseline", "prompt-before.txt", "pm209-grounded-ko-v1")):
            cli.select_answer_prompt(name)
            source = {"source_id": "S1", "chunk_type": "text", "text": "Voltage: 120 V.", "quality_flags": []}
            value = {"answer": "전압은 120 V입니다.", "source_ids": ["S1"], "insufficient_evidence": False}
            chat = cli.app.LocalChat.__new__(cli.app.LocalChat)
            chat.digest = cli.app.ANSWER_DIGEST
            chat.invoke = Mock(return_value={"done": True, "done_reason": "stop",
                "eval_count": 20, "prompt_eval_count": 200, "total_duration": 1, "load_duration": 0,
                "message": {"content": json.dumps(value)}})
            actual, metadata = chat.generate("전압은?", [source])
            self.assertEqual(actual, value)
            self.assertEqual(chat.invoke.call_args.args[0][0]["content"],
                             (directory / filename).read_text(encoding="utf-8"))
            self.assertEqual(metadata["prompt_version"], version)

    def test_both_prompts_fit_all_unchanged_parent_inputs(self):
        plans = json.loads((PM / "reports/qa30_parent_context_20261108_v2/frozen-input.json").read_text(encoding="utf-8"))
        for name in ("baseline", "focused"):
            cli.select_answer_prompt(name)
            for plan in plans:
                bound = answering.prompt_upper_bound(answering.messages_for(
                    plan["question"], plan["sources"], retry=True))
                self.assertLessEqual(bound + answering.OPTIONS["num_predict"], answering.OPTIONS["num_ctx"])


if __name__ == "__main__":
    unittest.main()
