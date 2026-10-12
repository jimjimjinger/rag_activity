"""Offline regression checks. No PDF parsing, model downloads or API calls."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from langchain_core.embeddings import Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import rag
from build_corpus import CORRECTIONS, PARSED, apply_corrections, split_page
from llm_client import PROMPT


class ChunkTests(unittest.TestCase):
    def test_feature_context_and_procedure_stay_together(self):
        text = (
            "### Guide\nMenu > Screen\nOverview.\n\n"
            "## 1 Search\n1. Enter a name.\n2. Click Search.\n\n"
            "## 2 Reset\nClick Clear filter.\n"
        )
        chunks = split_page(text)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all("Menu > Screen" in c["content"] for c in chunks))
        self.assertIn("1. Enter a name.\n2. Click Search.", chunks[0]["content"])
        self.assertNotIn("Click Clear filter.", chunks[0]["content"])

    def test_ambiguous_pages_and_tables_are_not_split(self):
        examples = [
            "## 1 Search\nSearch.\n## 2 Reset\nReset.\n3Download the results.",
            "## 1 Search\n\n## 2 Reset\nReset.",
            "## 1 Search\nSearch.\n## 3 Reset\nReset.",
            "## 1 Search\nSearch.\n## 2 Scores\n<table><tr><td>10</td></tr></table>",
            "## 1 Search\nSearch.\n## 2 Scores\n| Item | Score |\n|---|---|\n| A | 10 |",
            "## Feature Guide\n1. Select country.\n2. Select business.\n3. Run.",
        ]
        for text in examples:
            with self.subTest(text=text):
                chunks = split_page(text)
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0]["content"], text)

    def test_actual_spec_monitoring_sections(self):
        text = (PARSED / "page_20/page_20.md").read_text(encoding="utf-8").strip()
        edits = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["pages"]
        text = apply_corrections(text, 20, edits)
        chunks = split_page(text)
        self.assertEqual([c["section"] for c in chunks],
                         ["1 Business Unit and Country", "2 Issue Filter", "3 Page Details"])
        self.assertIn("Clear filter", chunks[1]["content"])
        self.assertTrue(all("MONITORING RESULTS > Spec Monitoring" in c["content"] for c in chunks))

    def test_every_original_line_is_preserved(self):
        edits = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["pages"]
        for path in sorted(PARSED.glob("page_*/page_*.md")):
            text = path.read_text(encoding="utf-8").strip()
            text = apply_corrections(text, int(path.stem.split("_")[1]), edits)
            combined = "\n".join(c["content"] for c in split_page(text))
            for line in text.splitlines():
                if line.strip():
                    self.assertIn(line.strip(), combined, str(path))

    def test_reviewed_pages_have_correct_boundaries_and_bodies(self):
        expected = {
            9: ["1 Question Input", "2 Category Selection", "3 Suggested Questions"],
            11: ["1 Page Type", "2 Country Results", "3 Summary Download"],
            15: ["3 Strategic Task", "4 Applied Area & Target Customer Channel"],
            20: ["1 Business Unit and Country", "2 Issue Filter", "3 Page Details"],
            22: ["1 Page Type", "2 Item Details"],
            23: ["1 DICT List and My Requests", "2 DICT Type", "3 Search", "4 Download"],
            24: ["5 Request Change", "6 Change, Delete"],
            31: ["1 DICT List and Change Log", "2 DICT Type", "3 Search", "4 Download"],
            32: ["5 Add item", "6 Edit & Delete"],
        }
        edits = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["pages"]
        for page, titles in expected.items():
            raw = (PARSED / f"page_{page:02d}/page_{page:02d}.md").read_text(encoding="utf-8").strip()
            chunks = split_page(apply_corrections(raw, page, edits))
            self.assertEqual([c["section"] for c in chunks], titles)
            for c in chunks:
                body = c["content"].split("## " + c["section"], 1)[1].strip()
                self.assertTrue(body, f"Page {page}: {c['section']}")
                self.assertNotIn("\n## ", body, f"Leaked next item: {page}")

    def test_stale_correction_is_rejected(self):
        edits = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["pages"]
        with self.assertRaisesRegex(ValueError, "Markdown changed"):
            apply_corrections("Changed OCR result", 9, edits)

    def test_unnumbered_heading_can_be_a_section(self):
        chunks = split_page("Menu\n## 1 Filter\nFilter results.\n## Page Details\nClick Detail.")
        self.assertEqual([c["section"] for c in chunks], ["1 Filter", "Page Details"])


class TestEmbedding(Embeddings):
    def embed_documents(self, texts):
        # Deliberately different magnitudes: raw inner product gives wrong order.
        values = {"near": [2.0, 0.0], "angled": [100.0, 100.0], "away": [-4.0, 0.0]}
        return [values[t] for t in texts]

    def embed_query(self, text):
        return [10.0, 0.0]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="rag-test-")
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        (root / "documents").mkdir()
        (root / "documents/manual.pdf").write_bytes(b"test PDF")
        self.corpus = root / "corpus.jsonl"
        source_hash = hashlib.sha256(b"test PDF").hexdigest()
        chunks = [
            {"id": f"p{i}-s1", "page": i, "section": name, "content": name,
             "source_pdf": "documents/manual.pdf", "source_sha256": source_hash}
            for i, name in enumerate(("near", "angled", "away"), 1)
        ]
        self.corpus.write_text("".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
        for name, value in {
            "ROOT": root, "CORPUS": self.corpus, "INDEX": root / "index.json",
            "VECTORS": root / "index.faiss",
        }.items():
            p = patch.object(rag, name, value)
            p.start()
            self.addCleanup(p.stop)
        for name, value in {"model_digest": "test-model", "embeddings": TestEmbedding()}.items():
            p = patch.object(rag, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)
        rag.build_index()

    def test_saved_index_cosine_ranking_and_source_mapping(self):
        hits = rag.retrieve("query", top_k=10)
        self.assertEqual([d.id for d, _ in hits], ["p1-s1", "p2-s1", "p3-s1"])
        np.testing.assert_allclose([s for _, s in hits], [1, 2 ** -0.5, -1], atol=1e-6)
        context, skipped = rag.build_context(hits)
        self.assertIn("[p.1]\nnear", context)
        self.assertEqual(skipped, [])

    def test_changed_corpus_is_rejected(self):
        self.corpus.write_text(self.corpus.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Corpus changed"):
            rag.load_retriever()

    def test_changed_source_is_rejected(self):
        (rag.ROOT / "documents/manual.pdf").write_bytes(b"new PDF")
        with self.assertRaisesRegex(ValueError, "Source PDF changed"):
            rag.load_retriever()

    def test_changed_model_is_rejected(self):
        with patch.object(rag, "model_digest", return_value="other-model"):
            with self.assertRaisesRegex(ValueError, "Embedding model changed"):
                rag.load_retriever()

    def test_mismatched_faiss_file_is_rejected(self):
        rag.VECTORS.write_bytes(b"incomplete index")
        with self.assertRaisesRegex(ValueError, "FAISS index and metadata differ"):
            rag.load_retriever()

    def test_old_index_is_rejected(self):
        rag.INDEX.write_text('{"chunks": []}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Old index format"):
            rag.read_manifest()


class PromptTests(unittest.TestCase):
    def test_evidence_and_question_remain_data(self):
        messages = PROMPT.format_messages(context="[p.20] {untrusted}", question="Reset?")
        self.assertIn("[p.20] {untrusted}", messages[1].content)
        self.assertIn("Reset?", messages[1].content)
        self.assertNotIn("{untrusted}", messages[0].content)


if __name__ == "__main__":
    unittest.main()
