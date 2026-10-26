import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from repair_paddle_html import normalize, parse, signature


class RepairTests(unittest.TestCase):
    def test_optional_group_and_values_are_preserved(self):
        text = '<table><tr><td rowspan="2">10℃ & 20%</td><td>5</td></tr><tr><td>6</td></tr></tbody></table>'
        result, audit = normalize(text)
        self.assertIn('<tbody>', result)
        self.assertIn('10℃ &amp; 20%', result)
        self.assertTrue(audit['logical_positions_preserved'])
        self.assertEqual(signature(parse(text)[0]), signature(parse(result)[0]))

    def test_literal_ocr_code_does_not_swallow_cell_boundaries(self):
        text = '<table><tr><td><data id="a"</td><td>A2<B2 & <enabled/></td></tr></table>'
        result, audit = normalize(text)
        self.assertIn('&lt;data id="a"', result)
        self.assertIn('A2&lt;B2 &amp; &lt;enabled/&gt;', result)
        self.assertEqual(audit['cells'], 2)

    def test_inferred_header_and_mismatched_group_are_bounded(self):
        r, _ = normalize('<table><tr><th>Head</th></tr></thead><tbody><tr><td>2</td></tr></table>')
        self.assertIn('<thead><tr><th>Head</th></tr></thead><tbody>', r)
        r, _ = normalize('<table><thead><tr><td>X</td></tr></tbody></table>')
        self.assertIn('</thead></table>', r)

    def test_never_closes_missing_cells_or_infers_tail(self):
        for text in ('<table><tr><td>12</table>', '<table><tr><td>12</td></tr><tr></table>',
                     '<table><tr><td rowspan="2">12</td></tr></table>'):
            with self.assertRaises(ValueError): normalize(text)

    def test_repetitive_empty_rows_require_review(self):
        with self.assertRaisesRegex(ValueError, 'repeated_empty'):
            normalize('<table><tr><td>X</td></tr>' + '<tr><td></td></tr>' * 3 + '</table>')

    def test_row_group_spans_are_not_silently_flattened(self):
        with self.assertRaisesRegex(ValueError, 'crosses_row_group'):
            normalize('<table><thead><tr><td rowspan="2">X</td></tr></thead><tbody><tr><td>Y</td></tr></tbody></table>')

    def test_rows_fully_covered_by_rowspan_are_valid(self):
        text='<table><tr><td rowspan="4">X</td></tr><tr></tr><tr></tr><tr></tr></table>'
        result,audit=normalize(text)
        self.assertEqual(audit['rows'],4)
        self.assertEqual(audit['geometry']['trailing_empty_rows'],0)
        self.assertEqual(result.count('<tr></tr>'),3)


if __name__ == '__main__': unittest.main()
