import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from chunking import chunk_page, split_text, table_parts, table_units


def region(rid, text, x=10, y=100, width=200, height=20, kind='Text'):
    return {'id':rid,'structure':kind,'shape':dict(x=x,y=y,width=width,height=height),
            'ocr_info':[{'word':text}]}


def page(regions):
    return {'id':'m_1','image_filename':'images/m/images/m_1.jpg','bounding_boxes':regions,
            'qa_data':[{'question':{'text':'SECRET QUESTION'},'answer':{'text':'SECRET ANSWER'}}]}


class ChunkTests(unittest.TestCase):
    def test_long_text_preserves_every_nonspace_character(self):
        for text in [' '.join(f'Step {i}: Model X-12.5 uses 230 V. Turn it off safely.' for i in range(50)),
                     ' '.join('word'+str(i) for i in range(600)),
                     '짧은 문장입니다. '*150]:
            parts=split_text(text, 'Instructions')
            covered=set()
            last_end=-1
            for p in parts:
                self.assertLessEqual(len(p['text']),500)
                self.assertLessEqual(p['overlap'],100)
                self.assertGreater(p['end'],last_end)
                last_end=p['end']
                covered.update(range(p['start'],p['end']))
            self.assertTrue(all(i in covered or ch.isspace() for i,ch in enumerate(text)))

    def test_indivisible_identifier_flagged_not_cut(self):
        result=split_text('PART-'+ 'x'*600)
        self.assertEqual(len(result),1)
        self.assertIn('unbreakable_token_length_exception',result[0]['flags'])

    def test_qa_ignored_and_source_unchanged(self):
        source=page([region('t','Power',y=10,kind='Title'),region('b','Press power to start.')])
        before=copy.deepcopy(source)
        result=chunk_page(source,'train',{'m':{}})
        self.assertEqual(source,before)
        self.assertNotIn('SECRET',str(result))
        self.assertEqual(result,chunk_page(source,'train',{'m':{}}))
        self.assertEqual(result[0]['context_region_ids'],['t'])

    def test_different_titles_and_columns_do_not_merge(self):
        rs=[region('lt','LEFT',x=10,y=0,kind='Title'),region('rt','RIGHT',x=400,y=0,kind='Title')]
        for i in range(3):
            rs.extend([region('l'+str(i),'Left '+str(i),x=10,y=50+i*25),region('r'+str(i),'Right '+str(i),x=400,y=50+i*25)])
        result=chunk_page(page(rs),'train',{'m':{}})
        for c in result:
            self.assertFalse(any(i.startswith('l') for i in c['content_region_ids']) and any(i.startswith('r') for i in c['content_region_ids']))

    def test_table_spans_empty_cells_and_multilevel_header(self):
        raw='<table><thead><tr><th rowspan="2">Mode</th><th colspan="2">Power</th></tr><tr><th>Voltage</th><th>Current</th></tr></thead><tr><td rowspan="2">AC</td><td>230 V</td><td></td></tr><tr><td>110 V</td><td>2 A</td></tr></table>'
        parsed=table_units(raw)
        self.assertIn('Power > Voltage: 230 V',parsed['units'][0]['text'])
        self.assertIn('Mode: AC',parsed['units'][1]['text'])
        self.assertIn('[empty cell]',parsed['units'][0]['text'])
        self.assertEqual(parsed['units'][1]['cells'][0]['row'],2)

    def test_long_table_cells_preserve_tail(self):
        value=' '.join('Instruction '+str(i)+'.' for i in range(150))+' FINAL 42 V'
        raw='<table><tr><th>Mode</th><th>Steps</th></tr><tr><td>AC</td><td>'+value+'</td></tr></table>'
        parts=table_parts(raw,'Procedure')
        self.assertGreater(len(parts),2)
        self.assertTrue(all(len(p['text'])<=500 for p in parts))
        self.assertTrue(any('FINAL 42 V' in p['text'] for p in parts))
        self.assertTrue(all('Mode: AC' in p['text'] for p in parts[1:]))

    def test_table_dedup_only_exact_and_never_manual_summary(self):
        tab=region('tab','',kind='Table',y=90,height=100)
        tab['table_content']='<table><tr><th>Rating</th></tr><tr><td>230 V</td></tr></table>'
        src=page([tab,region('exact','230 V'),region('different','240 V',y=130)])
        result=chunk_page(src,'train',{'m':{}})
        self.assertTrue(any('exact' in c.get('deduplicated_region_ids',[]) for c in result))
        self.assertTrue(any('different' in c['content_region_ids'] for c in result))
        manual=chunk_page(src,'train',{'m':{}},{'tab'})
        self.assertTrue(any('exact' in c['content_region_ids'] for c in manual))
        self.assertTrue(any('manual_condensed_review_required' in c['quality_flags'] for c in manual))

    def test_malformed_grid_keeps_literal_values_and_footnote(self):
        raw='<table><caption>Ratings</caption><tr><td colspan="3">Heading</td></tr><tr><td>AC</td><td>230 V</td></tr></table><p>Only at 50 Hz.</p>'
        parts=table_parts(raw)
        self.assertTrue(all('table_structure_unresolved' in p['flags'] for p in parts))
        text=' '.join(p['text'] for p in parts)
        for value in ('Ratings','Heading','AC','230 V','50 Hz'):
            self.assertIn(value,text)

    def test_unmarked_first_row_not_claimed_as_confirmed_header(self):
        raw='<table><tr><td>Height</td><td>29.99 mm</td></tr><tr><td>Width</td><td>424 mm</td></tr></table>'
        parts=table_parts(raw)
        self.assertIn('header unconfirmed',parts[0]['text'])
        self.assertIn('Column 1: Width | Column 2: 424 mm',parts[0]['text'])


if __name__=='__main__':
    unittest.main()
