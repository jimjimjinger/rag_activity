import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import hybrid_retrieval as app


class HybridTests(unittest.TestCase):
    def test_bm25_manual_filter_phrase_identifier_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'index.sqlite3'
            chunks=[{'chunk_id':'a','manual_id':'one','text':'10-1 means your signal is bad.'},
                    {'chunk_id':'b','manual_id':'two','text':'10-1 10-1 10-1 means disconnect.'},
                    {'chunk_id':'c','manual_id':'one','text':'10-2 means your signal is good.'}]
            index=app.BM25Index(p,'abc',chunks)
            self.assertEqual([x['chunk_id'] for x in index.search('one','What does 10-1 mean?')],['a','c'])
            self.assertEqual(index.search('one','10-1')[0]['chunk_id'],'a')
            self.assertEqual(index.search('missing','10-1'),[])
            self.assertEqual(index.search('one','the and'),[])
            self.assertEqual(index.search('one','zzzz OR zzzzzz'),[])
            index.close()
            resumed=app.BM25Index(p,'abc');self.assertEqual(resumed.count,3);resumed.close()
            with self.assertRaises(ValueError): app.BM25Index(p,'other')

    def test_identifier_tokens_and_fts_operator_are_data(self):
        self.assertEqual(app.search_terms('How can I remove SamsungPortableSSD.exe using 10-1?'),
                         ['remove','samsungportablessd.exe','using','10-1'])
        self.assertNotIn('or',app.search_terms('" OR NOT * NEAR(10-1)'))

    def test_rrf_consensus_and_deduplication(self):
        dense=[{'chunk_id':'a','score':0.9},{'chunk_id':'b','score':0.8}]
        keyword=[{'chunk_id':'b','bm25_score':-50},{'chunk_id':'c','bm25_score':-40}]
        fused=app.fuse_rrf(dense,keyword)
        self.assertEqual([r['chunk_id'] for r in fused],['b','a','c'])
        self.assertAlmostEqual(fused[0]['score'],0.5/62+0.5/61)
        self.assertEqual([r['chunk_id'] for r in app.fuse_rrf(dense,keyword,1,0)],['a','b'])
        with self.assertRaises(ValueError):app.fuse_rrf(dense+dense,keyword)

    def test_translation_rejects_missing_identifiers_and_truncation(self):
        q='SamsungPortableSSD.exe로 설치한 소프트웨어는 어떻게 삭제하나요?'
        def raw(t,reason='stop'):
            return {'done':True,'done_reason':reason,'message':{'content':json.dumps({'english_query':t})}}
        good='How do I uninstall software installed using SamsungPortableSSD.exe?'
        self.assertEqual(app.validate_translation(raw(good),q),good)
        with self.assertRaises(ValueError):app.validate_translation(raw('How do I delete software?'),q)
        with self.assertRaises(ValueError):app.validate_translation(raw(good,'length'),q)
        with self.assertRaises(ValueError):app.validate_translation(raw('SamsungPortableSSD.exe 삭제?'),q)
        with self.assertRaises(ValueError):app.validate_translation(raw('What does 110-12 mean?'),'10-1은 무슨 뜻인가요?')

    def test_translation_input_is_question_only(self):
        messages=app.translation_messages('10-1은 무슨 뜻인가요?')
        value=json.loads(messages[1]['content'])
        self.assertEqual(set(value),{'question','protected_terms'})
        self.assertEqual(value['protected_terms'],['10-1'])


if __name__=='__main__':unittest.main()
