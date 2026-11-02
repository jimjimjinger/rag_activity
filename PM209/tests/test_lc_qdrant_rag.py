import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import lc_qdrant_rag as app


class FixedDense(app.Embeddings):
    def embed_documents(self,texts):return [self.embed_query(t) for t in texts]
    def embed_query(self,text):return [1.0,0.0]+[0.0]*1022


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sparse=app.make_sparse()

    def setUp(self):
        self.db=app.QdrantClient(':memory:')
        self.db.create_collection(app.COLLECTION,
            vectors_config={'dense':app.models.VectorParams(size=1024,distance=app.models.Distance.COSINE)},
            sparse_vectors_config={'bm25':app.models.SparseVectorParams(modifier=app.models.Modifier.IDF)})
        self.dense=FixedDense()
        self.original=[]
        for i in range(27):
            text=('battery voltage '+str(i)) if i<25 else 'unrelated text'
            self.original.append({'text':text,'chunk_id':str(uuid.uuid4()),'manual_id':'A' if i<24 else 'B',
                                  'chunk_type':'text','quality_flags':[],'content_region_ids':['r'+str(i)]})
        store=app.make_store(self.db,self.dense,self.sparse)
        store.add_documents([app.Document(page_content=c['text'],metadata={k:v for k,v in c.items() if k!='text'}) for c in self.original],
                            ids=[c['chunk_id'] for c in self.original])
        self.retriever=app.NativeRetriever(self.db,self.dense,self.sparse)

    def tearDown(self):self.db.close()

    def test_real_native_fusion_and_both_prefetch_filters(self):
        app.checked_collection(self.db)
        with patch.object(self.db,'query_points',wraps=self.db.query_points) as spy:
            one=self.retriever.search('battery voltage','A',25,1)
            args=spy.call_args.kwargs
            self.assertIsInstance(args['query'],app.models.RrfQuery)
            self.assertEqual(args['query'].rrf.weights,[.75,.25])
            self.assertEqual(args['query'].rrf.k,60)
            self.assertEqual([p.limit for p in args['prefetch']],[20,20])
            self.assertTrue(all(p.filter.must[0].key=='metadata.manual_id' and p.filter.must[0].match.value=='A' for p in args['prefetch']))
            direct=self.db.query_points(**args).points
        five=self.retriever.search('battery voltage','A',25,5)
        self.assertEqual(one,five[:1])
        self.assertEqual([h['chunk_id'] for h in five],[str(p.id) for p in direct[:5]])
        self.assertEqual([h['score'] for h in five],[p.score for p in direct[:5]])
        self.assertTrue(all(h['manual_id']=='A' for h in five))
        for c in self.original:
            payload=self.db.retrieve(app.COLLECTION,[c['chunk_id']])[0].payload
            self.assertEqual({'text':payload['text'],**payload['metadata']},c)

    def test_endpoints_are_pure_and_empty_keyword_has_no_evidence(self):
        for weight,name in ((0,'dense'),(100,'bm25')):
            with patch.object(self.db,'query_points',wraps=self.db.query_points) as spy:
                hits=self.retriever.search('battery voltage','A',weight,5)
            self.assertEqual(spy.call_args.kwargs['using'],name)
            self.assertNotIn('prefetch',spy.call_args.kwargs)
            self.assertEqual(len(hits),5)
        self.assertEqual(self.retriever.search('!!!','A',100,5),[])


class ModelContractTests(unittest.TestCase):
    def test_transient_windows_replace_lock_retries_without_losing_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'status.json'
            app.write_json(path,{'completed':1})
            original=Path.replace
            calls=[]
            def replace(source,target):
                calls.append(1)
                if len(calls)<3:raise PermissionError('Windows reader lock')
                return original(source,target)
            with patch.object(Path,'replace',replace),patch.object(app.time,'sleep'):
                app.write_json(path,{'completed':2})
            self.assertEqual(json.loads(path.read_text()),{'completed':2})
            self.assertEqual(len(calls),3)
            self.assertEqual(list(Path(directory).glob('*.tmp')),[])

    def test_strict_embedding_query_instruction_and_no_truncation(self):
        model=app.StrictOllamaEmbeddings(model='qwen3-embedding:0.6b',keep_alive=120)
        with patch.object(model._client,'embed',return_value=Mock(embeddings=[[1.0]+[0.0]*1023])) as mock:
            model.embed_query('battery')
        self.assertEqual(mock.call_args.kwargs['input'],['Instruct: '+app.legacy.INSTRUCTION+'\nQuery: battery'])
        self.assertIs(mock.call_args.kwargs['truncate'],False)
        self.assertEqual(mock.call_args.kwargs['options'],app.legacy.EMBED_OPTIONS)

    def test_old_cache_is_read_only_and_missing_document_never_generates(self):
        embeddings=app.ReusedQwenEmbeddings()
        try:
            with self.assertRaises(sqlite3.OperationalError):
                embeddings.db.execute('CREATE TABLE must_not_write (id INTEGER)')
            with patch.object(app.StrictOllamaEmbeddings,'embed_documents') as generate:
                with self.assertRaises(ValueError):embeddings.embed_documents(['missing-doc-'+uuid.uuid4().hex])
                generate.assert_not_called()
        finally:embeddings.close()

    def test_chain_keeps_qa_out_and_preserves_citation_provenance(self):
        hit={'manual_id':'A','chunk_id':'c','chunk_type':'text','text':'Voltage 120 V.',
             'qa':'SECRET_ANSWER','answer':'SECRET_ANSWER','quality_flags':[],'page_id':'p'}
        original=copy.deepcopy(hit)
        retriever=Mock()
        retriever.search.return_value=[hit]
        chat=Mock()
        chat.translate.return_value={'english_query':'What voltage?'}
        chat.generate.return_value=({'answer':'120 V입니다.','source_ids':['S1'],'insufficient_evidence':False},{'called':True})
        result=app.rag_chain(retriever,chat).invoke({'question':'전압은?','manual':'A'})
        self.assertNotIn('SECRET',json.dumps(chat.generate.call_args.args,ensure_ascii=False))
        self.assertEqual(result['citations'][0]['chunk_id'],'c')
        self.assertEqual(hit,original)
        retriever.search.assert_called_once_with('What voltage?','A',50,5)


if __name__=='__main__':unittest.main()
