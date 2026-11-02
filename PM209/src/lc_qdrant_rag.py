"""LangChain orchestration with dense + BM25 retrieval and weighted RRF in Qdrant.

Legacy SQLite embeddings are opened read-only for reuse. No SQLite keyword index
or application-side rank fusion is used in this module.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import uuid

# Keep document/query content local even if a shell enables LangSmith tracing.
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

import httpx
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

PM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PM/'scripts'))
import rag_pipeline as legacy
import answering
from hybrid_retrieval import (TRANSLATION_OPTIONS, translation_messages,
                              validate_translation)

REPO = PM.parent
SOURCE = legacy.FULL_OUTPUT
OUTPUT = PM/'datasets/PM209/rag/full_char500_lc_qdrant_v1'
RUNTIME = REPO/'.runtime/pm209-lc-qdrant'
COLLECTION = 'pm209_hybrid_v1'
DENSE_NAME, SPARSE_NAME = 'dense', 'bm25'
RRF_K, CANDIDATE_K = 60, 20
ANSWER_DIGEST = 'd8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04'
BM25_OPTIONS = {'model_name':'Qdrant/bm25','k':1.2,'b':0.75,'avg_len':256.0,
                'language':'english','token_max_length':40,'disable_stemmer':False}
PACKAGES = ['langchain-core','langchain-ollama','langchain-qdrant','qdrant-client','fastembed','ollama']


def versions():
    return {name:importlib.metadata.version(name) for name in PACKAGES}


def write_json(path,value):
    """Atomic JSON replacement with bounded retries for Windows reader/AV locks."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        for attempt in range(40):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt==39:raise
                time.sleep(0.05)
    finally:
        temporary.unlink(missing_ok=True)


def checked_vector(value):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(1024,) or not np.isfinite(a).all() or np.linalg.norm(a)<=0:
        raise ValueError('Expected a finite nonzero 1024-dimensional Qwen embedding')
    return a.tolist()


class StrictOllamaEmbeddings(OllamaEmbeddings):
    """Preserve the existing Qwen instruction, num_batch and truncate=false policy."""
    def embed_documents(self, texts):
        # The stock wrapper omits truncate and num_batch in this pinned version.
        response=self._client.embed(model=self.model,input=texts,truncate=False,
                                    options=legacy.EMBED_OPTIONS,keep_alive='2m')
        if len(response.embeddings)!=len(texts):
            raise ValueError('Embedding count mismatch')
        return [checked_vector(v) for v in response.embeddings]

    def embed_query(self, text):
        return self.embed_documents([f'Instruct: {legacy.INSTRUCTION}\nQuery: {text}'])[0]


class ReusedQwenEmbeddings(Embeddings):
    """LangChain adapter: reuse saved document vectors, cache new query vectors as JSON."""
    def __init__(self, source=SOURCE, cache=RUNTIME/'query-cache', url='http://127.0.0.1:11434'):
        self.identity=legacy.read_json(source/'index-identity.json')['model_identity']
        self.cache=Path(cache)
        self.url=url
        self.model=StrictOllamaEmbeddings(model=self.identity['model'],base_url=url,
                    num_ctx=2048,num_thread=4,num_gpu=99,keep_alive=120,
                    client_kwargs={'timeout':180,'trust_env':False})
        self.db=sqlite3.connect((source/'embedding-cache.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        self.document_reused=self.query_reused=self.query_generated=0
        self.verified=False

    def key(self, text, query=False):
        return hashlib.sha256(json.dumps([self.identity,'query' if query else 'document',text],
                                         sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def cached(self, text, query=False):
        row=self.db.execute('SELECT dimension,vector FROM vectors WHERE key=?',(self.key(text,query),)).fetchone()
        if row:
            if row[0]!=1024:raise ValueError('Legacy cache dimension changed')
            return checked_vector(np.frombuffer(row[1],dtype=np.float32))
        if query:
            path=self.cache/(self.key(text,True)+'.json')
            if path.exists():
                data=legacy.read_json(path)
                if data['identity']!=self.identity or data['text']!=text:
                    raise ValueError('Query cache identity mismatch')
                return checked_vector(data['vector'])
        return None

    def verify_model(self):
        if self.verified:return
        with httpx.Client(base_url=self.url,timeout=15,trust_env=False) as client:
            response=client.get('/api/tags');response.raise_for_status()
            installed=next((m for m in response.json()['models'] if m['name']==self.identity['model']),None)
            if not installed or installed['digest']!=self.identity['digest']:
                raise ValueError('Qwen embedding model digest differs from stored document vectors')
        if self.identity!={'model':legacy.MODEL,'digest':installed['digest'],'options':legacy.EMBED_OPTIONS,
                           'query_instruction':legacy.INSTRUCTION,'truncate':False}:
            raise ValueError('Embedding configuration is not the frozen document configuration')
        self.verified=True

    def embed_documents(self, texts):
        vectors=[self.cached(t) for t in texts]
        if any(v is None for v in vectors):
            raise ValueError('A document embedding is missing; refusing to silently regenerate documents')
        self.document_reused+=len(texts)
        return vectors

    def embed_query(self, text):
        self.verify_model()
        vector=self.cached(text,True)
        if vector is not None:
            self.query_reused+=1
            return vector
        vector=self.model.embed_query(text)
        write_json(self.cache/(self.key(text,True)+'.json'),
                          {'identity':self.identity,'text':text,'vector':vector})
        self.query_generated+=1
        return vector

    def unload(self):
        with httpx.Client(base_url=self.url,timeout=30,trust_env=False) as client:
            response=client.post('/api/generate',json={'model':self.identity['model'],'keep_alive':0})
            response.raise_for_status()

    def close(self):
        self.db.close()


def make_sparse():
    return FastEmbedSparse(**BM25_OPTIONS,cache_dir=str(REPO/'.runtime/fastembed'),
                           batch_size=128,threads=2,parallel=None,providers=['CPUExecutionProvider'])


class CanonicalQdrantVectorStore(QdrantVectorStore):
    @classmethod
    def _document_from_point(cls, point, collection_name, content_payload_key, metadata_payload_key):
        # Avoid mutating a Qdrant local payload when adding LangChain's provenance.
        metadata=copy.deepcopy(point.payload.get(metadata_payload_key) or {})
        metadata.update(_id=str(point.id),_collection_name=collection_name)
        return Document(page_content=point.payload[content_payload_key],metadata=metadata)


def make_store(client, embeddings, sparse, mode=RetrievalMode.HYBRID):
    return CanonicalQdrantVectorStore(client=client,collection_name=COLLECTION,
        embedding=embeddings,sparse_embedding=sparse,retrieval_mode=mode,
        vector_name=DENSE_NAME,sparse_vector_name=SPARSE_NAME,
        content_payload_key='text',metadata_payload_key='metadata',
        validate_collection_config=False)


def checked_collection(client):
    info=client.get_collection(COLLECTION)
    params=info.config.params
    if (not isinstance(params.vectors,dict) or set(params.vectors)!={DENSE_NAME} or
        params.vectors[DENSE_NAME].size!=1024 or params.vectors[DENSE_NAME].distance!=models.Distance.COSINE or
        set(params.sparse_vectors or {})!={SPARSE_NAME} or
        params.sparse_vectors[SPARSE_NAME].modifier!=models.Modifier.IDF):
        raise ValueError('Qdrant dense/BM25 collection configuration differs')
    return info


class NativeRetriever:
    def __init__(self, client, embeddings, sparse):
        self.client,self.embeddings=client,embeddings
        self.stores={0:make_store(client,embeddings,sparse,RetrievalMode.DENSE),
                     100:make_store(client,embeddings,sparse,RetrievalMode.SPARSE),
                     'hybrid':make_store(client,embeddings,sparse)}

    def search(self, query, manual, keyword_percent=50, top_k=5):
        if not 0<=keyword_percent<=100 or not 1<=top_k<=CANDIDATE_K:
            raise ValueError('Use keyword 0..100 and top_k 1..20')
        f=models.Filter(must=[models.FieldCondition(key='metadata.manual_id',match=models.MatchValue(value=manual))])
        kwargs={}
        mode=keyword_percent if keyword_percent in (0,100) else 'hybrid'
        if mode=='hybrid':
            w=keyword_percent/100
            kwargs['hybrid_fusion']=models.RrfQuery(rrf=models.Rrf(k=RRF_K,weights=[1-w,w]))
        # LangChain forwards this RrfQuery and both filtered prefetches to Qdrant.
        # k=20 fixes candidate depth for both Top-1 and Top-5 comparisons.
        pairs=self.stores[mode].similarity_search_with_score(query,k=CANDIDATE_K,filter=f,**kwargs)
        hits=[]
        for doc,score in pairs[:top_k]:
            metadata={k:v for k,v in doc.metadata.items() if k not in ('_id','_collection_name')}
            if metadata.get('manual_id')!=manual:raise ValueError('Cross-manual retrieval')
            hits.append({**metadata,'text':doc.page_content,'score':score})
        return hits


class LocalChat:
    def __init__(self,url='http://127.0.0.1:11434'):
        self.url=url
        self.digest=ANSWER_DIGEST
        with httpx.Client(base_url=url,timeout=15,trust_env=False) as client:
            response=client.get('/api/tags');response.raise_for_status()
            installed=next((m for m in response.json()['models'] if m['name']==answering.MODEL),None)
            if not installed or installed['digest']!=self.digest:raise ValueError('Answer model changed')
        self.model=ChatOllama(model=answering.MODEL,base_url=url,reasoning=False,
            keep_alive='1m',temperature=0,disable_streaming=True,
            client_kwargs={'timeout':300,'trust_env':False})

    def invoke(self,messages,options,schema):
        prompt=ChatPromptTemplate.from_messages([
            ('system',messages[0]['content']),('human','{payload}')])
        chain=prompt | self.model.bind(options=options,format=schema,stream=False)
        result=chain.invoke({'payload':messages[1]['content']})
        metadata=result.response_metadata
        return {'done':metadata.get('done'),'done_reason':metadata.get('done_reason'),
                'message':{'content':result.content},**{k:metadata.get(k) for k in
                ('eval_count','prompt_eval_count','total_duration','load_duration')}}

    def translate(self,question):
        started=time.perf_counter();attempts=[]
        schema={'type':'object','properties':{'english_query':{'type':'string'}},
                'required':['english_query'],'additionalProperties':False}
        for n in range(2):
            raw=self.invoke(translation_messages(question,retry=bool(n)),TRANSLATION_OPTIONS,schema)
            record={k:raw[k] for k in ('done_reason','eval_count','prompt_eval_count')}
            record.update(attempt=n+1)
            try:query=validate_translation(raw,question)
            except (ValueError,TypeError) as e:
                record.update(error=str(e),rejected_content=raw['message']['content']);attempts.append(record);continue
            attempts.append(record)
            return {'english_query':query,'attempts':attempts,'elapsed_seconds':time.perf_counter()-started,
                    'backend':'LangChain ChatOllama'}
        raise ValueError('Translation validation failed: '+json.dumps(attempts,ensure_ascii=False))

    def generate(self,question,sources):
        if not sources:
            return {'answer':answering.ABSTENTION,'source_ids':[],'insufficient_evidence':True}, {'called':False,'attempts':[]}
        started=time.perf_counter();attempts=[]
        for n in range(2):
            messages=answering.messages_for(question,sources,retry=bool(n))
            bound=answering.prompt_upper_bound(messages)
            if bound+answering.OPTIONS['num_predict']>answering.OPTIONS['num_ctx']:
                raise ValueError('Answer context budget exceeded')
            raw=self.invoke(messages,answering.OPTIONS,answering.response_schema(sources))
            record={k:raw[k] for k in ('done_reason','eval_count','prompt_eval_count','total_duration','load_duration')}
            record.update(attempt=n+1,prompt_token_upper_bound=bound)
            try:value=answering.validate_response(raw,sources)
            except (ValueError,TypeError) as e:
                record.update(error=str(e),rejected_content=raw['message']['content']);attempts.append(record);continue
            attempts.append(record)
            return value,{'called':True,'model':answering.MODEL,'digest':self.digest,'options':answering.OPTIONS,
                'think':False,'timeout_seconds':300,'prompt_version':answering.PROMPT_VERSION,
                'attempts':attempts,'elapsed_seconds':time.perf_counter()-started,
                'backend':'LangChain ChatPromptTemplate | ChatOllama'}
        raise answering.AnswerGenerationError(attempts)


def rag_chain(retriever,chat,keyword_percent=50,top_k=5,translate=True):
    def question_stage(state):
        answering.prepare_context(state['question'],[],state['manual'])
        if translate and re.search('[가-힣]',state['question']):
            translation=chat.translate(state['question'])
            return {**state,'search_query':translation['english_query'],'translation':translation}
        return {**state,'search_query':state['question'],'translation':None}
    def retrieve_stage(state):
        hits=retriever.search(state['search_query'],state['manual'],keyword_percent,top_k)
        retriever.embeddings.unload()
        sources,excluded=answering.prepare_context(state['question'],hits,state['manual'])
        return {**state,'sources':sources,'excluded_sources':excluded,
                'retrieved_chunk_ids':[h['chunk_id'] for h in hits]}
    def answer_stage(state):
        value,generation=chat.generate(state['question'],state['sources'])
        citations,figures=answering.resolve_citations(value,state['sources'])
        return {**state,**value,'generation':generation,'citations':citations,'cited_figures':figures,
                'status':'insufficient_evidence' if value['insufficient_evidence'] else 'answered'}
    return (RunnableLambda(question_stage,name='translate_question') |
            RunnableLambda(retrieve_stage,name='qdrant_native_retrieval') |
            RunnableLambda(answer_stage,name='grounded_korean_answer'))
