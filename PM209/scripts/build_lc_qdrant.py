"""Copy cached dense vectors and add BM25 through LangChain to a new Qdrant index."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import lc_qdrant_rag as app


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def batches(path, size=128):
    with path.open(encoding='utf-8') as file:
        group=[]
        for line in file:
            group.append(json.loads(line))
            if len(group)==size:
                yield group
                group=[]
        if group:yield group


def protected_files():
    original=app.legacy.read_json(app.SOURCE/'final-validation.json')['protected_files']
    for name,digest in original.items():
        if app.legacy.sha(Path(name))!=digest:raise ValueError('Original data hash changed: '+name)
    paths=[app.SOURCE/'chunks.jsonl',app.SOURCE/'embedding-cache.sqlite3',
           app.SOURCE/'index-identity.json',Path(app.answering.__file__)]
    paths += [p for p in (app.SOURCE/'qdrant').rglob('*') if p.is_file() and p.name!='.lock']
    for directory in ('qa30_20261102','qa30_hybrid_20261102','qa30_weights_20261102','qa30_top1_20261102','qa10_text_20261102'):
        paths += [p for p in (app.PM/'reports'/directory).rglob('*') if p.is_file()]
    return original | {str(p.resolve()):app.legacy.sha(p) for p in paths}


def run():
    started=time.perf_counter()
    app.OUTPUT.mkdir(parents=True,exist_ok=True)
    app.RUNTIME.mkdir(parents=True,exist_ok=True)
    state={'status':'running','pid':os.getpid(),'started_at':now(),'total':126938,'completed':0}
    def update(**values):
        state.update(values,updated_at=now(),elapsed_seconds=time.perf_counter()-started)
        app.write_json(app.RUNTIME/'status.json',state)
    embeddings=client=None
    try:
        update(stage='protecting_original_files')
        manifest=app.OUTPUT/'source-preservation.json'
        protected=protected_files()
        if manifest.exists():
            if app.legacy.read_json(manifest)!=protected:raise ValueError('Protected files changed since initial build')
        else:app.write_json(manifest,protected)
        identity={'source_chunk_sha256':app.legacy.sha(app.SOURCE/'chunks.jsonl'),
                  'dense_identity':app.legacy.read_json(app.SOURCE/'index-identity.json')['model_identity'],
                  'sparse_model':app.BM25_OPTIONS,'sparse_modifier':'idf','packages':app.versions(),
                  'collection':app.COLLECTION,'vectors':{'dense':1024,'bm25':'sparse'},
                  'payload':{'text':'original text','metadata':'all remaining original chunk fields'}}
        identity_file=app.OUTPUT/'index-identity.json'
        if identity_file.exists() and app.legacy.read_json(identity_file)!=identity:
            raise ValueError('Index identity changed; use a separate output')
        app.write_json(identity_file,identity)
        update(stage='opening_new_index')
        client=app.QdrantClient(path=str(app.OUTPUT/'qdrant'))
        if not client.collection_exists(app.COLLECTION):
            client.create_collection(app.COLLECTION,
                vectors_config={app.DENSE_NAME:app.models.VectorParams(size=1024,distance=app.models.Distance.COSINE)},
                sparse_vectors_config={app.SPARSE_NAME:app.models.SparseVectorParams(modifier=app.models.Modifier.IDF)})
        app.checked_collection(client)
        known=set();offset=None
        while True:
            points,offset=client.scroll(app.COLLECTION,offset=offset,limit=1024,with_payload=False,with_vectors=False)
            known.update(str(p.id) for p in points)
            if offset is None:break
        embeddings=app.ReusedQwenEmbeddings()
        sparse=app.make_sparse()
        store=app.make_store(client,embeddings,sparse)
        update(stage='indexing',completed=len(known),resumed=len(known))
        build_started=time.perf_counter();added=0
        for group in batches(app.SOURCE/'chunks.jsonl'):
            todo=[c for c in group if c['chunk_id'] not in known]
            if todo:
                docs=[app.Document(page_content=c['text'],metadata={k:v for k,v in c.items() if k!='text'}) for c in todo]
                store.add_documents(docs,ids=[c['chunk_id'] for c in todo],batch_size=128)
                added+=len(todo);known.update(c['chunk_id'] for c in todo)
                elapsed=time.perf_counter()-build_started
                update(completed=len(known),dense_vectors_reused=embeddings.document_reused,
                       eta_seconds=(126938-len(known))*elapsed/added)
                if added%2048==0:print(f'Indexed {len(known)}/126938; elapsed {elapsed:.1f}s',flush=True)
        if len(known)!=126938 or client.count(app.COLLECTION,exact=True).count!=126938:
            raise ValueError('Point count differs from 126938')
        update(stage='validating_every_point',validated=0,eta_seconds=None)
        validated=empty_sparse=0;max_difference=0.0
        for group in batches(app.SOURCE/'chunks.jsonl'):
            actual={str(p.id):p for p in client.retrieve(app.COLLECTION,ids=[c['chunk_id'] for c in group],with_vectors=True,with_payload=True)}
            expected_sparse=sparse.embed_documents([c['text'] for c in group])
            for c,sv in zip(group,expected_sparse):
                point=actual[c['chunk_id']]
                if {'text':point.payload['text'],**point.payload['metadata']}!=c:
                    raise ValueError('Payload changed: '+c['chunk_id'])
                dense=app.np.asarray(embeddings.cached(c['text']),dtype=app.np.float32)
                dense/=app.np.linalg.norm(dense)
                difference=float(app.np.max(app.np.abs(dense-app.np.asarray(point.vector[app.DENSE_NAME]))))
                max_difference=max(max_difference,difference)
                if difference>2e-6:raise ValueError('Dense vector changed: '+c['chunk_id'])
                found=point.vector[app.SPARSE_NAME]
                exp=dict(zip(sv.indices,sv.values));obs=dict(zip(found.indices,found.values))
                if set(exp)!=set(obs) or any(abs(exp[k]-obs[k])>2e-6 for k in exp):
                    raise ValueError('Sparse vector differs: '+c['chunk_id'])
                empty_sparse+=not bool(exp)
            validated+=len(group)
            update(validated=validated)
            if validated%8192==0:print(f'Validated {validated}/126938',flush=True)
        client.close();client=None
        update(stage='verifying_preservation')
        changed=[name for name,digest in protected.items() if app.legacy.sha(Path(name))!=digest]
        if changed:raise ValueError('Protected files changed: '+str(changed))
        report={'status':'complete','checked_at':now(),'points_verified':validated,'dimension':1024,
                'payload_exact_match':True,'cached_dense_vectors_preserved':True,'max_dense_abs_difference':max_difference,
                'bm25_vectors_recomputed_and_verified':True,'empty_sparse_vectors':empty_sparse,
                'source_and_previous_results_preserved':True,'protected_files':len(protected),
                'document_embeddings_generated':0,'errors':[], 'elapsed_seconds':time.perf_counter()-started,
                'semantic_accuracy_verified':False,'collection':app.COLLECTION,'packages':app.versions()}
        app.write_json(app.OUTPUT/'final-validation.json',report)
        app.write_json(app.RUNTIME/'final-validation.json',report)
        update(status='complete',stage='complete',completed=validated,eta_seconds=0)
        print(json.dumps(report,ensure_ascii=False),flush=True)
    except BaseException as error:
        update(status='failed',error=str(error))
        raise
    finally:
        if client is not None:client.close()
        if embeddings is not None:embeddings.close()


if __name__=='__main__':run()
