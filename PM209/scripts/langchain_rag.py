"""Interactive CLI for the validated LangChain/Qdrant native hybrid index."""
import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import lc_qdrant_rag as app


def run(args):
    validation=app.legacy.read_json(app.OUTPUT/'final-validation.json')
    identity=app.legacy.read_json(app.OUTPUT/'index-identity.json')
    if validation['status']!='complete' or identity['packages']!=app.versions():
        raise ValueError('Build/validate the index with this pinned environment first')
    if identity['dense_identity']!=app.legacy.read_json(app.SOURCE/'index-identity.json')['model_identity']:
        raise ValueError('Stored embedding identities differ')
    if identity['sparse_model']!=app.BM25_OPTIONS or identity['collection']!=app.COLLECTION:
        raise ValueError('Stored BM25 configuration differs')
    embeddings=app.ReusedQwenEmbeddings(url=args.url)
    client=None
    try:
        client=app.QdrantClient(path=str(app.OUTPUT/'qdrant'))
        info=app.checked_collection(client)
        if info.points_count!=validation['points_verified']:raise ValueError('Index count changed')
        retriever=app.NativeRetriever(client,embeddings,app.make_sparse())
        # Reject an unknown manual instead of presenting an empty search as evidence.
        f=app.models.Filter(must=[app.models.FieldCondition(key='metadata.manual_id',match=app.models.MatchValue(value=args.manual))])
        if client.count(app.COLLECTION,count_filter=f,exact=True).count==0:
            raise ValueError('Unknown manual_id: '+args.manual)
        if args.stage=='answer':
            chain=app.rag_chain(retriever,app.LocalChat(args.url),args.keyword_percent,args.top_k,not args.no_translate)
            result=chain.invoke({'question':args.query,'manual':args.manual})
        else:
            app.answering.prepare_context(args.query,[],args.manual)
            translation=(app.LocalChat(args.url).translate(args.query)
                         if not args.no_translate and re.search('[가-힣]',args.query) else None)
            query=translation['english_query'] if translation else args.query
            hits=retriever.search(query,args.manual,args.keyword_percent,args.top_k)
            embeddings.unload()
            result={'question':args.query,'manual':args.manual,'search_query':query,'translation':translation,'hits':hits}
        result.update(keyword_percent=args.keyword_percent,top_k=args.top_k,rrf_k=app.RRF_K,
                      candidate_k=app.CANDIDATE_K,backend='LangChain + Qdrant native dense/BM25/weighted RRF',
                      query_embeddings_reused=embeddings.query_reused,query_embeddings_generated=embeddings.query_generated)
        destination=args.save or app.OUTPUT/'answers'/(dt.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]+'.json')
        app.write_json(destination,result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        print('Saved: '+str(destination.resolve()))
    finally:
        if client is not None:client.close()
        embeddings.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='LangChain → Qdrant dense/BM25 → local Korean answer')
    parser.add_argument('stage',choices=['search','answer'])
    parser.add_argument('--manual',required=True,help='One exact PM209 manual_id')
    parser.add_argument('--query',required=True)
    parser.add_argument('--keyword-percent',type=int,choices=range(101),metavar='0..100',default=50)
    parser.add_argument('--top-k',type=int,choices=range(1,21),metavar='1..20',default=5)
    parser.add_argument('--no-translate',action='store_true',help='Use original question as search query')
    parser.add_argument('--url',default='http://127.0.0.1:11434')
    parser.add_argument('--save',type=Path)
    try:run(parser.parse_args())
    except Exception as error:
        print('ERROR: '+str(error),file=sys.stderr)
        sys.exit(1)
