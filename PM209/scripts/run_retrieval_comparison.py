"""Freeze query-only translations, then compare English dense with English hybrid.

Never reads QA references. Review/metrics are a separate post-generation step.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import time

import httpx
import rag_pipeline as app
import answering
import hybrid_retrieval as hybrid

VARIANTS=('B_english_dense', 'C_english_hybrid')


def now():
    return datetime.now(timezone.utc).isoformat()


def unload(url, model):
    with httpx.Client(base_url=url, timeout=30, trust_env=False) as client:
        response=client.post('/api/generate', json={'model':model, 'keep_alive':0})
        response.raise_for_status()


def run(args):
    questions=app.read_json(args.questions)
    if not isinstance(questions,list) or not 1<=len(questions)<=100:
        raise ValueError('Supply a fixed question-only list of 1..100 cases')
    for q in questions:
        if set(q)!={'case_id','manual_id','question'} or not all(isinstance(v,str) and v.strip() for v in q.values()):
            raise ValueError('Only case_id, manual_id, question may enter the experiment')
        if not re.fullmatch(r'[A-Za-z0-9_-]+',q['case_id']):
            raise ValueError('Invalid case ID')
        answering.prepare_context(q['question'],[],q['manual_id'])
    if len({q['case_id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate case IDs')
    baseline=app.read_json(args.baseline/'config.json')
    if app.sha(args.questions)!=baseline['questions_sha256']:
        raise ValueError('Questions differ from the frozen baseline')
    if app.sha(answering.__file__)!=baseline['answering_code_sha256']:
        raise ValueError('Answer code differs from the frozen baseline')
    index_identity=app.read_json(args.output/'index-identity.json')
    if index_identity!=baseline['index_identity']:
        raise ValueError('Index differs from baseline')
    with httpx.Client(base_url=args.url,timeout=10,trust_env=False) as client:
        response=client.get('/api/tags');response.raise_for_status()
        digests={m['name']:m['digest'] for m in response.json()['models']}
    baseline_digests={r['generation']['digest'] for r in app.read_json(args.baseline/'results.json')
                      if r.get('generation',{}).get('called')}
    if baseline_digests!={digests[args.answer_model]}:
        raise ValueError('Answer model weights differ from the baseline')
    config={'questions_sha256':app.sha(args.questions), 'baseline_results_sha256':app.sha(args.baseline/'results.json'),
        'index_identity':index_identity, 'answering_code_sha256':app.sha(answering.__file__),
        'hybrid_code_sha256':app.sha(hybrid.__file__), 'runner_code_sha256':app.sha(__file__),
        'translation_prompt_sha256':hashlib.sha256(hybrid.TRANSLATION_PROMPT.encode()).hexdigest(),
        'translation_options':hybrid.TRANSLATION_OPTIONS, 'answer_options':answering.OPTIONS,
        'answer_prompt_version':answering.PROMPT_VERSION,
        'answer_prompt_sha256':hashlib.sha256(answering.SYSTEM_PROMPT.encode()).hexdigest(),
        'answer_model':args.answer_model, 'translation_model':args.answer_model,
        'model_digests':{m:digests[m] for m in [args.model,args.answer_model]},
        'candidate_k':20,'final_k':5,'dense_weight':0.5,'keyword_weight':0.5,'rrf_k':60,
        'bm25_tokenizer':hybrid.TOKENIZER,'bm25_statistics_scope':'all_chunks',
        'tie_break':'dense_rank_then_keyword_rank_then_chunk_id',
        'manual_filter':'same as baseline', 'reference_input':False}
    dest=args.results;dest.mkdir(parents=True,exist_ok=True)
    config_path=dest/'config.json'
    if config_path.exists() and app.read_json(config_path)!=config:
        raise ValueError('Experiment configuration changed; use a new results directory')
    app.write_json(config_path,config)
    start=time.perf_counter()
    status={'status':'running','stage':args.stage,'started_at':now(),'total_questions':len(questions)}
    def update(**values):
        status.update(values,updated_at=now(),elapsed_seconds=time.perf_counter()-start)
        app.write_json(dest/'status.json',status)
    update()
    try:
        if args.stage=='translate':
            translations=[]
            for i,q in enumerate(questions,1):
                path=dest/'translations'/(q['case_id']+'.json')
                if path.exists():
                    record=app.read_json(path)
                    if any(record[k]!=q[k] for k in q):
                        raise ValueError('Saved translation does not match question')
                else:
                    record=dict(q,**hybrid.translate_question(q['question'],args.url,args.answer_model))
                    app.write_json(path,record)
                translations.append(record)
                update(translated=i)
                print(f"Translated {i}/{len(questions)} {q['case_id']}: {record['english_query']}",flush=True)
            app.write_json(dest/'translations.json',translations)
            unload(args.url,args.answer_model)
            update(status='translations_ready',translation_seconds=sum(r['elapsed_seconds'] for r in translations))
            return

        translation_review=app.read_json(dest/'translation-review.json')
        if (translation_review.get('status')!='accepted' or
                translation_review.get('translations_sha256')!=app.sha(dest/'translations.json')):
            raise ValueError('Translations must be checked and frozen before comparison')
        translations=app.read_json(dest/'translations.json')
        if len(translations)!=len(questions) or any(any(q[k]!=t[k] for k in q) for q,t in zip(questions,translations)):
            raise ValueError('Translation question identity mismatch')
        frozen={'translations_sha256':app.sha(dest/'translations.json'),
                'translation_review_sha256':app.sha(dest/'translation-review.json')}
        frozen_path=dest/'frozen-input.json'
        if frozen_path.exists() and app.read_json(frozen_path)!=frozen:
            raise ValueError('Translation changed after retrieval; use a new result directory')
        app.write_json(frozen_path,frozen)
        retrieval_path=dest/'retrieval.json'
        if retrieval_path.exists():
            retrieval=app.read_json(retrieval_path)
            if [r['case_id'] for r in retrieval]!=[q['case_id'] for q in questions]:
                raise ValueError('Saved retrieval case IDs differ')
        else:
            update(stage='bm25_index')
            if app.sha(args.output/'chunks.jsonl')!=index_identity['chunks_sha256']:
                raise ValueError('Chunk hash mismatch')
            build_start=time.perf_counter()
            bm25=hybrid.BM25Index(args.bm25_index,index_identity['chunks_sha256'],app.read_lines(args.output/'chunks.jsonl'))
            app.write_json(dest/'bm25-index.json',dict(bm25.identity,count=bm25.count,
                           path=str(args.bm25_index.resolve()),open_or_build_seconds=time.perf_counter()-build_start))
            update(stage='load_dense_index',bm25_chunks=bm25.count)
            manifest,chunks,model,db=app.open_search(args)
            try:
                if bm25.count!=len(chunks):
                    raise ValueError('BM25 and dense chunk counts differ')
                by_id={c['chunk_id']:c for c in chunks}
                manuals={c['manual_id'] for c in chunks}
                if any(q['manual_id'] not in manuals for q in questions):
                    raise ValueError('Question manual missing from index')
                retrieval=[]
                update(stage='retrieval')
                for i,t in enumerate(translations,1):
                    begin=time.perf_counter()
                    dense=app.retrieve(db,model,t['manual_id'],t['english_query'],20)
                    keyword=bm25.search(t['manual_id'],t['english_query'],20)
                    fusion=hybrid.fuse_rrf(dense,keyword)
                    variants={}
                    for name,hits in [(VARIANTS[0],dense[:5]),
                                      (VARIANTS[1],[dict(by_id[r['chunk_id']],score=r['score']) for r in fusion[:5]])]:
                        sources,excluded=answering.prepare_context(t['question'],hits,t['manual_id'])
                        variants[name]={'sources':sources,'excluded_sources':excluded,
                                        'retrieved_chunk_ids':[h['chunk_id'] for h in hits]}
                    retrieval.append({k:t[k] for k in ('case_id','manual_id','question','english_query')} |
                        {'dense_candidates':dense,'keyword_candidates':keyword,'fused_candidates':fusion,
                         'variants':variants,'retrieval_seconds':time.perf_counter()-begin})
                    app.write_json(dest/'retrieval-cases'/(t['case_id']+'.json'),retrieval[-1])
                    update(retrieved=i)
                    print(f"Retrieved {i}/{len(questions)} {t['case_id']} B and C",flush=True)
                update(index_points=db.count(app.COLLECTION,exact=True).count,
                       query_embeddings_generated=model.generated,query_embedding_cache_hits=model.reused)
                unload(args.url,args.model)
            finally:
                db.close();model.close();bm25.close()
            del chunks,by_id,db,model,bm25
            gc.collect()
            app.write_json(retrieval_path,retrieval)
        update(stage='generation',processed=0)
        all_results=[]
        for variant in VARIANTS:
            results=[]
            for item in retrieval:
                path=dest/variant/'responses'/(item['case_id']+'.json')
                if path.exists():
                    result=app.read_json(path)
                    if (result['question']!=item['question'] or result['english_query']!=item['english_query']
                            or result['retrieved_chunk_ids']!=item['variants'][variant]['retrieved_chunk_ids']):
                        raise ValueError('Saved response identity differs')
                else:
                    begin=time.perf_counter()
                    result={k:item[k] for k in ('case_id','manual_id','question','english_query')}
                    result.update(item['variants'][variant],variant=variant,created_at=now(),status='failed')
                    try:
                        value,generation=answering.generate_answer(item['question'],result['sources'],args.url,args.answer_model)
                        citations,figures=answering.resolve_citations(value,result['sources'])
                        result.update(value,generation=generation,citations=citations,cited_figures=figures,
                                      status='insufficient_evidence' if value['insufficient_evidence'] else 'answered')
                    except Exception as error:
                        result['error']=str(error)
                        if isinstance(error,answering.AnswerGenerationError):
                            result['failed_attempts']=error.attempts
                    result.update(elapsed_seconds=time.perf_counter()-begin,gpu_observation=app.gpu_snapshot())
                    app.write_json(path,result)
                results.append(result);all_results.append(result)
                update(processed=len(all_results),variant=variant,
                       answered=sum(r['status']=='answered' for r in all_results),
                       abstained=sum(r['status']=='insufficient_evidence' for r in all_results),
                       errors=sum(r['status']=='failed' for r in all_results))
                print(f"Generated {len(all_results)}/{len(questions)*2} {variant} {item['case_id']}: {result['status']}",flush=True)
            app.write_json(dest/variant/'results.json',results)
        protected=app.read_json(args.output/'final-validation.json')['protected_files']
        preserved={p:app.sha(p)==v for p,v in protected.items()}
        checks={'protected_files':preserved,
            'chunks_unchanged':app.sha(args.output/'chunks.jsonl')==index_identity['chunks_sha256'],
            'baseline_results_unchanged':app.sha(args.baseline/'results.json')==config['baseline_results_sha256'],
            'baseline_questions_unchanged':app.sha(args.questions)==config['questions_sha256']}
        app.write_json(dest/'preservation.json',checks)
        if not all(preserved.values()) or not all(v for k,v in checks.items() if k!='protected_files'):
            raise ValueError('Input preservation check failed')
        update(stage='complete',status='completed_with_errors' if status['errors'] else 'complete')
        print(json.dumps(status,ensure_ascii=False),flush=True)
    except Exception as error:
        update(status='failed',error=str(error))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['translate','compare'],required=True)
    p.add_argument('--questions',type=Path,default=app.PM/'reports/qa30_20261102/questions.json')
    p.add_argument('--baseline',type=Path,default=app.PM/'reports/qa30_20261102/run')
    p.add_argument('--results',type=Path,default=app.PM/'reports/qa30_hybrid_20261102')
    p.add_argument('--output',type=Path,default=app.FULL_OUTPUT)
    p.add_argument('--bm25-index',type=Path,default=app.REPO/'.runtime/pm209-hybrid/bm25-v1.sqlite3')
    p.add_argument('--url',default='http://127.0.0.1:11434')
    p.add_argument('--model',default=app.MODEL)
    p.add_argument('--answer-model',default=answering.MODEL)
    run(p.parse_args())


if __name__=='__main__':
    main()
