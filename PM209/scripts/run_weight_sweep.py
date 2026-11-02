"""Compare RRF keyword weights using frozen English queries and search candidates.

Does not translate, embed, search Qdrant, or read QA references. Only rank weights change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import httpx
import rag_pipeline as app
import answering
import hybrid_retrieval as hybrid


def now():
    return datetime.now(timezone.utc).isoformat()


def variant_name(percent):
    return f'keyword_{percent:03d}'


def run(args):
    if len(args.keyword_percent)!=len(set(args.keyword_percent)) or any(p not in (0,25,50,75,100) for p in args.keyword_percent):
        raise ValueError('Use distinct weights from 0,25,50,75,100')
    weights=sorted(args.keyword_percent)
    source=args.source
    old=app.read_json(source/'config.json')
    if app.sha(answering.__file__)!=old['answering_code_sha256'] or app.sha(hybrid.__file__)!=old['hybrid_code_sha256']:
        raise ValueError('Answering or rank fusion code changed since the frozen experiment')
    if app.read_json(args.output/'index-identity.json')!=old['index_identity']:
        raise ValueError('Index identity changed')
    frozen=app.read_json(source/'frozen-input.json')
    if app.sha(source/'translations.json')!=frozen['translations_sha256']:
        raise ValueError('Frozen English queries changed')
    protected_inputs={str((source/p).resolve()):app.sha(source/p) for p in
        ['config.json','retrieval.json','translations.json','translation-review.json','frozen-input.json',
         'B_english_dense/results.json','C_english_hybrid/results.json']}
    with httpx.Client(base_url=args.url,timeout=10,trust_env=False) as client:
        response=client.get('/api/tags');response.raise_for_status()
        digests={m['name']:m['digest'] for m in response.json()['models']}
    if digests.get(old['answer_model'])!=old['model_digests'][old['answer_model']]:
        raise ValueError('Answer model weights changed')
    retrieved=app.read_json(source/'retrieval.json')
    translated={t['case_id']:t for t in app.read_json(source/'translations.json')}
    if len(retrieved)!=30 or len({r['case_id'] for r in retrieved})!=30:
        raise ValueError('Expected the frozen 30-question experiment')
    for item in retrieved:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',item['case_id']):
            raise ValueError('Invalid case ID')
        if any(item[k]!=translated[item['case_id']][k] for k in ('case_id','manual_id','question','english_query')):
            raise ValueError('Saved query identity mismatch')
        if any(h['manual_id']!=item['manual_id'] for h in item['dense_candidates']):
            raise ValueError('Candidate list crosses manuals')
        check=hybrid.fuse_rrf(item['dense_candidates'],item['keyword_candidates'],0.5,0.5,k=60)
        if check!=item['fused_candidates']:
            raise ValueError('Cannot reproduce the saved 50:50 ranking')
    config={'keyword_percent':weights,'dense_percent':[100-p for p in weights],
        'rrf_k':60,'candidate_k':20,'final_k':5,'questions':30,
        'source':str(source.resolve()),'protected_inputs':protected_inputs,
        'index_identity':old['index_identity'], 'runner_code_sha256':app.sha(__file__),
        'answering_code_sha256':app.sha(answering.__file__),'hybrid_code_sha256':app.sha(hybrid.__file__),
        'answer_model':old['answer_model'],'answer_model_digest':digests[old['answer_model']],
        'answer_options':answering.OPTIONS,'prompt_version':answering.PROMPT_VERSION,
        'prompt_sha256':hashlib.sha256(answering.SYSTEM_PROMPT.encode()).hexdigest(),
        'translation_rerun':False,'embedding_rerun':False,'retrieval_rerun':False,
        'answer_reuse':'Only existing same-variant files on resume; all new weight cases invoke the model.'}
    dest=args.results;dest.mkdir(parents=True,exist_ok=True)
    cp=dest/'config.json'
    if cp.exists() and app.read_json(cp)!=config:
        raise ValueError('Configuration changed; use a new result directory')
    app.write_json(cp,config)
    started=time.perf_counter()
    status={'status':'running','started_at':now(),'total':30*len(weights),'processed':0}
    def update(**values):
        status.update(values,updated_at=now(),elapsed_seconds=time.perf_counter()-started)
        app.write_json(dest/'status.json',status)
    update(stage='rank_fusion')
    try:
        plans={p:[] for p in weights};needed=set()
        for p in weights:
            for item in retrieved:
                fusion=hybrid.fuse_rrf(item['dense_candidates'],item['keyword_candidates'],(100-p)/100,p/100,k=60)
                chosen=[r['chunk_id'] for r in fusion[:5]]
                needed.update(chosen)
                plans[p].append({k:item[k] for k in ('case_id','manual_id','question','english_query')} |
                    {'keyword_percent':p,'dense_percent':100-p,'fused_candidates':fusion,'retrieved_chunk_ids':chosen})
        if app.sha(args.output/'chunks.jsonl')!=old['index_identity']['chunks_sha256']:
            raise ValueError('Chunk contents changed')
        # Stream the corpus; retain just the selected payloads, never load Qdrant into RAM.
        by_id={c['chunk_id']:c for c in app.read_lines(args.output/'chunks.jsonl') if c['chunk_id'] in needed}
        if set(by_id)!=needed:
            raise ValueError('Saved candidate points are missing from the current corpus')
        for p,items in plans.items():
            for item in items:
                hits=[dict(by_id[r['chunk_id']],score=r['score']) for r in item['fused_candidates'][:5]]
                sources,excluded=answering.prepare_context(item['question'],hits,item['manual_id'])
                item.update(sources=sources,excluded_sources=excluded)
            path=dest/variant_name(p)/'retrieval.json'
            if path.exists() and app.read_json(path)!=items:
                raise ValueError('Saved weighted retrieval differs')
            app.write_json(path,items)
        del by_id
        update(stage='generation',selected_payloads=len(needed))
        all_results=[]
        for p,items in plans.items():
            results=[];variant=variant_name(p)
            for item in items:
                path=dest/variant/'responses'/(item['case_id']+'.json')
                if path.exists():
                    result=app.read_json(path)
                    if any(result[k]!=item[k] for k in ('question','manual_id','english_query','retrieved_chunk_ids','sources')):
                        raise ValueError('Saved answer input differs')
                else:
                    begin=time.perf_counter()
                    result={k:item[k] for k in ('case_id','manual_id','question','english_query','keyword_percent',
                        'dense_percent','retrieved_chunk_ids','sources','excluded_sources')}
                    result.update(status='failed',variant=variant,created_at=now())
                    try:
                        value,generation=answering.generate_answer(item['question'],item['sources'],args.url,old['answer_model'])
                        citations,figures=answering.resolve_citations(value,item['sources'])
                        result.update(value,generation=generation,citations=citations,cited_figures=figures,
                                      status='insufficient_evidence' if value['insufficient_evidence'] else 'answered')
                    except Exception as error:
                        result['error']=str(error)
                        if isinstance(error,answering.AnswerGenerationError):result['failed_attempts']=error.attempts
                    result.update(elapsed_seconds=time.perf_counter()-begin,gpu_observation=app.gpu_snapshot())
                    app.write_json(path,result)
                results.append(result);all_results.append(result)
                update(processed=len(all_results),keyword_percent=p,
                    answered=sum(r['status']=='answered' for r in all_results),
                    abstained=sum(r['status']=='insufficient_evidence' for r in all_results),
                    errors=sum(r['status']=='failed' for r in all_results))
                print(f"Processed {len(all_results)}/{status['total']} keyword={p}% {item['case_id']}: {result['status']}",flush=True)
            app.write_json(dest/variant/'results.json',results)
        protected=app.read_json(args.output/'final-validation.json')['protected_files']
        preservation={'source_files':{p:app.sha(p)==v for p,v in protected.items()},
            'previous_experiment_inputs_and_results':{p:app.sha(p)==v for p,v in protected_inputs.items()},
            'chunks_unchanged':app.sha(args.output/'chunks.jsonl')==old['index_identity']['chunks_sha256']}
        app.write_json(dest/'preservation.json',preservation)
        if not preservation['chunks_unchanged'] or not all(preservation['source_files'].values()) or not all(preservation['previous_experiment_inputs_and_results'].values()):
            raise ValueError('Preservation check failed')
        update(stage='complete',status='completed_with_errors' if status['errors'] else 'complete')
        print(json.dumps(status,ensure_ascii=False),flush=True)
    except Exception as error:
        update(status='failed',error=str(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=app.PM/'reports/qa30_hybrid_20261102')
    p.add_argument('--results',type=Path,default=app.PM/'reports/qa30_weights_20261102')
    p.add_argument('--output',type=Path,default=app.FULL_OUTPUT)
    p.add_argument('--keyword-percent',type=int,nargs='+',default=[25,75,100])
    p.add_argument('--url',default='http://127.0.0.1:11434')
    run(p.parse_args())


if __name__=='__main__':main()
