"""Generate Top-1 answers for the five frozen English retrieval weight settings.

Only the first retrieved chunk is supplied. Does not read QA references, search,
translate, embed, change existing results, or reuse answers across conditions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx
import rag_pipeline as app
import answering
from run_weight_sweep import now, variant_name


def run(args):
    old_root = app.PM / 'reports/qa30_hybrid_20261102'
    weights_root = app.PM / 'reports/qa30_weights_20261102'
    old = app.read_json(old_root / 'config.json')
    sources = {0: old_root / 'B_english_dense', 50: old_root / 'C_english_hybrid'}
    sources.update({w: weights_root / variant_name(w) for w in (25, 75, 100)})
    weights = [0, 25, 50, 75, 100]
    if app.sha(answering.__file__) != old['answering_code_sha256']:
        raise ValueError('Answering code changed since the frozen Top-5 comparison')
    protected = {str(p.resolve()): app.sha(p)
                 for directory in (old_root, weights_root)
                 for p in sorted(directory.rglob('*')) if p.is_file()}
    translations = {r['case_id']: r for r in app.read_json(old_root / 'translations.json')}
    frozen = app.read_json(old_root / 'frozen-input.json')
    if app.sha(old_root / 'translations.json') != frozen['translations_sha256']:
        raise ValueError('Frozen English queries changed')
    with httpx.Client(base_url=args.url, timeout=10, trust_env=False) as client:
        response = client.get('/api/tags')
        response.raise_for_status()
        digests = {m['name']: m['digest'] for m in response.json()['models']}
    model = old['answer_model']
    if digests.get(model) != old['model_digests'][model]:
        raise ValueError('Answer model weights changed')
    plans = {}
    for w in weights:
        rows = app.read_json(sources[w] / 'results.json')
        if len(rows) != 30 or {r['case_id'] for r in rows} != set(translations):
            raise ValueError('Expected the same 30 questions')
        plans[w] = []
        for r in rows:
            if any(r[k] != translations[r['case_id']][k]
                   for k in ('question', 'english_query', 'manual_id')):
                raise ValueError('Question identity changed')
            if len(r['sources']) != 5 or r['excluded_sources']:
                raise ValueError('Top-5 context no longer matches the frozen experiment')
            first = r['sources'][0]
            if first['chunk_id'] != r['retrieved_chunk_ids'][0] or first['source_id'] != 'S1':
                raise ValueError('First context is not the first retrieved chunk')
            context, excluded = answering.prepare_context(r['question'], [first], r['manual_id'])
            if context != [first] or excluded:
                raise ValueError('Top-1 context changed unexpectedly')
            plans[w].append({k: r[k] for k in ('case_id', 'manual_id', 'question', 'english_query')} | {
                'keyword_percent': w, 'dense_percent': 100-w, 'top_k': 1,
                'retrieved_chunk_ids': [first['chunk_id']], 'sources': context,
                'excluded_sources': [], 'top5_response_file': str((sources[w] / 'results.json').resolve())})
    config = {'keyword_percent': weights, 'final_k': 1, 'questions': 30,
              'rrf_k': 60, 'candidate_k': 20, 'ranking_source': 'First chunk of each frozen Top-5 result',
              'answer_model': model, 'answer_model_digest': digests[model],
              'answer_options': answering.OPTIONS, 'prompt_version': answering.PROMPT_VERSION,
              'prompt_sha256': hashlib.sha256(answering.SYSTEM_PROMPT.encode()).hexdigest(),
              'answering_code_sha256': app.sha(answering.__file__), 'runner_code_sha256': app.sha(__file__),
              'protected_previous_files': protected, 'translation_rerun': False,
              'embedding_rerun': False, 'retrieval_rerun': False,
              'answer_reuse': 'Only same-condition files on resume; all 150 new cases invoke the model.'}
    dest = args.results
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'config.json').exists() and app.read_json(dest / 'config.json') != config:
        raise ValueError('Configuration changed; use a new result directory')
    app.write_json(dest / 'config.json', config)
    start = time.perf_counter()
    status = {'status': 'running', 'started_at': now(), 'total': 150, 'processed': 0}

    def update(**values):
        status.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-start)
        app.write_json(dest / 'status.json', status)

    update(stage='generation')
    all_results = []
    try:
        for w, items in plans.items():
            variant = variant_name(w)
            app.write_json(dest / variant / 'retrieval.json', items)
            results = []
            for item in items:
                path = dest / variant / 'responses' / (item['case_id']+'.json')
                if path.exists():
                    result = app.read_json(path)
                    if any(result[k] != v for k, v in item.items()):
                        raise ValueError('Saved response input differs')
                else:
                    begin = time.perf_counter()
                    result = dict(item, status='failed', variant=variant, created_at=now())
                    try:
                        value, generation = answering.generate_answer(item['question'], item['sources'], args.url, model)
                        citations, figures = answering.resolve_citations(value, item['sources'])
                        result.update(value, generation=generation, citations=citations, cited_figures=figures,
                                      status='insufficient_evidence' if value['insufficient_evidence'] else 'answered')
                    except Exception as error:
                        result['error'] = str(error)
                        if isinstance(error, answering.AnswerGenerationError):
                            result['failed_attempts'] = error.attempts
                    result.update(elapsed_seconds=time.perf_counter()-begin, gpu_observation=app.gpu_snapshot())
                    app.write_json(path, result)
                results.append(result)
                all_results.append(result)
                update(processed=len(all_results), keyword_percent=w,
                       answered=sum(x['status']=='answered' for x in all_results),
                       abstained=sum(x['status']=='insufficient_evidence' for x in all_results),
                       errors=sum(x['status']=='failed' for x in all_results))
                print(f"Processed {len(all_results)}/150 keyword={w}% {item['case_id']}: {result['status']}", flush=True)
            app.write_json(dest / variant / 'results.json', results)
        update(stage='preservation_validation')
        original = app.read_json(app.FULL_OUTPUT / 'final-validation.json')['protected_files']
        identity = old['index_identity']
        preservation = {
            'source_files': {p: app.sha(p)==v for p, v in original.items()},
            'previous_experiments': {p: app.sha(p)==v for p, v in protected.items()},
            'chunks_unchanged': app.sha(app.FULL_OUTPUT/'chunks.jsonl')==identity['chunks_sha256']}
        app.write_json(dest / 'preservation.json', preservation)
        if not (all(preservation['source_files'].values()) and
                all(preservation['previous_experiments'].values()) and preservation['chunks_unchanged']):
            raise ValueError('Preservation check failed')
        update(stage='complete', status='completed_with_errors' if status['errors'] else 'complete')
        print(json.dumps(status, ensure_ascii=False), flush=True)
    except Exception as error:
        update(status='failed', error=str(error))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, default=app.PM/'reports/qa30_top1_20261102')
    p.add_argument('--url', default='http://127.0.0.1:11434')
    run(p.parse_args())


if __name__ == '__main__':
    main()
