"""Run a fixed question-only evaluation batch; references are never read here."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import time

import rag_pipeline as app
import answering


def now():
    return datetime.now(timezone.utc).isoformat()


def run(args):
    questions = app.read_json(args.questions)
    if not isinstance(questions, list) or not 1 <= len(questions) <= 100:
        raise ValueError('Use an explicit question list with 1..100 cases')
    for q in questions:
        if set(q) != {'case_id', 'manual_id', 'question'}:
            raise ValueError('Question input must not contain answers, references, or other fields')
        if not all(isinstance(v, str) and v.strip() for v in q.values()):
            raise ValueError('Question fields must be nonempty strings')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', q['case_id']):
            raise ValueError('Invalid case ID')
        answering.prepare_context(q['question'], [], q['manual_id'])
    if len({q['case_id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate case IDs')
    config = {'questions_sha256': app.sha(args.questions), 'top_k': args.top_k,
              'embedding_model': args.model, 'answer_model': args.answer_model,
              'index_identity': app.read_json(args.output/'index-identity.json'),
              'answering_code_sha256': app.sha(answering.__file__),
              'runner_code_sha256': app.sha(__file__), 'options': answering.OPTIONS,
              'prompt_version': answering.PROMPT_VERSION,
              'prompt_sha256': hashlib.sha256(answering.SYSTEM_PROMPT.encode()).hexdigest()}
    destination = args.results
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination/'config.json'
    if config_path.exists() and app.read_json(config_path) != config:
        raise ValueError('Evaluation configuration changed; use a new results folder')
    app.write_json(config_path, config)
    started = time.perf_counter()
    status = {'status': 'running', 'started_at': now(), 'total': len(questions), 'processed': 0}

    def update(**values):
        status.update(values, updated_at=now(), elapsed_seconds=time.perf_counter()-started)
        app.write_json(destination/'status.json', status)

    update(stage='retrieval')
    retrieval_path = destination/'retrieval.json'
    if retrieval_path.exists():
        retrieved = app.read_json(retrieval_path)
        if [r['case_id'] for r in retrieved] != [q['case_id'] for q in questions]:
            raise ValueError('Saved retrieval case IDs differ')
    else:
        print('Loading full index once for the fixed evaluation questions...', flush=True)
        manifest, chunks, model, db = app.open_search(args)
        try:
            manuals = {c['manual_id'] for c in chunks}
            if any(q['manual_id'] not in manuals for q in questions):
                raise ValueError('A selected manual is missing from the index')
            retrieved = []
            for index, q in enumerate(questions, 1):
                begin = time.perf_counter()
                hits = app.retrieve(db, model, q['manual_id'], q['question'], args.top_k)
                sources, excluded = answering.prepare_context(q['question'], hits, q['manual_id'])
                retrieved.append({**q, 'retrieved_chunk_ids': [h['chunk_id'] for h in hits],
                                  'sources': sources, 'excluded_sources': excluded,
                                  'retrieval_seconds': time.perf_counter()-begin})
                update(retrieved=index)
                print(f"Retrieved {index}/{len(questions)} {q['case_id']}", flush=True)
            released = model.client.post('/api/generate', json={'model': args.model, 'keep_alive': 0})
            released.raise_for_status()
            update(index_points=db.count(app.COLLECTION, exact=True).count)
        finally:
            db.close(); model.close()
        del chunks, db, model
        gc.collect()
        app.write_json(retrieval_path, retrieved)
    update(stage='generation')
    results = []
    for index, item in enumerate(retrieved, 1):
        path = destination/'responses'/(item['case_id']+'.json')
        if path.exists():
            result = app.read_json(path)
            if result['question'] != item['question'] or result['manual_id'] != item['manual_id']:
                raise ValueError('Saved response identity differs')
        else:
            begin = time.perf_counter()
            result = dict(item, created_at=now(), status='failed')
            try:
                value, generation = answering.generate_answer(item['question'], item['sources'],
                                                             args.url, args.answer_model)
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
        update(processed=index, answered=sum(r['status']=='answered' for r in results),
               abstained=sum(r['status']=='insufficient_evidence' for r in results),
               errors=sum(r['status']=='failed' for r in results))
        print(f"Generated {index}/{len(questions)} {item['case_id']}: {result['status']}", flush=True)
    # Compare source files with the preserved full-index baseline. No QA is parsed.
    baseline = app.read_json(args.output/'final-validation.json')['protected_files']
    preserved = {p: app.sha(p)==expected for p, expected in baseline.items()}
    chunk_preserved = app.sha(args.output/'chunks.jsonl') == config['index_identity']['chunks_sha256']
    app.write_json(destination/'preservation.json', {'protected_files': preserved,
                   'chunks_unchanged': chunk_preserved, 'index_identity': config['index_identity']})
    if not all(preserved.values()) or not chunk_preserved:
        update(status='failed', error='Preservation check failed')
        raise ValueError('Preservation check failed')
    app.write_json(destination/'results.json', results)
    update(stage='complete', status='completed_with_errors' if status['errors'] else 'complete')
    print(json.dumps(status, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path, default=app.FULL_OUTPUT)
    p.add_argument('--model', default=app.MODEL)
    p.add_argument('--answer-model', default=answering.MODEL)
    p.add_argument('--url', default='http://127.0.0.1:11434')
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.top_k <= 10:
        p.error('Use --top-k 1..10')
    run(args)


if __name__ == '__main__':
    main()
