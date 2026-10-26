"""Durable, single-process PM209 full embedding/index runner; no QA evaluation."""
import argparse
from collections import deque
import ctypes
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
import traceback

import numpy as np
from qdrant_client import QdrantClient

import rag_pipeline as app

RUNTIME = app.REPO / '.runtime/pm209-rag-full'
OUTPUT = app.DATA / 'rag/full_char500_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def validate_output(args, progress, protected):
    manifest, chunks = app.checked_chunks(args)
    app.check_sources(manifest)
    expected = {c['chunk_id']: c for c in chunks}
    errors, seen, dimensions = [], set(), set()
    progress('validate', completed=0, total=len(chunks))
    db = QdrantClient(path=str(args.output/'qdrant'))
    try:
        offset = None
        while True:
            batch, offset = db.scroll(app.COLLECTION, limit=128, offset=offset, with_payload=True, with_vectors=True)
            for point in batch:
                rid = str(point.id)
                seen.add(rid)
                if point.payload != expected.get(rid):
                    errors.append({'id': rid, 'error': 'payload mismatch'})
                v = np.asarray(point.vector)
                dimensions.add(len(v))
                if not np.isfinite(v).all() or np.linalg.norm(v) == 0:
                    errors.append({'id': rid, 'error': 'invalid vector'})
                if {'qa_data','question','answer','relevant'} & set(point.payload):
                    errors.append({'id': rid, 'error': 'QA payload leakage'})
            progress('validate', completed=len(seen), total=len(chunks))
            if offset is None:
                break
        if seen != set(expected):
            errors.append({'error': 'point ID set mismatch'})
        if dimensions != {1024}:
            errors.append({'error': 'dimension mismatch', 'dimensions': sorted(dimensions)})
        # A stored document vector must retrieve itself inside its own manual.
        first = chunks[0]
        saved = db.retrieve(app.COLLECTION, [first['chunk_id']], with_vectors=True)[0]
        from qdrant_client import models
        hits = db.query_points(app.COLLECTION, query=saved.vector,
            query_filter=models.Filter(must=[models.FieldCondition(key='manual_id',match=models.MatchValue(value=first['manual_id']))]), limit=5).points
        if not hits or hits[0].score < .999 or any(p.payload['manual_id'] != first['manual_id'] for p in hits):
            errors.append({'error': 'stored-vector search/filter check failed'})
    finally:
        db.close()
    after = {filename: app.sha(filename) for filename in protected}
    if after != protected:
        errors.append({'error': 'protected source file hashes changed'})
    result = {'checked_at': now(), 'status': 'complete' if not errors else 'needs_review',
              'pages': manifest['pages'], 'manuals': len(manifest['identity']['manuals']),
              'chunks': len(chunks), 'points_verified': len(seen), 'dimension': sorted(dimensions),
              'source_hashes_unchanged': after == protected, 'protected_files': protected,
              'over_500': manifest['over_500'], 'quality_flags': manifest['quality_flags'],
              'errors': errors, 'semantic_accuracy_verified': False,
              'qa_evaluation_run': False, 'review': str(args.output/'review.html')}
    app.write_json(args.output/'final-validation.json', result)
    app.write_json(RUNTIME/'final-validation.json', result)
    if errors:
        raise ValueError(f'Final validation failed: {errors[:3]}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', required=True)
    parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    lock = (RUNTIME/'runner.lock').open('a+b')
    if lock.seek(0, 2) == 0:
        lock.write(b'0'); lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        raise SystemExit('Another PM209 full RAG process is active')
    if (RUNTIME/'STOP').exists():
        raise SystemExit('STOP exists. Remove it explicitly before resuming.')

    started = time.monotonic()
    state = {'status':'running','stage':'prepare','pid':os.getpid(),'started_at':now(),
             'updated_at':now(),'last_event_at':now(),'completed':0,'total':None,
             'output':str(OUTPUT),'model':app.MODEL,'batch_size':4,'eta_seconds':None,
             'eta_scope':'current stage only','embedded':0,'indexed':0,'failed':0,
             'max_gpu_used_mib':0,'max_gpu_temperature_c':0}
    mutex = threading.RLock()
    stopped = threading.Event()
    history = deque(maxlen=200)
    last_write = [0.0]

    def publish():
        state.update(updated_at=now(), elapsed_seconds=round(time.monotonic()-started,1))
        app.write_json(RUNTIME/'status.json', dict(state))

    def progress(stage, **values):
        if (RUNTIME/'STOP').exists():
            raise InterruptedError('STOP requested; committed cache/index remains reusable')
        current = time.monotonic()
        with mutex:
            changed = stage != state['stage']
            if changed:
                history.clear()
                state.update(completed=0,total=None,eta_seconds=None,stage_started_at=now())
            state.update(stage=stage,last_event_at=now(),**values)
            if stage == 'embed':
                state['embedded'] = values.get('completed', state['embedded'])
            elif stage == 'index':
                state['indexed'] = values.get('completed', state['indexed'])
            count = state.get('processed',0) if stage == 'embed' else state.get('completed',0)
            history.append((current,count))
            if len(history) > 2 and history[-1][0]-history[0][0] > 15 and state.get('total'):
                delta = history[-1][1]-history[0][1]
                if delta > 0:
                    state['eta_seconds'] = round((state['total']-count)*(history[-1][0]-history[0][0])/delta)
            if changed or current-last_write[0] >= 5 or values.get('stage_complete'):
                publish();last_write[0]=current

    def monitor():
        while not stopped.wait(5):
            snapshot = app.gpu_snapshot()
            with mutex:
                if snapshot:
                    state['gpu'] = snapshot
                    state['max_gpu_used_mib'] = max(state['max_gpu_used_mib'],snapshot['used_mib'])
                    state['max_gpu_temperature_c'] = max(state['max_gpu_temperature_c'],snapshot['temperature_c'])
                publish()

    args = argparse.Namespace(input=app.DEFAULT_INPUT, output=OUTPUT, all=True, manual=None,
        split=['train','val','test'], tokenizer=app.REPO/'.runtime/pm209-token-length/tokenizer.json',
        model=app.MODEL, url='http://127.0.0.1:11434', batch_size=4, progress=progress)
    publish()
    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    # Prevent automatic idle sleep for this process only; allow the display off.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        sources = [app.DATA/'data'/folder/(split+'.jsonl')
                   for folder in ['', 'vlm', 'paddle_tables_manual_20261026']
                   for split in args.split]
        protected_path=RUNTIME/'source-hashes.json'
        current={str(p):app.sha(p) for p in sources}
        if protected_path.exists() and app.read_json(protected_path) != current:
            raise ValueError('Protected sources changed since the previous full run')
        app.write_json(protected_path,current)
        manifest_path=OUTPUT/'chunks-manifest.json'
        if not manifest_path.exists():
            app.build_chunks(args)
            gc.collect()
        manifest=app.read_json(manifest_path)
        if manifest['identity'] != app.input_identity(args,list(app.read_json(app.DATA/'meta_data/name2meta.json'))):
            raise ValueError('Full chunk input identity mismatch')
        app.check_sources(manifest)
        if manifest['pages'] != 10018 or len(manifest['identity']['manuals']) != 209 or manifest['missing_nonempty_regions']:
            raise ValueError('Full corpus count/coverage mismatch')
        if manifest['max_token_count'] > 2000:
            raise ValueError('Oversized token input needs review before embedding')
        with mutex:
            state.update(pages=manifest['pages'],manuals=209,chunks=manifest['chunks'],
                         over_500=manifest['over_500'],chunk_quality_flags=manifest['quality_flags'])
            publish()
        sample_cache=app.DEFAULT_OUTPUT/'embedding-cache.sqlite3'
        target_cache=OUTPUT/'embedding-cache.sqlite3'
        if not target_cache.exists() and sample_cache.exists():
            src=sqlite3.connect(sample_cache.resolve().as_uri()+'?mode=ro',uri=True)
            dst=sqlite3.connect(target_cache)
            try:
                src.backup(dst)
            finally:
                src.close();dst.close()
        print(f"Starting embeddings for {manifest['chunks']} chunks / 209 manuals",flush=True)
        app.embed(args)
        gc.collect()
        app.index(args)
        gc.collect()
        result=validate_output(args,progress,current)
        with mutex:
            state.update(status='complete',stage='complete',completed=result['chunks'],total=result['chunks'],
                         embedded=result['chunks'],indexed=result['chunks'],eta_seconds=0,completed_at=now(),last_event_at=now())
            publish()
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        with mutex:
            state.update(status='stopped' if isinstance(exc,(KeyboardInterrupt,InterruptedError)) else 'failed',
                         error=str(exc),last_event_at=now(),eta_seconds=None)
            publish()
        traceback.print_exc()
        raise
    finally:
        stopped.set();worker.join(timeout=6)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    main()
