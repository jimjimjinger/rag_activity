"""Recover only the 232 failed Paddle tables into a separate, resumable output."""
import argparse
from collections import Counter
import ctypes
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'PM209/src'))
import paddle_tables as app
from repair_paddle_html import normalize

OLD = app.OUTPUT
OUT = OLD.with_name('paddle_tables_html_repaired')
RT = ROOT / '.runtime/pm209-paddle-repair'
FAILURES = ROOT / '.runtime/pm209-paddle-tables/table-failures.jsonl'


def table_fragment(content):
    return content.removeprefix('<html><body>').removesuffix('</body></html>')


def failures():
    data = [json.loads(line) for line in FAILURES.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(data) != 232 or len({t['key'] for t in data}) != 232:
        raise ValueError('Expected exactly 232 distinct failed inputs')
    if any(app.asset_dir(t, OLD).joinpath('complete.json').exists() for t in data):
        raise ValueError('Failure list overlaps existing successful tables')
    return data


def original_tree():
    return {str(p.relative_to(OLD)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in OLD.rglob('*') if p.is_file()}


def protection():
    return dict(protected=app.hashes(), original_jsonl={s: app.digest(OLD / (s+'.jsonl')) for s in app.SPLITS},
                original_tree=original_tree(), failures_sha256=app.digest(FAILURES))


def ensure_manifest():
    p = RT / 'before.json'
    if p.exists(): return app.read_json(p)
    result = protection()
    expected = app.read_json(ROOT / '.runtime/pm209-paddle-tables/final-validation.json')['protected_sha256']
    if expected != result['protected']: raise ValueError('Protected originals changed before repair')
    app.atomic_json(p, result)
    return result


def target_record(t):
    p = app.asset_dir(t, OUT) / 'complete.json'
    if not p.exists(): return None
    r = app.read_json(p)
    if r['source']['region'] != t['region'] or r['source']['page_id'] != t['page_id']:
        raise ValueError('Completed recovery source changed')
    for name, digest in r['artifact_sha256'].items():
        if app.digest(p.parent / name) != digest: raise ValueError('Completed recovery artifact changed')
    normalize(r['table_content'])
    return r


def save(t, raw, content, audit, seconds, method, variant=None):
    from PIL import Image
    folder = app.asset_dir(t, OUT)
    if target_record(t): raise ValueError('Refusing to replace an accepted recovery')
    source = app.asset_dir(t, OLD)
    audit = dict(audit, method=method, seconds=seconds, variant=variant,
                 original_raw_sha256=app.digest(source/'failed.raw.json'),
                 pipeline_raw_html_preserved=True, semantic_accuracy_verified=False)
    app.atomic_json(folder/'repair.json', audit)
    image, box = app.crop(t)
    with Image.open(source/'image.png') as previous:
        if image.size != previous.size or image.tobytes() != previous.convert('RGB').tobytes():
            raise ValueError('Original crop changed')
    stamp = app.source_stamp(t, app.digest(app.image_path(t)))
    record = app.save_record(t, stamp, image, box, content, raw, seconds, method, output=OUT)
    record['repair'] = audit
    if 'image_review' in audit:
        record['quality']['manual_review']='reviewed_with_known_issues'
        record['quality']['flags'].append('see_image_review_notes')
    record['artifact_sha256']['repair.json'] = app.digest(folder/'repair.json')
    app.atomic_json(folder/'complete.json', record)
    return record


def repair(data):
    pending = []
    for t in data:
        if target_record(t): continue
        tick = time.monotonic()
        raw = app.read_json(app.asset_dir(t, OLD)/'failed.raw.json')
        try:
            content, audit = normalize(raw['table_res_list'][0]['pred_html'])
            save(t, raw, content, audit, time.monotonic()-tick, 'normalized_original')
        except ValueError as exc:
            pending.append(dict(key=t['key'], split=t['split'], table_id=t['region']['id'], reason=str(exc)))
    app.atomic_json(RT/'pending.json', pending)
    print(json.dumps({'normalized':sum(bool(target_record(t)) for t in data), 'pending':len(pending),
                      'reasons':dict(Counter(t['reason'] for t in pending))}), flush=True)


def retry(data, review=False, e2e=False):
    pending_keys = {t['key'] for t in app.read_json(RT/'pending.json')}
    pending = [t for t in data if t['key'] in pending_keys and not target_record(t)]
    if review:
        decisions=app.read_json(RT/'manual-decisions.json')
        pending=[t for t in pending if decisions.get(t['key'],{}).get('decision')=='needs_retry']
    if not pending: return
    app.configure_environment()
    os.environ['DISABLE_MODEL_SOURCE_CHECK']='True'
    import numpy as np
    import paddle
    from paddleocr import TableRecognitionPipelineV2
    if not paddle.is_compiled_with_cuda(): raise RuntimeError('GPU build required')
    paddle.set_device('gpu:0')
    config = app.settings()
    pipeline = TableRecognitionPipelineV2(**config['init'])
    pipeline.export_paddlex_config_to_yaml(str(RT/'pipeline.yaml'))
    state = dict(status='running', pid=os.getpid(), planned=len(pending), processed=0, accepted=0,
                 attempts=0, max_gpu_used_mib=0, max_gpu_temperature_c=0, started_at=app.now())
    stop = threading.Event()
    def monitor():
        while not stop.is_set():
            try:
                r = subprocess.run(['nvidia-smi','--query-gpu=memory.used,temperature.gpu','--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW)
                used, temp = map(int, r.stdout.splitlines()[0].split(','))
                state.update(gpu_used_mib=used, gpu_temperature_c=temp, updated_at=app.now())
                state['max_gpu_used_mib'] = max(state['max_gpu_used_mib'],used)
                state['max_gpu_temperature_c'] = max(state['max_gpu_temperature_c'],temp)
                app.atomic_json(RT/('gpu-status-e2e.json' if e2e else 'gpu-status-review.json' if review else 'gpu-status.json'), state)
            except Exception: pass
            stop.wait(2)
    thread = threading.Thread(target=monitor, daemon=True); thread.start()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        # Use the pipeline's cell-box-to-HTML option to avoid the repeated SLANeXt
        # structure tail. OCR and RT-DETR remain local; no VLM or QA input.
        for t in pending:
            if (RT/'STOP').exists(): raise KeyboardInterrupt('STOP requested')
            state['current_table_id'] = t['region']['id']
            variants=(('cells_global_ocr',1.0),('structure_padded',1.0)) if review else (('cells_to_html',1.0),('cells_to_html_1_5x',1.5))
            if e2e: variants=(('structure_e2e_padded',1.0),)
            for variant, scale in variants:
                folder = app.asset_dir(t, OUT)/'attempts'/variant
                if (folder/'result.json').exists():
                    if not review and app.read_json(folder/'result.json')['status']=='candidate': break
                    continue
                tick = time.monotonic()
                try:
                    image, _ = app.crop(t)
                    padding=24 if variant.startswith('structure_') else 0
                    if padding:
                        from PIL import ImageOps
                        image=ImageOps.expand(image,border=padding,fill='white')
                    if scale != 1:
                        from PIL import Image
                        image = image.resize((round(image.width*scale),round(image.height*scale)),Image.Resampling.LANCZOS)
                    pixels = np.asarray(image)[:,:,::-1].copy()
                    predict = dict(config['predict'], use_wired_table_cells_trans_to_html=True,
                                   use_wireless_table_cells_trans_to_html=True)
                    if review: predict['use_ocr_results_with_table_cells']=False
                    if variant.startswith('structure_'):
                        predict.update(use_wired_table_cells_trans_to_html=False,use_wireless_table_cells_trans_to_html=False)
                    if e2e: predict.update(use_e2e_wired_table_rec_model=True,use_e2e_wireless_table_rec_model=True)
                    results = pipeline.predict(pixels, **predict)
                    paddle.device.synchronize()
                    if len(results)!=1 or len(results[0]['table_res_list'])!=1:
                        raise ValueError('Expected one table')
                    raw = results[0].json['res']
                    app.atomic_json(folder/'raw.json', raw)
                    raw_html = raw['table_res_list'][0]['pred_html']
                    app.atomic_bytes(folder/'pipeline.html', raw_html.encode('utf-8'))
                    content,audit = normalize(raw_html)
                    # Successful syntax is a candidate for image review, not automatic acceptance.
                    app.atomic_bytes(folder/'candidate.html',content.encode('utf-8'))
                    record = dict(status='candidate', seconds=round(time.monotonic()-tick,3), scale=scale,padding=padding,
                                  predict=predict, audit=audit, semantic_accuracy_verified=False)
                except Exception as exc:
                    record = dict(status='failed', seconds=round(time.monotonic()-tick,3), scale=scale,
                                  error=f'{type(exc).__name__}: {exc}')
                app.atomic_json(folder/'result.json',record)
                state['attempts']+=1
                print(json.dumps(dict(key=t['key'],id=t['region']['id'],variant=variant,status=record['status'],
                                      seconds=record['seconds'],error=record.get('error'),
                                      rows=record.get('audit',{}).get('rows'),cells=record.get('audit',{}).get('cells')),
                                 ensure_ascii=False),flush=True)
                if record['status']=='candidate' and not review: break
                if any(v in record.get('error','').lower() for v in ('out of memory','illegal memory','device-side assert')):
                    raise RuntimeError(record['error'])
            state['processed']+=1
        state['status']='awaiting_image_review'
    finally:
        stop.set(); thread.join(timeout=10)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        state.update(finished_at=app.now(),current_table_id=None)
        app.atomic_json(RT/('gpu-status-e2e.json' if e2e else 'gpu-status-review.json' if review else 'gpu-status.json'),state)


def accept(data):
    decisions=app.read_json(RT/'manual-decisions.json')
    by_key={t['key']:t for t in data}
    for key,d in decisions.items():
        if d['decision']!='accept' or target_record(by_key[key]): continue
        t=by_key[key]; folder=app.asset_dir(t,OUT)/'attempts'/d['variant']
        result=app.read_json(folder/'result.json')
        if result['status']!='candidate': raise ValueError('Only candidates may be accepted')
        raw=app.read_json(folder/'raw.json')
        content,audit=normalize(raw['table_res_list'][0]['pred_html'])
        audit['image_review']=d
        save(t,raw,content,audit,result['seconds'],'paddle_cell_boxes_retry',
             dict(name=d['variant'],scale=result['scale'],padding=result.get('padding',0),predict=result['predict']))


def finalize(data):
    before=ensure_manifest()
    records={p.parent.name:app.read_json(p) for p in OLD.glob('assets/*/*/complete.json')}
    if len(records)!=2088: raise ValueError('Original successful count changed')
    restored={t['key']:target_record(t) for t in data}
    restored={k:r for k,r in restored.items() if r}
    if set(records)&set(restored): raise ValueError('Recovery would replace a successful table')
    records.update(restored)
    app.checkpoint(records,output=OUT)
    validation=app.validate_output(records,before['protected'],output=OUT)
    after=protection()
    if after!=before: raise ValueError('Original results changed during recovery')
    unresolved=[dict(key=t['key'],split=t['split'],table_id=t['region']['id']) for t in data if t['key'] not in restored]
    validation.update(original_successes_unchanged=True, original_result_files_unchanged=True,
                      recovered=len(restored), recovered_by_method=dict(Counter(r['provenance'] for r in restored.values())),
                      unresolved=unresolved, status='complete' if not unresolved else 'needs_review')
    # Compare the previous successful table_content strings in all full pages.
    from itertools import zip_longest
    for split in app.SPLITS:
        with (OLD/(split+'.jsonl')).open(encoding='utf-8') as a,(OUT/(split+'.jsonl')).open(encoding='utf-8') as b:
            for x,y in zip_longest(a,b):
                if x is None or y is None: raise ValueError('Page count differs')
                old,new=json.loads(x),json.loads(y)
                for u,v in zip(old['bounding_boxes'],new['bounding_boxes']):
                    if 'table_content' in u and u!=v: raise ValueError('Existing successful result changed')
    app.atomic_json(OUT/'final-validation.json',validation)
    app.atomic_json(RT/'final-validation.json',validation)
    rows=[]
    decisions=app.read_json(RT/'manual-decisions.json') if (RT/'manual-decisions.json').exists() else {}
    app.atomic_json(OUT/'manual-decisions.json',decisions)
    for t in data:
        key=t['key']; r=restored.get(key); base=f"assets/{t['split']}/{key}"
        source=f"../{OLD.name}/{base}"
        links=f'<a href="reviews/{key}.html">이미지와 결과 비교</a> · <a href="{source}/failed.raw.json">이전 원시 결과</a>'
        if r: links+=f' · <a href="{base}/table.html">보정 HTML</a> · <a href="{base}/repair.json">보정 기록</a>'
        else:
            for p in (app.asset_dir(t,OUT)/'attempts').glob('*/candidate.html'):
                rel=p.relative_to(OUT).as_posix();links+=f' · <a href="{rel}">재인식 후보 {p.parent.name}</a>'
        rows.append(f'<tr><td>{html.escape(t["region"]["id"])}</td><td>{r["provenance"] if r else "미완료·검수 필요"}</td><td>{links}</td></tr>')
        review='<meta charset="utf-8"><style>body{font:14px/1.6 system-ui;margin:20px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #777;padding:5px;vertical-align:top;overflow-wrap:anywhere}summary{cursor:pointer;padding:8px;background:#eee}</style>'
        review+=f'<h2>{html.escape(t["region"]["id"])}</h2><p>{html.escape(decisions.get(key,{}).get("notes","셀 내용·순서·병합 값 보존 검사 통과. 전체 의미 정확도 수동 검수는 하지 않았습니다."))}</p><div class="pair"><div><h3>원본 표 영역</h3><img src="../{source}/image.png"></div><div>'
        if r: review+='<h3>채택한 HTML</h3>'+table_fragment(r['table_content'])
        else: review+='<h3>미완료 — 아래 후보는 채택하지 않았습니다</h3>'
        for p in sorted((app.asset_dir(t,OUT)/'attempts').glob('*/candidate.html')):
            review+=f'<details><summary>{html.escape(p.parent.name)} 후보</summary>'+table_fragment(p.read_text(encoding='utf-8'))+'</details>'
        review+='</div></div>'
        app.atomic_bytes(OUT/'reviews'/f'{key}.html',review.encode('utf-8'))
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PM209 실패 표 보완 결과</title>
<style>body{font:15px/1.6 system-ui;margin:24px}td,th{border:1px solid #bbb;padding:8px}table{border-collapse:collapse}a{white-space:nowrap}</style>
<h1>PM209 실패 표 232개 보완 결과</h1><p>기존 성공 2,088개는 변경하지 않았습니다. HTML 형식 복구와 인식 정확도는 다릅니다. 전체 의미 정확도 검수가 완료된 자료가 아닙니다.</p>'''
    doc+=f'<p>복구 {len(restored)}개 / 미완료 {len(unresolved)}개 · <a href="../{OLD.name}/review_index.html">기존 성공 2,088개 검수 목록</a></p><table><tr><th>ID</th><th>처리</th><th>파일</th></tr>'+''.join(rows)+'</table></html>'
    app.atomic_bytes(OUT/'review_index.html',doc.encode('utf-8'))
    print(json.dumps({k:v for k,v in validation.items() if k not in ('protected_sha256','unresolved')},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('repair','retry','retry-review','retry-e2e','accept','finalize'))
    args=parser.parse_args(); RT.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    lock=(RT/'runner.lock').open('a+b')
    if lock.seek(0,2)==0: lock.write(b'0');lock.flush()
    lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    try:
        data=failures();ensure_manifest()
        {'repair':repair,'retry':retry,'retry-review':lambda d:retry(d,True),
         'retry-e2e':lambda d:retry(d,True,True),'accept':accept,'finalize':finalize}[args.stage](data)
    finally: lock.close()


if __name__=='__main__': main()
