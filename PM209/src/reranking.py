"""BGE text reranking in an isolated CUDA worker; no vector/index mutation."""
import copy
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'


def rank_hits(hits, scores, top_k):
    if len(hits) != len(scores) or not 1 <= top_k <= len(hits):
        raise ValueError('Candidate/score count or top_k invalid')
    if len({h['chunk_id'] for h in hits}) != len(hits):
        raise ValueError('Duplicate candidate IDs')
    if not all(math.isfinite(s) for s in scores):
        raise ValueError('Nonfinite reranker score')
    order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
    ranked = []
    for position, i in enumerate(order, 1):
        hit = copy.deepcopy(hits[i])
        hit.update(rerank_score=scores[i], rerank_original_rank=i+1, rerank_rank=position)
        ranked.append(hit)
    return ranked[:top_k], ranked


def unload_project_models(url):
    try:
        with urllib.request.urlopen(url + '/api/ps', timeout=10) as r:
            loaded = json.load(r)['models']
    except OSError:
        return
    for model in loaded:
        if model['name'] in ('qwen3.5:4b', 'qwen3-embedding:0.6b'):
            payload = json.dumps({'model': model['name'], 'keep_alive': 0}).encode()
            request = urllib.request.Request(url+'/api/generate', data=payload, headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request, timeout=60) as response:
                response.read()


class BGEReranker:
    def __init__(self, url='http://127.0.0.1:11434'):
        self.url = url
        self.python = REPO / '.runtime/venv-reranker/Scripts/python.exe'

    def rerank(self, query, hits, top_k=5):
        if not hits:
            return [], {'model': MODEL, 'candidates': 0}
        if not self.python.exists():
            raise RuntimeError('Prepare .runtime/venv-reranker; see PM209/RERANKING.md')
        unload_project_models(self.url)
        runtime = REPO / '.runtime/pm209-bge'
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='query-', dir=runtime) as directory:
            input_path, output_path = (Path(directory)/n for n in ('input.json','output.json'))
            input_path.write_text(json.dumps([{'id':'query','query':query,
                'passages':[{'id':h['chunk_id'],'text':h['text']} for h in hits]}], ensure_ascii=False), encoding='utf-8')
            started = time.perf_counter()
            process = subprocess.run([str(self.python), '-X', 'utf8', '-B', str(REPO/'PM209/scripts/bge_worker.py'),
                                      '--input',str(input_path),'--output',str(output_path)],
                                     capture_output=True,text=True,encoding='utf-8',timeout=1200,
                                     creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if process.returncode:
                raise RuntimeError('BGE worker failed: '+process.stderr[-2500:])
            output = json.loads(output_path.read_text(encoding='utf-8'))
            item = output['jobs'][0]
            if item['passage_ids'] != [h['chunk_id'] for h in hits]:
                raise ValueError('Reranker output order differs')
            selected, ranked = rank_hits(hits,item['scores'],min(top_k,len(hits)))
            return selected, {**output['identity'], 'candidates':len(hits), 'top_k':len(selected),
                'elapsed_seconds':time.perf_counter()-started, 'worker':output['metrics'],
                'ranking':[{'chunk_id':h['chunk_id'],'score':h['rerank_score'],
                            'original_rank':h['rerank_original_rank']} for h in ranked]}
