"""Official Transformers BGE sequence-classification inference, FP16/batch 1."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'PM209/src'))
from reranking import MODEL, REVISION


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)


def main(args):
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    jobs=json.loads(args.input.read_text(encoding='utf-8'))
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.65,0)
    model_path=REPO/'.runtime/models/bge-reranker-v2-m3'
    marker=json.loads((model_path/'download-identity.json').read_text())
    if marker != {'model':MODEL,'revision':REVISION}:raise ValueError('Model revision marker differs')
    start=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(model_path,local_files_only=True,
        trust_remote_code=False,dtype=torch.float16,attn_implementation='sdpa').to('cuda:0').eval()
    identity={'model':MODEL,'revision':REVISION,'dtype':'float16','batch_size':1,'device':'cuda:0',
              'max_input_tokens':2048,'truncation':False,'backend':'Transformers AutoModelForSequenceClassification',
              'torch':torch.__version__,'transformers':importlib.metadata.version('transformers')}
    cache_dir=REPO/'.runtime/pm209-bge/pair-cache'
    loaded=time.perf_counter()-start
    generated=cached=0; outputs=[]; max_tokens=0; gpu=[]
    for job in jobs:
        job_start=time.perf_counter();scores=[];lengths=[];pair_times=[]
        for passage in job['passages']:
            key=hashlib.sha256(json.dumps([identity,job['query'],passage['text']],sort_keys=True).encode()).hexdigest()
            path=cache_dir/(key+'.json')
            if path.exists():
                record=json.loads(path.read_text());cached+=1
            else:
                inputs=tokenizer(job['query'],passage['text'],return_tensors='pt',truncation=False)
                n=inputs['input_ids'].shape[1]
                if n>2048:raise ValueError(f'Input length {n} exceeds budget; nothing truncated')
                inputs={k:v.to('cuda:0') for k,v in inputs.items()}
                torch.cuda.synchronize();begin=time.perf_counter()
                with torch.inference_mode():score=float(model(**inputs).logits.float().reshape(-1)[0].item())
                torch.cuda.synchronize()
                if not torch.isfinite(torch.tensor(score)):raise ValueError('Nonfinite BGE output')
                record={'score':score,'tokens':n,'forward_seconds':time.perf_counter()-begin,'key':key}
                save(path,record);generated+=1
            scores.append(record['score']);lengths.append(record['tokens']);pair_times.append(record['forward_seconds'])
            max_tokens=max(max_tokens,record['tokens'])
        outputs.append({'id':job['id'],'passage_ids':[p['id'] for p in job['passages']], 'scores':scores,
                        'token_counts':lengths,'pair_forward_seconds':pair_times,'elapsed_seconds':time.perf_counter()-job_start})
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True,
                                         creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            mem,temp=map(int,raw.strip().split(',')[:2]);gpu.append({'used_mib':mem,'temperature_c':temp})
        except Exception:pass
        print(f'BGE {len(outputs)}/{len(jobs)} pairs generated={generated} cached={cached}',flush=True)
        save(args.output,{'identity':identity,'jobs':outputs,'metrics':{
            'model_load_seconds':loaded,'elapsed_seconds':time.perf_counter()-start,'generated_pairs':generated,
            'cache_hits':cached,'max_input_tokens':max_tokens,'truncated_pairs':0,
            'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'cuda_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'gpu_observations':gpu,'gpu_name':torch.cuda.get_device_name(0)}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
