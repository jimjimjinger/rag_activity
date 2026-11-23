"""Download one pinned official BGE revision and verify the weights SHA256."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reranking import MODEL, REVISION, REPO


def main():
    from huggingface_hub import snapshot_download
    destination=REPO/'.runtime/models/bge-reranker-v2-m3'
    snapshot_download(MODEL,revision=REVISION,local_dir=destination,
        allow_patterns=['*.json','*.safetensors','sentencepiece.bpe.model'],max_workers=2)
    digest=hashlib.sha256()
    with (destination/'model.safetensors').open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    if digest.hexdigest()!='d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286':
        raise ValueError('BGE weights differ from the pinned official LFS SHA256')
    (destination/'download-identity.json').write_text(
        json.dumps({'model':MODEL,'revision':REVISION},indent=2)+'\n',encoding='utf-8')
    print('Pinned model verified:',destination)


if __name__=='__main__':main()
