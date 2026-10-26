"""Separate chunk / embed / index / search / evaluate stages for PM209.

Default scope is three train manuals. Source data is opened read-only. QA is
read only in evaluate(), after chunks and vectors have already been saved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import httpx
import numpy as np
from qdrant_client import QdrantClient, models
from tokenizers import Tokenizer

PM = Path(__file__).resolve().parents[1]
REPO = PM.parent
sys.path.insert(0, str(PM / "src"))
from chunking import VERSION, FIGURES, chunk_page, manual_id, normalize, ocr_text

DATA = PM / "datasets" / "PM209"
DEFAULT_INPUT = DATA / "data" / "paddle_tables_manual_20261026"
DEFAULT_OUTPUT = DATA / "rag" / "sample_char500_v1"
MODEL = "qwen3-embedding:0.6b"
COLLECTION = "pm209_chunks"
SAMPLE_MANUALS = [
    "bose_headphones_manuals_768a0fac-5355-4e11-a6e6-8aa105222381",
    "HUAWEI_Sound Joy User Guide-(EGRT-09,02,en)",
    "alienware-17-r5_users-guide_en-us",
]
INSTRUCTION = "Given a question about a product manual, retrieve passages that answer the question."
EMBED_OPTIONS = {"num_ctx": 2048, "num_batch": 128, "num_thread": 4, "num_gpu": 99}


def progress(args, stage, **values):
    callback = getattr(args, 'progress', None)
    if callback:
        callback(stage, **values)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_lines(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_lines(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def gpu_snapshot():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        fields = [v.strip() for v in result.stdout.splitlines()[0].split(",")]
        return dict(name=fields[0], total_mib=int(fields[1]), used_mib=int(fields[2]),
                    temperature_c=int(fields[3]), utilization_percent=int(fields[4]))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def input_identity(args, manuals):
    return {"chunking_version": VERSION, "manuals": sorted(manuals), "splits": sorted(args.split),
            "source_files": {str((args.input / (s+".jsonl")).resolve()): sha(args.input / (s+".jsonl")) for s in args.split},
            "manual_reviews_sha256": sha(args.input / "manual-review.json"),
            "metadata_sha256": sha(DATA / "meta_data/name2meta.json")}


def check_sources(manifest):
    for name, expected in manifest["identity"]["source_files"].items():
        if sha(name) != expected:
            raise ValueError(f"Source changed since chunking: {name}; create a new output directory")


def tokenizer(args):
    tok = Tokenizer.from_file(str(args.tokenizer))
    tok.no_truncation()
    tok.no_padding()
    return tok


def build_chunks(args):
    started = time.perf_counter()
    progress(args, 'chunk', completed=0)
    metadata = read_json(DATA / "meta_data/name2meta.json")
    manuals = list(metadata) if args.all else args.manual or SAMPLE_MANUALS
    identity = input_identity(args, manuals)
    manifest_path = args.output / "chunks-manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
        raise ValueError("Output belongs to a different source/scope/version. Use a new --output directory.")
    manual_reviews = {e["table_id"] for e in read_json(args.input / "manual-review.json")["entries"]}
    tok = tokenizer(args)
    chunks, pages, empty, missing, seen = [], [], [], [], set()
    for split in args.split:
        for page in read_lines(args.input / (split+".jsonl")):
            mid = manual_id(page, metadata)
            if mid not in manuals:
                continue
            seen.add(mid)
            # Explicit whitelist: QA never crosses the chunking boundary.
            safe_page = {k: page[k] for k in ("id", "image_filename", "bounding_boxes")}
            result = chunk_page(safe_page, split, metadata, manual_reviews)
            represented = {rid for c in result for field in ("content_region_ids", "context_region_ids", "deduplicated_region_ids") for rid in c.get(field, [])}
            for r in safe_page["bounding_boxes"]:
                value = ocr_text(r) if r["structure"] in {"Text", "Title"} else r.get("table_content", "") if r["structure"] == "Table" else r.get("description", "")
                if not value.strip():
                    empty.append({"page_id": page["id"], "region_id": r["id"], "type": r["structure"]})
                elif r["id"] not in represented:
                    missing.append({"page_id": page["id"], "region_id": r["id"]})
            for c in result:
                c["token_count"] = len(tok.encode(c["text"], add_special_tokens=False).ids)
            chunks.extend(result)
            pages.append({"page_id": page["id"], "image_filename": page["image_filename"], "manual_id": mid})
            if len(pages) % 100 == 0:
                progress(args, 'chunk', completed=len(pages), chunks=len(chunks))
    if seen != set(manuals):
        raise ValueError(f"Manuals absent from selected splits: {set(manuals)-seen}")
    if missing:
        write_json(args.output / "chunk-errors.json", missing)
        raise ValueError(f"{len(missing)} nonempty regions lost during chunking")
    if len({c['chunk_id'] for c in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    dump_lines(args.output / "chunks.jsonl", chunks)
    report = {"identity": identity, "pages": len(pages), "chunks": len(chunks),
              "per_manual": dict(Counter(c['manual_id'] for c in chunks)),
              "types": dict(Counter(c['chunk_type'] for c in chunks)),
              "max_char_count": max(c['char_count'] for c in chunks),
              "max_token_count": max(c['token_count'] for c in chunks),
              "over_500": sum(c['char_count'] > 500 for c in chunks),
              "overlap_chunks": sum(c['overlap_char_count'] > 0 for c in chunks),
              "quality_flags": dict(Counter(f for c in chunks for f in c['quality_flags'])),
              "empty_source_regions": empty, "missing_nonempty_regions": missing,
              "chunks_sha256": sha(args.output / "chunks.jsonl"),
              "tokenizer_sha256": sha(args.tokenizer), "elapsed_seconds": time.perf_counter()-started}
    write_json(manifest_path, report)
    check_sources(report)
    make_review(args.output, pages, chunks)
    progress(args, 'chunk', completed=len(pages), chunks=len(chunks), stage_complete=True)
    print(json.dumps({k: v for k, v in report.items() if k not in {"identity", "empty_source_regions"}}, ensure_ascii=False), flush=True)


def make_review(output, pages, chunks):
    output.mkdir(parents=True, exist_ok=True)
    if len(pages) > 200:
        by_manual, manual_chunks = defaultdict(list), defaultdict(list)
        for page in pages:
            by_manual[page['manual_id']].append(page)
        for chunk in chunks:
            manual_chunks[chunk['manual_id']].append(chunk)
        links = []
        for manual, manual_pages in by_manual.items():
            slug = hashlib.sha256(manual.encode()).hexdigest()[:16]
            target = output / 'manual-reviews' / slug
            # Large manuals are split into bounded review pages as well.
            for offset in range(0, len(manual_pages), 100):
                subset = manual_pages[offset:offset+100]
                page_ids = {p['page_id'] for p in subset}
                subdir = target / str(offset // 100 + 1)
                make_review(subdir, subset, [c for c in manual_chunks[manual] if c['page_id'] in page_ids])
                link = (subdir / 'review.html').relative_to(output).as_posix()
                links.append(f"<li><a href='{link}'>{html.escape(manual)} · {offset+1}–{offset+len(subset)}</a></li>")
        (output / 'review.html').write_text("<!doctype html><meta charset='utf-8'><h1>PM209 전체 청크 검수</h1><ul>"+'\n'.join(links)+'</ul>', encoding='utf-8')
        return
    groups = defaultdict(list)
    for c in chunks:
        groups[c['page_id']].append(c)
    esc = html.escape
    fragments = ["<!doctype html><meta charset='utf-8'><title>PM209 청크 검수</title><style>body{font:16px system-ui;margin:24px;background:#f4f5f7}summary{cursor:pointer;font-weight:bold;padding:12px}.page{display:grid;grid-template-columns:minmax(300px,45%) 1fr;gap:20px}img{width:100%;position:sticky;top:0}.chunk{background:white;padding:16px;margin:8px 0;border:1px solid #ddd}pre{white-space:pre-wrap;word-break:break-word;font:15px/1.6 system-ui}small{overflow-wrap:anywhere;color:#555}.flag{color:#965500}</style><h1>PM209 · 500자 청크 검수</h1><p>왼쪽 원본 페이지, 오른쪽 검색용 청크. 제목·좌표 연결은 기하학적 휴리스틱입니다. QA는 포함하지 않았습니다.</p>"]
    for p in pages:
        source = (DATA / p['image_filename']).resolve().as_uri()
        fragments.append(f"<details><summary>{esc(p['page_id'])} · {len(groups[p['page_id']])} chunks</summary><div class='page'><a href='{esc(source)}'><img loading='lazy' src='{esc(source)}'></a><div>")
        for c in groups[p['page_id']]:
            fragments.append(f"<article class='chunk'><b>{c['chunk_type']} · {c['char_count']}자 / {c['token_count']}토큰 · 겹침 {c['overlap_char_count']}자</b><pre>{esc(c['text'])}</pre><small>{esc(', '.join(c['content_region_ids']))}</small><p class='flag'>{esc(', '.join(c['quality_flags']))}</p></article>")
        fragments.append("</div></div></details>")
    (output / "review.html").write_text("\n".join(fragments), encoding="utf-8")


class Embedder:
    def __init__(self, args):
        self.args = args
        self.client = httpx.Client(base_url=args.url, timeout=180, trust_env=False)
        tags = self.client.get("/api/tags"); tags.raise_for_status()
        found = next((m for m in tags.json()['models'] if m['name'] == args.model), None)
        if not found:
            raise ValueError(f"Pull the embedding model first: ollama pull {args.model}")
        show = self.client.post("/api/show", json={"model": args.model}); show.raise_for_status()
        if "embedding" not in show.json().get("capabilities", []):
            raise ValueError("Selected model does not support embeddings")
        self.identity = {"model": args.model, "digest": found['digest'], "options": EMBED_OPTIONS,
                         "query_instruction": INSTRUCTION, "truncate": False}
        self.db = sqlite3.connect(args.output / "embedding-cache.sqlite3")
        self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dimension INTEGER, vector BLOB NOT NULL)")
        self.generated, self.reused, self.prompt_tokens = 0, 0, 0
        self.gpus = []

    def key(self, text, query=False):
        payload = [self.identity, "query" if query else "document", text]
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def cached(self, text, query=False):
        row = self.db.execute("SELECT dimension,vector FROM vectors WHERE key=?", (self.key(text, query),)).fetchone()
        if not row:
            return None
        v = np.frombuffer(row[1], dtype=np.float32)
        if len(v) != row[0] or not np.isfinite(v).all() or not np.linalg.norm(v) > 0:
            raise ValueError("Invalid cached embedding")
        return v.tolist()

    def encode(self, texts, query=False):
        result = [self.cached(t, query) for t in texts]
        self.reused += sum(v is not None for v in result)
        absent = [i for i, v in enumerate(result) if v is None]
        if absent:
            inputs = [f"Instruct: {INSTRUCTION}\nQuery: {texts[i]}" if query else texts[i] for i in absent]
            response = self.client.post("/api/embed", json={"model": self.args.model, "input": inputs,
                "truncate": False, "keep_alive": "2m", "options": EMBED_OPTIONS})
            response.raise_for_status()
            data = response.json()
            if len(data['embeddings']) != len(absent):
                raise ValueError("Embedding count mismatch")
            self.prompt_tokens += data.get('prompt_eval_count', 0)
            for i, vector in zip(absent, data['embeddings']):
                arr = np.asarray(vector, dtype=np.float32)
                if arr.ndim != 1 or not len(arr) or not np.isfinite(arr).all() or np.linalg.norm(arr) == 0:
                    raise ValueError("Invalid model embedding")
                self.db.execute("INSERT OR REPLACE INTO vectors VALUES (?,?,?)", (self.key(texts[i], query), len(arr), arr.tobytes()))
                result[i] = arr.tolist()
                self.generated += 1
            self.db.commit()
            snapshot = gpu_snapshot()
            if snapshot:
                self.gpus.append(snapshot)
        return result

    def close(self):
        self.db.close()
        self.client.close()


def checked_chunks(args):
    manifest = read_json(args.output / 'chunks-manifest.json')
    if sha(args.output / 'chunks.jsonl') != manifest['chunks_sha256']:
        raise ValueError("Chunk file hash mismatch")
    return manifest, list(read_lines(args.output / 'chunks.jsonl'))


def embed(args):
    started = time.perf_counter()
    manifest, chunks = checked_chunks(args)
    model = Embedder(args)
    failures = []
    progress(args, 'embed', completed=0, total=len(chunks), failed=0)
    try:
        for offset in range(0, len(chunks), args.batch_size):
            batch = chunks[offset:offset+args.batch_size]
            try:
                if any(c['token_count'] > 2000 for c in batch):
                    raise ValueError("Chunk exceeds 2000-token guard; inspect before embedding")
                model.encode([c['text'] for c in batch])
            except (httpx.HTTPError, ValueError) as exc:
                # Isolate failed chunks; successful cache entries survive interruption.
                for c in batch:
                    try:
                        if c['token_count'] > 2000:
                            raise ValueError("Chunk exceeds 2000-token guard")
                        model.encode([c['text']])
                    except (httpx.HTTPError, ValueError) as individual:
                        failures.append({"chunk_id": c['chunk_id'], "error": str(individual)})
            progress(args, 'embed', completed=min(offset+len(batch),len(chunks))-len(failures),
                     processed=min(offset+len(batch),len(chunks)), total=len(chunks), failed=len(failures),
                     generated_this_run=model.generated, cache_hits_this_run=model.reused,
                     gpu=model.gpus[-1] if model.gpus else None)
            if offset % (args.batch_size*20) == 0:
                print(f"Embeddings {min(offset+len(batch),len(chunks))}/{len(chunks)}; generated={model.generated} cached={model.reused}", flush=True)
        dimensions = {len(model.cached(c['text'])) for c in chunks if model.cached(c['text']) is not None}
        if len(dimensions) != 1:
            raise ValueError(f"Inconsistent dimensions: {dimensions}")
        ps = model.client.get('/api/ps'); ps.raise_for_status()
        report = {"model_identity": model.identity, "chunks_sha256": manifest['chunks_sha256'],
                  "dimension": next(iter(dimensions)), "chunks": len(chunks),
                  "generated_this_run": model.generated, "cache_hits_this_run": model.reused,
                  "failures": failures, "elapsed_seconds": time.perf_counter()-started,
                  "batch_size": args.batch_size, "prompt_tokens_this_run": model.prompt_tokens,
                  "gpu_samples": model.gpus, "ollama_ps": ps.json()}
        write_json(args.output / 'embedding-report.json', report)
        write_json(args.output / 'runs' / (time.strftime('%Y%m%d-%H%M%S')+'-embedding.json'), report)
        print(json.dumps({k:v for k,v in report.items() if k not in {'gpu_samples','ollama_ps'}}, ensure_ascii=False), flush=True)
        if failures:
            raise ValueError(f"{len(failures)} embeddings failed; rerun embed to resume")
    finally:
        model.close()


def index(args):
    started = time.perf_counter()
    manifest, chunks = checked_chunks(args)
    report = read_json(args.output / 'embedding-report.json')
    if report['chunks_sha256'] != manifest['chunks_sha256'] or report['failures']:
        raise ValueError("Embeddings are incomplete or from different chunks")
    identity = {"chunks_sha256": manifest['chunks_sha256'], "model_identity": report['model_identity']}
    path = args.output / 'index-identity.json'
    if path.exists() and read_json(path) != identity:
        raise ValueError("Existing index has different source/model. Choose a new output directory.")
    write_json(path, identity)
    model = Embedder(args)
    if model.identity != report['model_identity']:
        model.close()
        raise ValueError("Embedding model tag/digest changed; use a new output directory")
    db = QdrantClient(path=str(args.output / 'qdrant'))
    progress(args, 'index', completed=0, total=len(chunks))
    try:
        if not db.collection_exists(COLLECTION):
            db.create_collection(COLLECTION, vectors_config=models.VectorParams(size=report['dimension'], distance=models.Distance.COSINE))
        for offset in range(0, len(chunks), 64):
            points = []
            for c in chunks[offset:offset+64]:
                v = model.cached(c['text'])
                if v is None:
                    raise ValueError(f"Missing cached vector: {c['chunk_id']}")
                points.append(models.PointStruct(id=c['chunk_id'], vector=v, payload=c))
            db.upsert(COLLECTION, points=points, wait=True)
            progress(args, 'index', completed=min(offset+len(points),len(chunks)), total=len(chunks))
        count = db.count(COLLECTION, exact=True).count
        if count != len(chunks):
            raise ValueError(f"Point count mismatch: {count} != {len(chunks)}")
        result = {"points": count, "dimension": report['dimension'], "distance": "Cosine",
                  "storage": str(args.output / 'qdrant'), "elapsed_seconds": time.perf_counter()-started}
        write_json(args.output / 'index-report.json', result)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        db.close(); model.close()


def open_search(args):
    manifest, chunks = checked_chunks(args)
    report = read_json(args.output / 'embedding-report.json')
    model = Embedder(args)
    identity = read_json(args.output / 'index-identity.json')
    if identity != {"chunks_sha256": manifest['chunks_sha256'], "model_identity": model.identity}:
        model.close()
        raise ValueError("Index/source/model identity mismatch")
    db = QdrantClient(path=str(args.output / 'qdrant'))
    if db.count(COLLECTION, exact=True).count != len(chunks):
        db.close(); model.close()
        raise ValueError("Index is incomplete; run index first")
    return manifest, chunks, model, db


def retrieve(db, model, manual, question, limit):
    vector = model.encode([question], query=True)[0]
    results = db.query_points(COLLECTION, query=vector,
        query_filter=models.Filter(must=[models.FieldCondition(key='manual_id', match=models.MatchValue(value=manual))]), limit=limit).points
    return [{"score": p.score, **p.payload} for p in results]


def search(args):
    manifest, chunks, model, db = open_search(args)
    try:
        manuals = sorted({c['manual_id'] for c in chunks})
        if not args.manual or len(args.manual) != 1 or args.manual[0] not in manuals:
            raise ValueError(f"Choose one --manual from {manuals}")
        if not args.query:
            raise ValueError("--query is required")
        hits = retrieve(db, model, args.manual[0], args.query, args.top_k)
        write_json(args.output / 'last-search.json', {'query': args.query, 'manual': args.manual[0], 'hits': hits})
        for h in hits:
            print(f"\n{h['score']:.4f} | {h['chunk_type']} | {h['page_id']}\n{h['text']}")
    finally:
        db.close(); model.close()


def evaluate(args):
    started = time.perf_counter()
    manifest, chunks, model, db = open_search(args)
    try:
        check_sources(manifest)
        region_pages = defaultdict(set)
        questions, manuals = [], set(manifest['identity']['manuals'])
        metadata = read_json(DATA / 'meta_data/name2meta.json')
        for filename in manifest['identity']['source_files']:
            for p in read_lines(filename):
                mid = manual_id(p, metadata)
                if mid not in manuals:
                    continue
                for r in p['bounding_boxes']:
                    region_pages[r['id']].add(p['id'])
                for q in p.get('qa_data', []):
                    # Gold answers are unnecessary even for this retrieval evaluation.
                    questions.append({'id': q['id'], 'manual': mid, 'question': q['question']['text'],
                                      'relevant': q.get('answer', {}).get('relevant', [])})
        counts = Counter(c['manual_id'] for c in chunks)
        results, skipped = [], []
        for i, q in enumerate(questions):
            if not q['relevant'] or any(rid not in region_pages for rid in q['relevant']):
                skipped.append(q['id']); continue
            gold = set().union(*(region_pages[rid] for rid in q['relevant']))
            hits = retrieve(db, model, q['manual'], q['question'], counts[q['manual']])
            pages = list(dict.fromkeys(h['page_id'] for h in hits))
            row = {'id': q['id'], 'manual': q['manual'], 'question': q['question'],
                   'gold_pages': sorted(gold), 'top_pages': pages[:5],
                   'top_chunks': [h['chunk_id'] for h in hits[:5]]}
            for k in (1, 3, 5):
                row[f'page_recall_at_{k}'] = len(gold & set(pages[:k])) / len(gold)
                row[f'page_hit_at_{k}'] = bool(gold & set(pages[:k]))
            ids = {rid for h in hits[:5] for rid in h['content_region_ids']}
            row['direct_region_recall_at_5'] = len(ids & set(q['relevant'])) / len(set(q['relevant']))
            row['direct_region_precision_at_5'] = len(ids & set(q['relevant'])) / len(ids) if ids else 0
            results.append(row)
            if i % 20 == 0:
                print(f"Retrieval evaluation {i+1}/{len(questions)}", flush=True)
        keys = [f'page_{metric}_at_{k}' for metric in ('recall', 'hit') for k in (1,3,5)] + ['direct_region_recall_at_5','direct_region_precision_at_5']
        metrics = {k: sum(float(r[k]) for r in results)/len(results) for k in keys} if results else {}
        report = {'evaluated': len(results), 'skipped_unresolved_gold': skipped, 'metrics': metrics,
                  'scope': f'{len(manuals)} selected manuals, splits={manifest["identity"]["splits"]}; development retrieval test, not a generalization estimate',
                  'page_aggregation': 'max chunk similarity, over every chunk of the selected manual',
                  'region_metric_caveat': 'Direct region IDs only; partial-region matches still count. Context and dedup aliases excluded.',
                  'elapsed_seconds': time.perf_counter()-started, 'results': results}
        write_json(args.output / 'retrieval-evaluation.json', report)
        print(json.dumps({k:v for k,v in report.items() if k != 'results'}, ensure_ascii=False))
    finally:
        db.close(); model.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['chunk', 'embed', 'index', 'search', 'evaluate', 'sample'])
    p.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--split', nargs='+', choices=['train','val','test'], default=['train'])
    scope = p.add_mutually_exclusive_group()
    scope.add_argument('--manual', action='append')
    scope.add_argument('--all', action='store_true', help='Explicitly opt in to all manuals; use a separate --output')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--url', default='http://127.0.0.1:11434')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--tokenizer', type=Path, default=REPO / '.runtime/pm209-token-length/tokenizer.json')
    p.add_argument('--query')
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()
    if not 1 <= args.batch_size <= 8:
        p.error('Use batch size 1..8 for this 8GB configuration')
    if args.all and args.output.resolve() == DEFAULT_OUTPUT.resolve():
        p.error('--all requires a separate --output; sample results are preserved')
    if args.stage == 'sample' and (args.all or args.split != ['train']):
        p.error('sample stage is train-only and does not allow --all')
    args.output.mkdir(parents=True, exist_ok=True)
    stages = {'chunk': build_chunks, 'embed': embed, 'index': index, 'search': search, 'evaluate': evaluate}
    if args.stage == 'sample':
        for stage in ('chunk', 'embed', 'index', 'evaluate'):
            stages[stage](args)
    else:
        stages[args.stage](args)


if __name__ == '__main__':
    main()
