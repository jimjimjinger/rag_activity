"""Pack/restore PM209 results as split, SHA256-verified transfer archives."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'PM209/results/20261022'
RUNTIME = ROOT / '.runtime/pm209-export'
CHUNK = 40 * 1024 * 1024
FOLDERS = (
    'PM209/datasets/PM209/data/vlm',
    'PM209/datasets/PM209/data/paddle_tables_html',
    'PM209/datasets/PM209/data/paddle_tables_html_repaired',
    'PM209/datasets/PM209/meta_data',
)
RECORDS = (
    '.runtime/pm209-paddle-tables/table-failures.jsonl',
    '.runtime/pm209-paddle-tables/final-validation.json',
    '.runtime/pm209-paddle-tables/input-inventory.json',
    '.runtime/pm209-paddle-tables/settings.json',
    '.runtime/pm209-paddle-repair/manual-decisions.json',
    '.runtime/pm209-paddle-repair/pending.json',
    '.runtime/pm209-paddle-repair/final-validation.json',
    '.runtime/pm209-paddle-repair/status.json',
    '.runtime/pm209-paddle-repair/gpu-status.json',
    '.runtime/pm209-paddle-repair/gpu-status-review.json',
    '.runtime/pm209-paddle-repair/gpu-status-e2e.json',
)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


class HashReader:
    def __init__(self, stream): self.stream, self.hash = stream, hashlib.sha256()
    def read(self, size):
        data = self.stream.read(size); self.hash.update(data); return data


class PartsReader(io.RawIOBase):
    def __init__(self, paths): self.paths, self.stream = iter(paths), None
    def readable(self): return True
    def readinto(self, buffer):
        while True:
            if self.stream is None:
                path = next(self.paths, None)
                if path is None: return 0
                self.stream = path.open('rb')
            n = self.stream.readinto(buffer)
            if n: return n
            self.stream.close(); self.stream = None
    def close(self):
        if self.stream is not None: self.stream.close()
        super().close()


def allowed(name):
    return name in RECORDS or any(name.startswith(folder + '/') for folder in FOLDERS)


def pack():
    if BUNDLE.exists(): raise FileExistsError('Bundle already exists; refusing to replace it')
    BUNDLE.mkdir(parents=True); RUNTIME.mkdir(parents=True, exist_ok=True)
    sources = sorted([p for folder in FOLDERS for p in (ROOT/folder).rglob('*') if p.is_file()]
                     + [ROOT/p for p in RECORDS])
    if not all(p.is_file() and not p.is_symlink() for p in sources): raise ValueError('Missing/linked input')
    archive = RUNTIME/'results.tar.gz'
    manifest = {'format': 'split-tar-gzip-sha256-v1', 'snapshot': '2026-10-22',
                'restore_base': 'repository root', 'files': [], 'parts': [],
                'figures_with_description': 23807, 'tables_saved': 2308, 'tables_unresolved': 12,
                'semantic_accuracy_verified': False}
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for index, path in enumerate(sources, 1):
            relative = path.relative_to(ROOT).as_posix()
            if not allowed(relative): raise ValueError('Unexpected input')
            info = tar.gettarinfo(str(path), arcname=relative)
            with path.open('rb') as source:
                reader = HashReader(source); tar.addfile(info, reader)
            manifest['files'].append({'path': relative, 'bytes': info.size, 'sha256': reader.hash.hexdigest()})
            if index % 2000 == 0: print(f'PACK {index}/{len(sources)}', flush=True)
    with archive.open('rb') as source:
        index = 1
        while data := source.read(CHUNK):
            name = f'results.tar.gz.part-{index:03d}'
            (BUNDLE/name).write_bytes(data)
            manifest['parts'].append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            index += 1
    manifest['archive_bytes'] = archive.stat().st_size
    manifest['archive_sha256'] = digest(archive)
    manifest['unpacked_bytes'] = sum(item['bytes'] for item in manifest['files'])
    (BUNDLE/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ('archive_bytes','unpacked_bytes','tables_saved','tables_unresolved')}))
    print(f'PACKED {len(sources)} files into {len(manifest["parts"])} parts', flush=True)


def read_manifest():
    manifest = json.loads((BUNDLE/'manifest.json').read_text(encoding='utf-8'))
    paths = []
    for part in manifest['parts']:
        if Path(part['name']).name != part['name']: raise ValueError('Invalid part name')
        path = BUNDLE/part['name']
        if path.stat().st_size != part['bytes'] or digest(path) != part['sha256']:
            raise ValueError(f'Archive part changed: {part["name"]}')
        paths.append(path)
    return manifest, paths


def unpack(verify_only):
    manifest, paths = read_manifest()
    expected = {item['path']:item for item in manifest['files']}
    if len(expected) != len(manifest['files']): raise ValueError('Duplicate manifest path')
    seen, archive_hash = set(), hashlib.sha256()
    for path in paths:
        with path.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''): archive_hash.update(block)
    if archive_hash.hexdigest()!=manifest['archive_sha256']: raise ValueError('Archive digest mismatch')
    with io.BufferedReader(PartsReader(paths)) as stream, tarfile.open(fileobj=stream,mode='r|gz') as tar:
        for member in tar:
            name = member.name; logical = PurePosixPath(name)
            if not member.isfile() or logical.is_absolute() or '..' in logical.parts or '\\' in name:
                raise ValueError('Unsafe archive entry')
            if name not in expected or name in seen or not allowed(name): raise ValueError('Unexpected archive member')
            info=expected[name]
            if member.size!=info['bytes']: raise ValueError('Member length mismatch')
            dest=(ROOT/name).resolve()
            if not dest.is_relative_to(ROOT.resolve()): raise ValueError('Entry escapes workspace')
            if dest.exists() and digest(dest)!=info['sha256']:
                raise ValueError(f'Existing result differs; preserved without overwrite: {name}')
            write = not verify_only and not dest.exists()
            if write: dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_name(dest.name+'.bundle-tmp')
            if write and tmp.exists(): raise FileExistsError(str(tmp))
            output=tmp.open('xb') if write else None
            h=hashlib.sha256()
            try:
                source=tar.extractfile(member)
                for data in iter(lambda:source.read(1024*1024),b''):
                    h.update(data)
                    if output: output.write(data)
                if output: output.flush(); os.fsync(output.fileno()); output.close();output=None
                if h.hexdigest()!=info['sha256']: raise ValueError('Member digest mismatch: '+name)
                if write: os.replace(tmp,dest)
            finally:
                if output: output.close()
            seen.add(name)
            if len(seen)%2000==0: print(f'VERIFY {len(seen)}/{len(expected)}',flush=True)
    if seen!=set(expected): raise ValueError('Archive members are missing')
    print(f'{"VERIFIED" if verify_only else "RESTORED"} {len(seen)} files; 23807 descriptions; 2308 tables; 12 unresolved',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('pack','verify','restore'))
    command=parser.parse_args().command
    pack() if command=='pack' else unpack(command=='verify')
