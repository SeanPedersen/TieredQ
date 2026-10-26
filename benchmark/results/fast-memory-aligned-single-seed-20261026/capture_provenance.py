"""Record as-run sources, machine and pinned upstream/overlay identities."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import platform
import sys

ROOT=Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def command(args):
    return subprocess.check_output(args,text=True).strip()

sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src','include','tests','benchmark/native') for p in (ROOT/folder).rglob('*') if p.is_file()}
for p in (ROOT/'benchmark').glob('*.py'): sources[str(p.relative_to(ROOT))]=sha(p)
for name in ('CMakeLists.txt','benchmark/uv.lock','benchmark/pyproject.toml'):
    sources[name]=sha(ROOT/name)
verified={}; differences=[]
with tarfile.open(ROOT/'committed-source.tar') as archive:
    for name in sources:
        expected=hashlib.sha256(archive.extractfile(name).read()).hexdigest()
        verified[name]=expected==sources[name]
        if not verified[name]: differences.append(name)
assert differences==['benchmark/native/module.cpp'],differences
(ROOT/'source-sha256.json').write_text(json.dumps(sources,indent=2))
(ROOT/'source-verification.json').write_text(json.dumps({'committed_files_match':verified,'only_modified_source':differences},indent=2))
meta=json.loads((ROOT/'output/meta.json').read_text())
meta.update(head='06cda8574db318207fe577f55b4feca4c29a9995',compiler=command(['g++','--version']),uname=command(['uname','-a']),lscpu=command(['lscpu']),rabitq_commit=command(['git','-C',str(ROOT/'benchmark/.deps/RaBitQ-Library'),'rev-parse','HEAD']),rabitq_search={'container':'direct-flat','implicit_row_ids':True,'high_accuracy':True,'extra_bit_bounds':True},nanobind_NOMINSIZE='NOMINSIZE' in (ROOT/'benchmark/native/CMakeLists.txt').read_text(),overlay_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'benchmark/.deps').rglob('*.hpp') if 'rabitq-native' in str(p)})
meta['python_version']=platform.python_version()
meta['python_executable']=sys.executable
meta['rabitq_upstream_dirty']=command(['git','-C',str(ROOT/'benchmark/.deps/RaBitQ-Library'),'status','--porcelain'])
assert not meta['rabitq_upstream_dirty']
meta['archive_sha256']=sha(ROOT/'committed-source.tar')
meta['rabitq_patches']=json.loads((ROOT/'benchmark/.deps/rabitq-native/patches.json').read_text())
assert meta['rabitq_patches']=={'utils/rotator.hpp':'4dcd65c8ae0ade65f5de1e407a507a1453f3620e2627b5022698bffb0b7c24d9'}
meta['protocol']={'seed':0,'queries':1000,'threads':[1,8],'batches':[1,32],'warmup_queries':3,'timing_repeats':3,'mode':'fast','build_threads':8,'blas_threads':1,'profiling':False}
(ROOT/'output/meta.json').write_text(json.dumps(meta,indent=2))
(ROOT/'binary-sha256.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'benchmark/build-native').glob('*.so')},indent=2))
