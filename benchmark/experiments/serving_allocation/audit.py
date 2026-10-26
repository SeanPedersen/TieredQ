"""Freeze source/compiler/layout provenance for the isolated serving study."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root=Path.cwd()
results=root/'results'
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):
    (results/name).write_text(json.dumps(value,indent=2)+'\n')
audits={}
for build in ('build','build-wide','build-profile'):
    commands=json.loads((root/build/'compile_commands.json').read_text())
    records=[]
    for row in commands:
        if '/src/' not in row['file'] and not row['file'].endswith('/native/module.cpp'):
            continue
        tokens=shlex.split(row['command'])
        optimization=[x for x in tokens if x.startswith('-O')]
        valid=optimization and optimization[-1]=='-O3' and '-march=native' in tokens and '-ffp-contract=off' in tokens
        assert valid, row
        records.append(dict(file=row['file'],command=row['command'],effective_optimization=optimization[-1]))
    assert records
    binary=next((root/build).glob('tiny_native*.so'))
    audits[build]=dict(commands=records,binary_sha256=digest(binary),binary_bytes=binary.stat().st_size)
write('compiler-audit.json',audits)
provenance={}
for variant,folder in (('primary',root),('byte-wide',root/'wide-source'),('diagnostic',root/'profile-source')):
    provenance[variant]={str(p.relative_to(folder)):digest(p) for name in ('src','include','cmake')
                         for p in (folder/name).rglob('*') if p.is_file()}
write('provenance.json',provenance)
write('cpu.json',json.loads(subprocess.check_output(['lscpu','-J'])))
layouts=[]
for name in ('allocation-quality.jsonl','cross-quality.jsonl'):
    for line in (results/name).read_text().splitlines():
        row=json.loads(line)
        n=1_000_000 if name.startswith('allocation') else 100_000
        tiers=row['tiers']
        narrow=all(t[2]<=4 for t in tiers)
        groups=sum((t[1]+4//t[2]-1)//(4//t[2]) for t in tiers) if narrow else sum(t[1] for t in tiers)
        bits=sum(t[1]*t[2] for t in tiers)
        block=256 if narrow else 16
        padded=(n+block-1)//block*block
        code_stride=(groups+1)//2 if narrow else 4*((bits+31)//32)
        register_bytes=16*2*((groups+1)//2) if narrow else sum(4*max(32,1<<t[2]) for t in tiers)
        byte_bytes=1024*((groups+1)//2) if narrow else 1024*len(tiers)
        layouts.append(dict(dataset=row.get('dataset','sift'),seed=row['seed'],policy=row['policy'],
                            budget_total_bits=bits,representation='narrow' if narrow else 'wide',
                            groups=groups,lookup_count=groups if not narrow else (groups+1)//2,
                            register_lookup_count=groups,byte_lookup_count=(groups+1)//2 if narrow else groups,
                            nominal_code_bytes=(bits+7)//8,stored_code_stride=code_stride,
                            packing_overhead_bytes=code_stride-(bits+7)//8,padded_documents=padded-n,
                            vector_bytes=(code_stride+9)*padded,measured_stride=row['stride'],
                            actual_bytes_per_vector=row['bytes_per_vector'],
                            register_table_bytes=register_bytes,byte_table_bytes=byte_bytes,
                            byte_total_table_bytes=byte_bytes+register_bytes if narrow else byte_bytes))
        assert code_stride+9==row['stride']
write('layouts.json',layouts)
print('Compiler, provenance and layout audits passed',flush=True)
