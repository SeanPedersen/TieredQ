"""Verify cached full data against manifests and historical dataset identity."""
import hashlib,json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark"))
import numpy as np
from data import SOURCES,verify_checksum,BASE_COUNTS,DIMENSIONS
root=Path('data'); report={}
historical=Path('/home/sean/experiments/TieredQ-issue4-20261023/TieredQ/benchmark/output-full-single-run/data-manifest.json')
for name,n in BASE_COUNTS.items():
    path=root/f'{name}-{n}-1000-v2.npz'; manifest=json.loads(path.with_suffix('.json').read_text())
    assert manifest['source']==SOURCES[name]
    with np.load(path) as data:
        assert data['base'].shape==(n,DIMENSIONS[name]) and data['queries'].shape==(1000,DIMENSIONS[name]) and data['truth'].shape==(1000,10)
        hashes={key:hashlib.sha256(data[key].tobytes()).hexdigest() for key in ['base','queries','truth']}
        assert hashes['base']==manifest['base_sha256'] and hashes['queries']==manifest['queries_sha256']
        assert np.isfinite(data['base']).all() and np.isfinite(data['queries']).all()
        assert data['truth'].min()>=0 and data['truth'].max()<n
    if name!='openai': verify_checksum(root/f'{name}.hdf5',SOURCES[name]['sha256'])
    report[name]={'manifest':manifest,'array_sha256':hashes,'cache_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(f'Verified full {name}: {n} vectors, 1000 queries, cached truth',flush=True)
report['split_indices']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*indices.npy')}
p=root/'openai-test-positions.npy'; report['split_indices'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
Path('../data-verification.json').write_text(json.dumps(report,indent=2))
