"""Load hash-verified archived arrays and build the measured-source exact-budget adapter."""
from pathlib import Path
import sys
import time
import json
import hashlib
from dataclasses import dataclass
ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT/'three-codec'
sys.path[:0] = [str(ROOT/'matched-storage/build'), str(ROOT/'source/benchmark')]
import numpy as np
import codec_native
from l2_transform import L2Transform
from measure import L2Search
from data import prepare
DATA = Path('/home/sean/experiments/TieredQ-issue4-current-20261026-tpj3ze/benchmark/data')
GEMINI = Path('/home/sean/experiments/TieredQ-gemini-turboquant-20261026-vDAm7R/TieredQ/benchmark/data/gemini-prepared.npz')


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(dataset):
    if dataset == 'gemini':
        with np.load(GEMINI) as data:
            arrays = tuple(data[k] for k in ('base','queries','truth'))
    else:
        arrays = prepare(DATA,dataset,None,1000)
    hashes = {k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in zip(('base','queries','truth'),arrays)}
    assert hashes == json.loads((ROOT/'output/inputs.json').read_text())[dataset]['sha256']
    save(HERE/f'inputs-{dataset}.json',hashes)
    np.save(HERE/f'truth-{dataset}.npy',arrays[2])
    return arrays


@dataclass
class Built:
    index: object
    build_s: float
    bytes_per_vector: float
    encode_vps: float


def build(base, budget, seed, directory, l2=False):
    codec_native.set_threads(8)
    tick=time.perf_counter()
    transform=L2Transform.fit(base) if l2 else None
    vectors=transform.base_vectors(base) if transform else base
    native=codec_native.Index(vectors,1,seed,True,l2,True,budget)
    elapsed=time.perf_counter()-tick
    size=native.storage(str(directory/'selection-index.bin'))
    if transform:
        size+=transform.storage_bytes
    index=L2Search(native,transform.scale,transform) if transform else native
    return Built(index,elapsed,size/len(base),len(base)/native.encode_seconds)
