"""Cached GloVe-200 at 1M, retaining the original 100K base as a subset."""
from pathlib import Path
import numpy as np

BASE_ROWS = 1_000_000
DIMENSIONS = 200
QUERY_COUNT = 1000


def prepare_glove1m(root, max_base=None, query_count=QUERY_COUNT):
    from data import ground_truth
    n=min(max_base or BASE_ROWS,BASE_ROWS)
    if n<10 or not 1<=query_count<=QUERY_COUNT:
        raise ValueError('GloVe-1M needs at least 10 base rows and 1–1000 queries')
    path=Path(root)/'glove1m-1000000-1000-v1.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Prepared GloVe-1M cache required: {path}; run experiments/issue4_current/prepare_glove1m.py')
    with np.load(path) as arrays:
        base=arrays['base']
        queries=arrays['queries'][:query_count]
        truth=arrays['truth'][:query_count]
    if base.shape!=(BASE_ROWS,DIMENSIONS) or queries.shape!=(query_count,DIMENSIONS):
        raise ValueError('Unexpected GloVe-1M cache dimensions')
    if n<BASE_ROWS:
        base=base[:n].copy()
        truth=ground_truth(base,queries,False)
    return base,queries,truth
