"""Tightness of shared-subspace block bounds on GloVe-100 (100k): pruned share vs oracle."""
import numpy as np
from hyperhull_ivf.real_data import load_glove
from hyperhull_ivf.ivf import build_ivf
from hyperhull_ivf.hulls import subdivide_lists, build_hulls, build_shared_block_hulls, project_to_list, shared_block_lower_bound_sq

d = load_glove(n_queries=100, n_base=100_000)
ivf0 = build_ivf(d.base, 256, 100_000)
for block_size in (8, 16, 32):
    ivf, layout = subdivide_lists(ivf0, block_size)
    sizes = np.diff(layout.block_offsets)
    for r in (16, 32, 64):
        lh = build_hulls(ivf.vectors, ivf.offsets, r)
        bh = build_shared_block_hulls(ivf, layout, lh)
        oracle = pruned = 0.0
        for q in d.queries:
            dd = ((ivf.vectors - q) ** 2).sum(1)
            kth = np.partition(dd, 9)[9]
            true_min = np.minimum.reduceat(dd, layout.block_offsets[:-1])
            lb = np.concatenate([
                shared_block_lower_bound_sq(*project_to_list(q, lh, l), bh, slice(layout.list_blocks[l], layout.list_blocks[l + 1]))
                for l in range(ivf.n_lists)])
            oracle += sizes[true_min > kth].sum(); pruned += sizes[lb > kth].sum()
        n = len(d.queries) * len(ivf.vectors)
        print(f"block={block_size:3d} R={r:3d}  oracle={oracle/n:.3f}  pruned={pruned/n:.3f}  storage/data={(3*r+3)/(block_size*100):.2f}", flush=True)
