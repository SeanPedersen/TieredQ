"""How tight must a sound bound be? Ratio true_min_dist / kth_dist for cells (non-squared distances),
weighted by cell size, plus a synthetic-data contrast."""
import numpy as np
from hyperhull_ivf.real_data import load_glove
from hyperhull_ivf.data import DataConfig, make_dataset
from hyperhull_ivf.ivf import build_ivf
from hyperhull_ivf.hulls import subdivide_lists

def margins(base, queries, n_lists, block):
    ivf, layout = subdivide_lists(build_ivf(base, n_lists, 100_000), block)
    out = {}
    for name, offs in (("lists", ivf.offsets), (f"blocks{block}", layout.block_offsets)):
        sizes = np.diff(offs); ratios = []; w = []
        for q in queries:
            dd = np.sqrt(((ivf.vectors - q) ** 2).sum(1)); kth = np.partition(dd, 9)[9]
            r = np.minimum.reduceat(dd, offs[:-1]) / kth
            ratios.append(r); w.append(sizes)
        r, w = np.concatenate(ratios), np.concatenate(w).astype(float)
        o = np.argsort(r); cw = np.cumsum(w[o]) / w.sum()
        pct = {p: r[o][np.searchsorted(cw, p)] for p in (0.1, 0.25, 0.5, 0.75, 0.9)}
        out[name] = "  ".join(f"p{int(p*100)}={v:.3f}" for p, v in pct.items())
    return out

g = load_glove(n_queries=100, n_base=100_000)
for k, v in margins(g.base, g.queries, 256, 16).items(): print(f"glove100k  {k:<9} true_min/kth (vector-weighted): {v}")
s = make_dataset(DataConfig(topic_spread=0.6, noise=0.09))
for k, v in margins(s.base, s.queries_in_dist[:100], 48, 16).items(): print(f"synthetic  {k:<9} true_min/kth (vector-weighted): {v}")
