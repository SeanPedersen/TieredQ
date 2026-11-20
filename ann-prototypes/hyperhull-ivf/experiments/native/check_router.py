"""Correctness on toy data: exhaustive search must equal brute force (chunked float64 truth, low memory)."""
import numpy as np
from hyperhull_ivf.native_router import make_router, native_arrays
from hyperhull_ivf.two_stage import build_two_level

rng = np.random.default_rng(0)
D = 256
centers = rng.standard_normal((50, D)).astype(np.float32)
base = (centers[rng.integers(0, 50, 20000)] + 0.5 * rng.standard_normal((20000, D))).astype(np.float32)
q = (centers[rng.integers(0, 50, 203)] + 0.5 * rng.standard_normal((203, D))).astype(np.float32)  # partial block
b64 = base.astype(np.float64)
exact = np.stack([((b64 - qi.astype(np.float64)) ** 2).sum(1) for qi in q])  # 203 x 20000
gt = np.argsort(exact, 1)[:, :10]
rec = lambda ids: np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, gt)])
idx = build_two_level(base, 70, 8)
for pca in (128, 256):
    r = make_router(native_arrays(idx, pca_dims=pca))
    d, ids, _ = r.search(q, 10, 70, 560, 20000, n_threads=1)
    bad = [i for i in range(len(q)) if set(ids[i]) != set(gt[i])]
    gaps = [float(np.sort(exact[i])[10] - np.sort(exact[i])[9]) for i in bad]
    same = np.array_equal(r.search(q, 10, 16, 128, 200, n_threads=4)[1], r.search(q, 10, 16, 128, 200, n_threads=1)[1])
    print(f"pca{pca}: exhaustive recall {rec(ids):.4f}, mismatches {bad} (10th/11th gap {gaps}), threads equal {same}")
    for T, F, V in [(8, 32, 50), (8, 64, 100), (16, 128, 200)]:
        print(f"   T={T} F={F} V={V} recall {rec(r.search(q, 10, T, F, V, n_threads=1)[1]):.3f}")

# two-pass sketch (prefix 128 of 256): exhaustive (every candidate refined) must still be exact
r = make_router(native_arrays(idx, pca_dims=256), prefix_dims=128)
_, ids, _ = r.search(q, 10, 70, 560, 20000, refine=20000, n_threads=1)
same = np.array_equal(r.search(q, 10, 16, 128, 200, refine=600, n_threads=4)[1], r.search(q, 10, 16, 128, 200, refine=600, n_threads=1)[1])
print(f"two-pass p128: exhaustive recall {rec(ids):.4f}, threads equal {same}")
for T, F, V, R in [(8, 32, 50, 200), (8, 64, 100, 400), (16, 128, 200, 800)]:
    print(f"   T={T} F={F} V={V} R={R} recall {rec(r.search(q, 10, T, F, V, refine=R, n_threads=1)[1]):.3f}")

# k-NN graph: exhaustive self-search must give the exact graph; expansion must never lose exactness and
# must help when the scan is starved
G = 16
knn = np.empty((len(base), G), dtype=np.int64)
b32 = base.astype(np.float32)
n2 = (b32 * b32).sum(1)
for s in range(0, len(base), 2000):
    d = n2[s : s + 2000, None] - 2 * b32[s : s + 2000] @ b32.T + n2[None, :]
    d[np.arange(len(d)), np.arange(s, s + len(d))] = np.inf
    knn[s : s + 2000] = np.argsort(d, 1)[:, :G]
row_of_id = np.empty(len(base), dtype=np.int64)
row_of_id[idx.ids] = np.arange(len(base))
for reverse in (False, True):
    r.build_graph(G, 70, 560, 20000, 20000, reverse_edges=reverse, n_threads=8)
    graph_ids = idx.ids[r.graph()]  # graph rows → original ids, per row in index order
    exact_rows = knn[idx.ids]       # exact neighbours (original ids) of each index row
    overlap = np.mean([len(set(a) & set(b)) / G for a, b in zip(graph_ids, exact_rows)])
    print(f"graph exhaustive build (reverse={reverse}): overlap with exact {G}-NN {overlap:.4f}")
_, ids, _ = r.search(q, 10, 70, 560, 20000, refine=20000, hops=2, expand_from=10, expand_k=16, expand_exact=50)
print(f"graph expansion, exhaustive scan: recall {rec(ids):.4f}")
r.build_graph(G, 16, 64, 60, 400, reverse_edges=True, n_threads=8)
for T, F, V, R in [(2, 4, 20, 100), (4, 16, 30, 200)]:
    plain = rec(r.search(q, 10, T, F, V, refine=R)[1])
    graph = rec(r.search(q, 10, T, F, V, refine=R, hops=2, expand_from=10, expand_k=16, expand_exact=20)[1])
    t4 = r.search(q, 10, T, F, V, refine=R, hops=2, expand_from=10, expand_k=16, expand_exact=20, n_threads=4)[1]
    t1 = r.search(q, 10, T, F, V, refine=R, hops=2, expand_from=10, expand_k=16, expand_exact=20, n_threads=1)[1]
    print(f"   T={T} F={F} V={V} R={R}: recall plain {plain:.3f} → graph {graph:.3f}, threads equal {np.array_equal(t4, t1)}")
