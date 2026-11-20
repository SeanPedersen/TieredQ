"""Idea 1: neighbour-graph expansion after a small cell scan (ranking-only validation).

Usage: graph_gap.py <export_dir>   (needs rank_gap.npz from rank_gap.py)
Scan the first F0 fine cells in router order (4-bit PCA-128, T=256), take the best P scanned vectors
(exact distance: optimistic, ≈ what the exact rerank of the scan yields), add their k nearest
neighbours (exact kNN graph), optionally a second hop. Report recall@10 coverage and vectors touched
(scanned + new expansion vectors). Test queries only (no tuning yet).
"""
import sys
from pathlib import Path

import numpy as np

from rank_gap import load

K = 10
KNN = 32


def knn_graph(vectors: np.ndarray, path: Path) -> np.ndarray:
    if path.exists():
        return np.load(path)
    import faiss

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    out = np.empty((len(vectors), KNN), dtype=np.int32)
    for s in range(0, len(vectors), 16384):
        _, ids = index.search(vectors[s : s + 16384], KNN + 1)
        out[s : s + 16384] = ids[:, 1:]
        print(f"knn {s}", file=sys.stderr, flush=True)
    np.save(path, out)
    return out


def main() -> None:
    export = Path(sys.argv[1])
    a = load(export)
    vectors = np.ascontiguousarray(a["vectors"])
    n = len(vectors)
    graph = knn_graph(vectors, export / "knn32.npy")
    fine_offsets = np.asarray(a["fine_offsets"])
    row_of_id = np.empty(n, dtype=np.int64)
    row_of_id[np.asarray(a["ids"])] = np.arange(n)
    queries = np.asarray(a["queries"])
    gt_rows = row_of_id[np.asarray(a["gt"])]
    router = np.load(export / "rank_gap.npz")["test_router"]
    norm2 = np.einsum("ij,ij->i", vectors, vectors)

    F0s, Ps, ks, hops_list = (25, 50, 100, 200, 400), (10, 30), (8, 16, 32), (1, 2)
    acc = {}
    for qi in range(len(queries)):
        q = queries[qi]
        d = norm2 - 2 * vectors @ q
        order = np.argsort(router[qi], kind="stable")
        gt = set(gt_rows[qi].tolist())
        for F0 in F0s:
            cells = order[:F0]
            scanned = np.concatenate([np.arange(fine_offsets[c], fine_offsets[c + 1]) for c in cells])
            base_hits = len(gt.intersection(scanned.tolist()))
            acc.setdefault((F0, 0, 0, 0), []).append((base_hits, len(scanned)))
            for P in Ps:
                for k in ks:
                    for hops in hops_list:
                        seen = set(scanned.tolist())
                        pool = scanned
                        added = 0
                        for _ in range(hops):
                            top = pool[np.argpartition(d[pool], min(P, len(pool) - 1))[:P]]
                            new = [v for v in np.unique(graph[top, :k]).tolist() if v not in seen]
                            seen.update(new)
                            added += len(new)
                            pool = np.fromiter(seen, dtype=np.int64)
                        acc.setdefault((F0, P, k, hops), []).append((len(gt & seen), len(scanned) + added))
        if qi % 100 == 99:
            print(f"query {qi + 1}", file=sys.stderr, flush=True)
    print("F0   P   k  hops   recall   vectors touched")
    for key in sorted(acc):
        hits, cost = np.array(acc[key]).T
        print(f"{key[0]:4d} {key[1]:3d} {key[2]:3d} {key[3]:4d}   {hits.mean() / K:.4f}   {cost.mean():9.0f}")


if __name__ == "__main__":
    main()
