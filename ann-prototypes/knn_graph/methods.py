"""All-points k-NN graph builders for issue #56 (dimensionality-reduction input).

Every builder maps (X, k, threads, **knobs) -> (n, k) int64 neighbor ids, self excluded.
Ranking is squared L2 (TieredQ uses inner product: the study data is L2-normalised, same order). Each builder does its own index build, so its wall time is the full
cost a DR algorithm pays: build plus searching every row.

`cluster_join` is the IVF variant under test. Standard IVF search treats each row as an
independent query, so it pays per-query routing and scans lists with matrix-vector work.
In the self-join case the queries are the indexed rows, so all rows of one list share a
probe set: the list's nearest lists by centroid. One GEMM per list (members x candidates)
then replaces |list| separate scans, which is BLAS-3 and cache-resident.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

QUERY_CHUNK_ROWS = 256
TRUTH_CHUNK_FLOATS = 1 << 26
PHASES: dict[str, float] = {}  # optional per-phase seconds a builder reports to the study
KMEANS_ITERATIONS = 10
KMEANS_SAMPLE_PER_LIST = 64


def _top_k_by_score(scores: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the k smallest scores per row, ordered."""
    if k >= scores.shape[1]:
        return np.argsort(scores, axis=1)[:, :k]
    part = np.argpartition(scores, k, axis=1)[:, :k]
    order = np.take_along_axis(scores, part, axis=1).argsort(axis=1)
    return np.take_along_axis(part, order, axis=1)


def _l2_scores(queries: np.ndarray, base: np.ndarray, base_norms: np.ndarray) -> np.ndarray:
    """|x|^2 - 2 q.x: squared L2 minus the per-query constant |q|^2."""
    return base_norms[None, :] - 2.0 * (queries @ base.T)


def _drop_self(ids: np.ndarray, row_ids: np.ndarray, k: int) -> np.ndarray:
    """Remove each row's own id from its k+1 candidates, keep the first k."""
    keep = ids != row_ids[:, None]
    # Duplicate vectors can push self out of the top k+1; then drop the last column instead.
    keep[keep.all(axis=1), -1] = False
    return ids[keep].reshape(len(ids), k)


def _parallel(tasks, threads: int):
    with threadpool_limits(limits=1), ThreadPoolExecutor(threads) as pool:
        return list(pool.map(lambda task: task(), tasks))


def exact(X: np.ndarray, k: int, threads: int) -> np.ndarray:
    norms = np.einsum("ij,ij->i", X, X)
    starts = range(0, len(X), QUERY_CHUNK_ROWS)

    def chunk(start: int):
        rows = np.arange(start, min(start + QUERY_CHUNK_ROWS, len(X)))
        ids = _top_k_by_score(_l2_scores(X[rows], X, norms), k + 1)
        return _drop_self(ids, rows, k)

    return np.vstack(_parallel([lambda s=s: chunk(s) for s in starts], threads))


def _kmeans(X: np.ndarray, lists: int, threads: int, seed: int):
    import faiss

    faiss.omp_set_num_threads(threads)
    rng = np.random.default_rng(seed)
    sample = X[rng.choice(len(X), min(len(X), lists * KMEANS_SAMPLE_PER_LIST), replace=False)]
    kmeans = faiss.Kmeans(X.shape[1], lists, niter=KMEANS_ITERATIONS, seed=seed)
    kmeans.train(sample)
    _, assignment = kmeans.index.search(X, 1)
    return kmeans.centroids, assignment[:, 0]


def cluster_join(X: np.ndarray, k: int, threads: int, *, lists: int, probes: int, seed: int = 0):
    started = time.perf_counter()
    centroids, assignment = _kmeans(X, lists, threads, seed)
    PHASES["kmeans_s"] = round(time.perf_counter() - started, 3)
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids)
    probe_lists = _top_k_by_score(_l2_scores(centroids, centroids, centroid_norms), probes)

    order = np.argsort(assignment, kind="stable")
    bounds = np.searchsorted(assignment[order], np.arange(lists + 1))
    members = [order[bounds[c] : bounds[c + 1]] for c in range(lists)]
    norms = np.einsum("ij,ij->i", X, X)
    graph = np.empty((len(X), k), dtype=np.int64)

    def join(c: int):
        rows = members[c]
        if len(rows) == 0:
            return
        candidates = np.concatenate([members[p] for p in probe_lists[c]])
        width = min(k + 1, len(candidates))
        local = _top_k_by_score(_l2_scores(X[rows], X[candidates], norms[candidates]), width)
        ids = candidates[local]
        if width < k + 1:
            ids = np.pad(ids, ((0, 0), (0, k + 1 - width)), mode="edge")
        graph[rows] = _drop_self(ids, rows, k)

    # Largest lists first so the pool's tail is short.
    _parallel([lambda c=c: join(c) for c in np.argsort([-len(m) for m in members])], threads)
    return graph


def faiss_ivf_flat(X: np.ndarray, k: int, threads: int, *, lists: int, probes: int, seed: int = 0):
    import faiss

    faiss.omp_set_num_threads(threads)
    quantizer = faiss.IndexFlatL2(X.shape[1])
    index = faiss.IndexIVFFlat(quantizer, X.shape[1], lists)
    index.cp.niter = KMEANS_ITERATIONS
    index.cp.seed = seed
    rng = np.random.default_rng(seed)
    index.train(X[rng.choice(len(X), min(len(X), lists * KMEANS_SAMPLE_PER_LIST), replace=False)])
    index.add(X)
    index.nprobe = probes
    _, ids = index.search(X, k + 1)
    return _drop_self(ids, np.arange(len(X)), k)


def hnswlib_graph(X: np.ndarray, k: int, threads: int, *, m: int, ef_construction: int, ef: int, seed: int = 0):
    import hnswlib

    index = hnswlib.Index(space="l2", dim=X.shape[1])
    index.init_index(max_elements=len(X), M=m, ef_construction=ef_construction, random_seed=seed)
    index.set_num_threads(threads)
    index.add_items(X)
    index.set_ef(max(ef, k + 1))
    ids, _ = index.knn_query(X, k=k + 1, num_threads=threads)
    return _drop_self(ids.astype(np.int64), np.arange(len(X)), k)


def pynndescent_graph(X: np.ndarray, k: int, threads: int, *, seed: int = 0, neighbors: int | None = None, **options):
    """neighbors > k builds a wider graph and keeps its first k (pynndescent's usual way to buy recall)."""
    from pynndescent import NNDescent

    width = (neighbors or k) + 1
    index = NNDescent(X, n_neighbors=width, n_jobs=threads, random_state=seed, compressed=True, **options)
    ids, _ = index.neighbor_graph
    return _drop_self(ids.astype(np.int64)[:, : k + 1], np.arange(len(X)), k)


def rinnd_graph(X: np.ndarray, k: int, threads: int, *, seed: int = 0, neighbors: int | None = None, **options):
    """rinnd (lmcinnes): Rust NN-descent with RP-tree init, pynndescent's algorithm. Self is in its graph."""
    from rinnd import RINND

    width = (neighbors or k) + 1
    index = RINND(X, metric="euclidean", n_neighbors=width, n_jobs=threads, random_state=seed, graph_only=True,
                  **options)
    ids, _ = index.neighbor_graph
    return _drop_self(np.asarray(ids, dtype=np.int64)[:, : k + 1], np.arange(len(X)), k)


def manifolds_graph(X: np.ndarray, k: int, threads: int, *, ann: str, seed: int = 0, **params):
    """manifolds-rs (ann-search-rs backends): its knn_graph excludes self; Rayon threads come from the environment."""
    import manifolds_rs as mf

    nn_params = mf.NeighbourParams(**params) if params else None
    ids, _ = mf.knn_graph(X, k=k, metric="euclidean", ann=ann, nn_params=nn_params, seed=seed)
    return np.asarray(ids, dtype=np.int64)


def tieredq_ivf_flat(X: np.ndarray, k: int, threads: int, *, probes: int, lists: int | None = None, seed: int = 0,
                     storage: str = "fp32"):
    """The library's flat IVF (FP32 or SQ8 rows) and its list-major knn_graph (#56)."""
    from tieredq import TieredQIVFFlatIndex

    started = time.perf_counter()
    index = TieredQIVFFlatIndex(X.shape[1], lists=lists, metric="l2", seed=seed, storage=storage)
    index.build(X)
    PHASES.update(build_s=round(time.perf_counter() - started, 3), lists=index.lists,
                  storage_mb=round(index.storage_bytes / 2**20, 1),
                  **{key: round(value, 3) for key, value in index.build_stats.items()})
    started = time.perf_counter()
    _, ids = index.knn_graph(k, nprobe=probes)
    PHASES["join_s"] = round(time.perf_counter() - started, 3)
    return ids.astype(np.int64)


def exact_rows(X: np.ndarray, rows: np.ndarray, k: int, threads: int) -> np.ndarray:
    """Exact neighbours (self excluded) of the given rows against all of X."""
    norms = np.einsum("ij,ij->i", X, X)
    # Each worker holds chunk x len(X) scores (plus argpartition's copy): cap that at TRUTH_CHUNK_FLOATS.
    chunk_rows = max(1, min(QUERY_CHUNK_ROWS, TRUTH_CHUNK_FLOATS // len(X)))

    def chunk(start: int):
        part = rows[start : start + chunk_rows]
        return _drop_self(_top_k_by_score(_l2_scores(X[part], X, norms), k + 1), part, k)

    return np.vstack(_parallel([lambda s=s: chunk(s) for s in range(0, len(rows), chunk_rows)], threads))


def recall(found: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(found, truth))
    return hits / truth.size


def _search_order(index, rows: int, list_sorted: bool) -> np.ndarray:
    # Sorting queries by their own list makes neighbouring OpenMP chunks probe the same lists (cache reuse).
    return np.argsort(index.list_assignment(), kind="stable") if list_sorted else np.arange(rows)


def _unpermute(ids: np.ndarray, order: np.ndarray, k: int) -> np.ndarray:
    graph = np.empty((len(order), k), dtype=np.int64)
    graph[order] = _drop_self(ids.astype(np.int64), order, k)
    return graph


def tieredq_ivf(X: np.ndarray, k: int, threads: int, *, bits: float, probes: int, lists=None,
                fine_tiers: bool = False, list_sorted: bool = False):
    from tieredq import TieredQIVFIndex

    started = time.perf_counter()
    index = TieredQIVFIndex(X.shape[1], int(bits * X.shape[1]), lists=lists, metric="ip", fine_tiers=fine_tiers)
    index.build(X)
    PHASES["build_s"] = round(time.perf_counter() - started, 3)
    order = _search_order(index, len(X), list_sorted)
    _, ids = index.search(np.ascontiguousarray(X[order]), k + 1, nprobe=probes)
    return _unpermute(ids, order, k)


def tieredq_nested(X: np.ndarray, k: int, threads: int, *, probes: int, prefix_bits: float = 1.0, lists=None):
    from tieredq import TieredQNestedIVFIndex

    started = time.perf_counter()
    index = TieredQNestedIVFIndex.build(X, prefix_bits=int(prefix_bits * X.shape[1]), lists=lists)
    index.attach_tier(index.encode_tier(X, 0))
    PHASES["build_s"] = round(time.perf_counter() - started, 3)
    _, ids = index.search(X, k + 1, nprobe=probes)
    return _drop_self(ids.astype(np.int64), np.arange(len(X)), k)


def _refine_kernel():
    import numba

    @numba.njit(inline="always", fastmath=True)
    def score(X, norms, i, j):
        dot = np.float32(0.0)
        for d in range(X.shape[1]):
            dot += X[i, d] * X[j, d]
        return norms[j] - 2.0 * dot

    @numba.njit(parallel=True, cache=True)
    def refine(X, norms, graph, reverse_offsets, reverse_ids, rounds_out):
        # One local-join round: each row rescans its neighbours' neighbours and reverse neighbours.
        n, k = graph.shape
        for i in numba.prange(n):
            best_ids = graph[i].copy()
            best = np.empty(k, dtype=np.float32)
            for j in range(k):
                best[j] = score(X, norms, i, best_ids[j])
            worst = np.argmax(best)
            first_hops = np.concatenate((graph[i], reverse_ids[reverse_offsets[i]:reverse_offsets[i + 1]]))
            for a in first_hops:
                for b in graph[a]:
                    if b == i:
                        continue
                    seen = False
                    for j in range(k):
                        if best_ids[j] == b:
                            seen = True
                            break
                    if seen:
                        continue
                    candidate = score(X, norms, i, b)
                    if candidate < best[worst]:
                        best[worst] = candidate
                        best_ids[worst] = b
                        worst = np.argmax(best)
            rounds_out[i] = best_ids[np.argsort(best)]
        return rounds_out

    return refine


def refine(X: np.ndarray, graph: np.ndarray, rounds: int, reverse_cap: int) -> np.ndarray:
    """NN-descent local join on a starting graph: candidates are 2-hop neighbours (forward and up to
    reverse_cap reverse first hops), so a cheap but low-recall graph is pulled toward the exact one."""
    kernel = _refine_kernel()
    norms = np.einsum("ij,ij->i", X, X).astype(np.float32)
    for _ in range(rounds):
        sources = np.repeat(np.arange(len(X)), graph.shape[1])
        targets = graph.ravel()
        order = np.argsort(targets, kind="stable")
        reverse_sorted, targets_sorted = sources[order], targets[order]
        offsets = np.searchsorted(targets_sorted, np.arange(len(X) + 1))
        # Cap reverse lists so hub rows do not dominate the cost.
        starts = offsets[:-1]
        counts = np.minimum(np.diff(offsets), reverse_cap)
        keep = np.concatenate([np.arange(s, s + c) for s, c in zip(starts, counts)])
        capped_offsets = np.concatenate(([0], np.cumsum(counts)))
        graph = kernel(X, norms, graph, capped_offsets, reverse_sorted[keep], np.empty_like(graph))
    return graph


def refined(builder, *, rounds: int = 1, reverse_cap: int = 15):
    def build(X: np.ndarray, k: int, threads: int):
        start_graph = builder(X, k, threads)
        started = time.perf_counter()
        graph = refine(X, start_graph, rounds, reverse_cap)
        PHASES["refine_s"] = round(time.perf_counter() - started, 3)
        return graph

    return build


def ivf_join(X: np.ndarray, k: int, threads: int, *, lists: int, probes: int, seed: int = 0):
    """Plain (unquantized) IVF with per-row probing, executed list-major for the self-join.

    Recall equals per-query IVF at the same probes; the work is regrouped so each list's members
    are scored against every row that probes it in one GEMM, instead of one matrix-vector scan
    per row and list.
    """
    started = time.perf_counter()
    centroids, _ = _kmeans(X, lists, threads, seed)
    PHASES["kmeans_s"] = round(time.perf_counter() - started, 3)
    norms = np.einsum("ij,ij->i", X, X)
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids)
    starts = range(0, len(X), QUERY_CHUNK_ROWS)

    def route(start: int):
        rows = slice(start, min(start + QUERY_CHUNK_ROWS, len(X)))
        return _top_k_by_score(_l2_scores(X[rows], centroids, centroid_norms), probes)

    probe_lists = np.vstack(_parallel([lambda s=s: route(s) for s in starts], threads))
    primary = probe_lists[:, 0]
    order = np.argsort(primary, kind="stable")
    bounds = np.searchsorted(primary[order], np.arange(lists + 1))
    members = [order[bounds[c] : bounds[c + 1]] for c in range(lists)]

    # Invert (row, rank, list) so each list knows which rows probe it and into which slot they write.
    flat_lists = probe_lists.ravel()
    by_list = np.argsort(flat_lists, kind="stable")
    probe_bounds = np.searchsorted(flat_lists[by_list], np.arange(lists + 1))
    width = k + 1
    slot_scores = np.full((len(X) * probes, width), np.inf, dtype=np.float32)
    slot_ids = np.full((len(X) * probes, width), -1, dtype=np.int64)

    def scan(c: int):
        base = members[c]
        if len(base) == 0:
            return
        slots = by_list[probe_bounds[c] : probe_bounds[c + 1]]
        scores = _l2_scores(X[slots // probes], X[base], norms[base])
        take = min(width, len(base))
        local = _top_k_by_score(scores, take)
        slot_scores[slots, :take] = np.take_along_axis(scores, local, axis=1)
        slot_ids[slots, :take] = base[local]

    _parallel([lambda c=c: scan(c) for c in np.argsort([-len(m) for m in members])], threads)

    def merge(start: int):
        rows = np.arange(start, min(start + QUERY_CHUNK_ROWS, len(X)))
        flat = slice(rows[0] * probes, (rows[-1] + 1) * probes)
        scores = slot_scores[flat].reshape(len(rows), probes * width)
        ids = slot_ids[flat].reshape(len(rows), probes * width)
        best = _top_k_by_score(scores, width)
        return _drop_self(np.take_along_axis(ids, best, axis=1), rows, k)

    return np.vstack(_parallel([lambda s=s: merge(s) for s in starts], threads))


# --- #56 follow-up: combinations aimed at pynndescent -------------------------------------------------------------
PCA_SAMPLE_ROWS = 32768


def pca(X: np.ndarray, dims: int, seed: int = 0) -> np.ndarray:
    """Rows projected on the top `dims` principal directions of a seeded sample (centered, float32)."""
    rng = np.random.default_rng(seed)
    sample = X[np.sort(rng.choice(len(X), min(len(X), PCA_SAMPLE_ROWS), replace=False))].astype(np.float64)
    mean = sample.mean(axis=0)
    centered = sample - mean
    _, vectors = np.linalg.eigh(centered.T @ centered)
    basis = np.ascontiguousarray(vectors[:, ::-1][:, :dims], dtype=np.float32)
    return np.ascontiguousarray((X - mean.astype(np.float32)) @ basis)


def _rerank_kernel():
    import numba

    @numba.njit(parallel=True, fastmath=True, cache=True)
    def rerank(X, candidates, k, out):
        n, width = candidates.shape
        for i in numba.prange(n):
            distances = np.full(width, np.inf, dtype=np.float32)
            for c in range(width):
                j = candidates[i, c]
                if j < 0 or j == i:
                    continue
                total = np.float32(0.0)
                for d in range(X.shape[1]):
                    diff = X[i, d] - X[j, d]
                    total += diff * diff
                distances[c] = total
            order = np.argsort(distances)[:k]
            for c in range(k):
                out[i, c] = candidates[i, order[c]] if distances[order[c]] < np.inf else -1
        return out

    return rerank


def rerank(X: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    """Each row's k nearest among its candidate ids by exact squared L2 (self and -1 ignored)."""
    candidates = np.where(candidates >= np.iinfo(np.int32).max, -1, candidates).astype(np.int64)
    return _rerank_kernel()(X, candidates, k, np.empty((len(X), k), dtype=np.int64))


def _timed(phase: str, action):
    started = time.perf_counter()
    result = action()
    PHASES[phase] = round(time.perf_counter() - started, 3)
    return result


def pca_nndescent(X: np.ndarray, k: int, threads: int, *, dims: int, candidates: int, seed: int = 0, **options):
    """NN-descent on a PCA projection (distances dims/d as costly) for `candidates` ids, reranked exactly."""
    projected = _timed("pca_s", lambda: pca(X, dims, seed))
    found = _timed("graph_s", lambda: pynndescent_graph(projected, candidates, threads, seed=seed, **options))
    return _timed("rerank_s", lambda: rerank(X, found, k))


def pca_ivf(X: np.ndarray, k: int, threads: int, *, dims: int, candidates: int, lists: int, probes: int,
            seed: int = 0):
    """IVF-Flat knn_graph on a PCA projection for `candidates` ids, reranked exactly."""
    projected = _timed("pca_s", lambda: pca(X, dims, seed))
    found = _timed("graph_s", lambda: tieredq_ivf_flat(projected, candidates, threads, lists=lists, probes=probes,
                                                         seed=seed))
    return _timed("rerank_s", lambda: rerank(X, found, k))


def pca_rerank(X: np.ndarray, k: int, threads: int, *, dims: int, candidates: int, builder: str, seed: int = 0,
               **options):
    """Any graph builder on a PCA projection for `candidates` ids, reranked exactly in the full dimension."""
    builders = {"pynndescent": pynndescent_graph, "rinnd": rinnd_graph, "manifolds": manifolds_graph,
                "tieredq": tieredq_ivf_flat}
    projected = _timed("pca_s", lambda: pca(X, dims, seed))
    found = _timed("graph_s", lambda: builders[builder](projected, candidates, threads, seed=seed, **options))
    return _timed("rerank_s", lambda: rerank(X, found, k))


def ivf_init_nndescent(X: np.ndarray, k: int, threads: int, *, lists: int, probes: int, seed: int = 0, **options):
    """pynndescent started from a cheap IVF-Flat graph instead of its random-projection trees."""
    from pynndescent import NNDescent

    start = _timed("init_s", lambda: tieredq_ivf_flat(X, k, threads, lists=lists, probes=probes, seed=seed))
    start = np.hstack([np.arange(len(X))[:, None], np.where(start >= np.iinfo(np.int32).max, -1, start)])

    def descend():
        index = NNDescent(X, n_neighbors=k + 1, n_jobs=threads, random_state=seed, compressed=True,
                          tree_init=False, init_graph=start.astype(np.int32), **options)
        return index.neighbor_graph[0].astype(np.int64)

    ids = _timed("descent_s", descend)
    return _drop_self(ids, np.arange(len(X)), k)
