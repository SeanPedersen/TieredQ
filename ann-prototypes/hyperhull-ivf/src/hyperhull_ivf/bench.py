"""Hyper-hull vs FAISS benchmark on 100k real-embedding subsets: recall vs cost,
build time, index size and build peak RAM.

Each method runs in a fresh subprocess (`--method NAME`) so peak RSS is not
polluted by other methods. Cost model (identical for all methods, per query):
    routing (centroids) + codes scanned · bits/32 + exact float distances
Wall time is single-threaded. FAISS is optimized C++; our hull search is a
Python reference loop, so its wall time is an upper bound, not a fair speed claim.

Usage: uv run hyperhull-bench [--dataset glove100k|dbpedia100k] [method ...]
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import K, recall
from .ivf import sq_dists
from .real_data import load_dbpedia, load_glove

N_BASE = 100_000
N_QUERIES = 200
N_LISTS = 256
NPROBES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
K_FACTORS = (1, 2, 4, 8, 16, 32)
EF_SEARCH = (10, 16, 32, 64, 128, 256, 512)
RECALL_TARGETS = (0.90, 0.95, 0.99, 1.0)
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
METHODS = ("faiss-ivf-flat", "faiss-ivf-sq8+refine", "faiss-ivf-sq4+refine", "faiss-ivf-pq4fs+refine", "faiss-hnsw32", "errball-pq4+adaptive", "va4-hull", "va2-hull")


# ---------------------------------------------------------------- shared data


@dataclass(frozen=True)
class DatasetSpec:
    title: str
    loader: object  # () -> RealDataset
    pq_subquantizers: int  # PQ m for 4-bit codes


def _subset(loader_fn):
    def load():
        d = loader_fn()
        rows = np.sort(np.random.default_rng(0).choice(len(d.base), size=N_BASE, replace=False))
        return np.asarray(d.base[rows], dtype=np.float32), d.queries

    return load


DATASETS = {
    "glove100k": DatasetSpec("GloVe-100", lambda: (lambda d: (d.base, d.queries))(load_glove(N_QUERIES, N_BASE)), 50),
    "dbpedia100k": DatasetSpec("DBpedia OpenAI-1536", _subset(lambda: load_dbpedia(N_QUERIES)), 192),
}


def load_data(dataset: str):
    cache = CACHE_DIR / f"{dataset}.npz"
    if cache.exists():
        c = np.load(cache)
        return c["base"], c["queries"], c["gt"]
    base, queries = DATASETS[dataset].loader()
    gt = np.argpartition(sq_dists(queries, base), K, axis=1)[:, :K]
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache, base=base, queries=queries, gt=gt)
    return base, queries, gt


def _rss_mb(field: str) -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(field):
            return int(line.split()[1]) / 1024
    return float("nan")


@dataclass
class Point:
    params: dict
    recall: float
    cost: float  # blended model: routing + codes·bits/(32·D) + exact
    ms_per_query: float
    codes: float = 0.0  # compressed codes scanned per query
    exact: float = 0.0  # full-precision distances per query


# ---------------------------------------------------------------- FAISS methods


def faiss_ivf_methods(dim: int, pq_m: int) -> dict[str, tuple[str, int]]:
    """name -> (index_factory string, code bits per vector)."""
    return {
        "faiss-ivf-flat": (f"IVF{N_LISTS},Flat", 32 * dim),
        "faiss-ivf-sq8+refine": (f"IVF{N_LISTS},SQ8,Refine(Flat)", 8 * dim),
        "faiss-ivf-sq4+refine": (f"IVF{N_LISTS},SQ4,Refine(Flat)", 4 * dim),
        "faiss-ivf-pq4fs+refine": (f"IVF{N_LISTS},PQ{pq_m}x4fs,Refine(Flat)", 4 * pq_m),
    }


def _faiss_ivf_points(index, code_bits: int, queries, gt, refine: bool) -> list[Point]:
    import faiss

    space = faiss.ParameterSpace()
    refine_index = faiss.downcast_index(index) if refine else None
    full_vector_bits = 32 * queries.shape[1]
    points = []
    for nprobe in NPROBES:
        for k_factor in K_FACTORS if refine else (1,):
            space.set_index_parameter(index, "nprobe", nprobe)
            params = {"nprobe": nprobe}
            if refine:
                refine_index.k_factor = k_factor
                params["k_factor"] = k_factor
            faiss.cvar.indexIVF_stats.reset()
            t = time.perf_counter()
            _, found = index.search(queries, K)
            ms = (time.perf_counter() - t) * 1000 / len(queries)
            codes = faiss.cvar.indexIVF_stats.ndis / len(queries)
            if refine:
                exact = k_factor * K
            else:  # IVF-Flat: every scanned "code" is a full vector
                codes, exact = 0.0, codes
            cost = N_LISTS + codes * code_bits / full_vector_bits + exact
            points.append(Point(params, recall(found, gt), cost, ms, codes, exact))
    return points


def _faiss_hnsw(base):
    import faiss

    index = faiss.IndexHNSWFlat(base.shape[1], 32)
    index.add(base)
    return index


def _faiss_hnsw_points(index, queries, gt) -> list[Point]:
    import faiss

    points = []
    for ef in EF_SEARCH:
        index.hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        t = time.perf_counter()
        _, found = index.search(queries, K)
        ms = (time.perf_counter() - t) * 1000 / len(queries)
        cost = faiss.cvar.hnsw_stats.ndis / len(queries)
        points.append(Point({"efSearch": ef}, recall(found, gt), cost, ms, 0.0, cost))
    return points


# ---------------------------------------------------------------- error-ball hull on IVF-PQ
# Each vector x lies in the ball of radius e_x = ||x - x̂|| around its PQ reconstruction,
# so with d̂ = ||q - x̂||^2 (exact float LUTs, i.e. non-fast-scan PQ):
#     (sqrt(d̂) - e_x)_+^2  <=  ||q - x||^2  <=  (sqrt(d̂) + e_x)^2
# Probabilistic tightening (RaBitQ-style): ||q-x||^2 = d̂ + e^2 - 2 (q-x̂)·δ, and for an error
# δ in a near-random direction (q-x̂)·δ ≈ sqrt(d̂)·e·N(0,1)/sqrt(D). TAU (in standard
# deviations) widens the ball; TAU = sqrt(D) recovers the sound bound above.
# Re-rank in lower-bound order until LB > threshold, where threshold = min(current K-th
# exact distance, K-th smallest upper bound). A query is *certified* (exact within its
# probed lists, sound bound only) when even the worst returned candidate lies beyond the threshold.

CANDIDATE_POOL = 1000
RERANK_CHUNK = 8
TAUS = (1.0, 2.0, 3.0, 4.0, None)  # None = sqrt(D): sound


def _errball_build(base, pq_m: int):
    import faiss

    index = faiss.index_factory(base.shape[1], f"IVF{N_LISTS},PQ{pq_m}x4")
    index.train(base)
    index.add(base)
    faiss.extract_index_ivf(index).make_direct_map()
    error = np.linalg.norm(base - index.reconstruct_n(0, len(base)), axis=1).astype(np.float32)
    return index, error


def _errball_rerank(base, q, approx_d, ids, error, tau):
    """Adaptive exact re-ranking of one query's candidates; returns (top-K ids, exact count, certified)."""
    valid = ids >= 0
    approx = np.sqrt(np.maximum(approx_d[valid], 0))
    ids, err = ids[valid], error[ids[valid]]
    slack = 2 * (tau / np.sqrt(base.shape[1]) if tau is not None else 1.0) * approx * err
    lower = np.maximum(approx**2 + err**2 - slack, 0)
    upper = approx**2 + err**2 + slack
    threshold = np.partition(upper, K - 1)[K - 1] if len(upper) >= K else np.inf
    order = np.argsort(lower)
    best_d, best_i, exact = np.full(K, np.inf, np.float32), np.full(K, -1), 0
    for start in range(0, len(order), RERANK_CHUNK):
        chunk = order[start : start + RERANK_CHUNK]
        chunk = chunk[lower[chunk] <= min(threshold, best_d[-1])]
        if len(chunk) == 0:
            break
        diff = base[ids[chunk]] - q
        d = np.einsum("nd,nd->n", diff, diff)
        exact += len(d)
        merged_d, merged_i = np.concatenate([best_d, d]), np.concatenate([best_i, ids[chunk]])
        keep = np.argsort(merged_d)[:K]
        best_d, best_i = merged_d[keep], merged_i[keep]
    worst_unseen_lb = (max(np.sqrt(approx_d.max()) - error.max(), 0)) ** 2
    certified = len(ids) < CANDIDATE_POOL or worst_unseen_lb > best_d[-1]
    return best_i, exact, certified


def _errball_points(index, error, base, queries, gt, code_bits: int) -> list[Point]:
    import faiss

    space = faiss.ParameterSpace()
    full_vector_bits = 32 * base.shape[1]
    points = []
    for nprobe in NPROBES:
        space.set_index_parameter(index, "nprobe", nprobe)
        faiss.cvar.indexIVF_stats.reset()
        t = time.perf_counter()
        approx_d, ids = index.search(queries, CANDIDATE_POOL)
        scan_ms = (time.perf_counter() - t) * 1000 / len(queries)
        codes = faiss.cvar.indexIVF_stats.ndis / len(queries)
        for tau in TAUS:
            t = time.perf_counter()
            reranked = [_errball_rerank(base, q, d, i, error, tau) for q, d, i in zip(queries, approx_d, ids)]
            ms = scan_ms + (time.perf_counter() - t) * 1000 / len(queries)
            exact = np.mean([r[1] for r in reranked])
            certified = np.mean([r[2] for r in reranked])
            found = np.stack([r[0] for r in reranked])
            cost = N_LISTS + codes * code_bits / full_vector_bits + exact
            params = {"nprobe": nprobe, "tau": tau, "exact_per_query": round(float(exact), 1), "certified": round(float(certified), 3)}
            points.append(Point(params, recall(found, gt), float(cost), ms, codes, float(exact)))
    return points


# ---------------------------------------------------------------- hull methods


def _hull_build(base, bits: int):
    import faiss

    from .ivf import IVFIndex
    from .va_search import quantize_lists

    kmeans = faiss.Kmeans(base.shape[1], N_LISTS, niter=20, seed=1)
    kmeans.train(base)
    assign = kmeans.index.search(base, 1)[1][:, 0]
    order = np.argsort(assign, kind="stable")
    offsets = np.concatenate([[0], np.cumsum(np.bincount(assign, minlength=N_LISTS))])
    ivf = IVFIndex(kmeans.centroids, base[order], order, offsets)
    return ivf, quantize_lists(ivf, bits)


def _hull_size_bytes(ivf, ql) -> int:
    packed_codes = ql.codes.size * ql.bits // 8
    side = ql.lo.nbytes + ql.step.nbytes + ql.radius.nbytes + ivf.centroids.nbytes + ivf.ids.nbytes
    return packed_codes + side + ivf.vectors.nbytes  # raw vectors kept for exact distances


def _hull_points(ivf, ql, queries, gt) -> list[Point]:
    from .va_search import search_one

    points = []
    for nprobe in NPROBES:
        t = time.perf_counter()
        results = [search_one(q, ivf, ql, 0.0, nprobe) for q in queries]
        ms = (time.perf_counter() - t) * 1000 / len(queries)
        found = ivf.ids[np.stack([r[0] for r in results])]
        cost = np.mean([c.routing + c.code_evals * ql.bits / 32 + c.exact for _, c in results])
        codes = float(np.mean([c.code_evals for _, c in results]))
        exact = float(np.mean([c.exact for _, c in results]))
        points.append(Point({"nprobe": nprobe}, recall(found, gt), float(cost), ms, codes, exact))
    return points


# ---------------------------------------------------------------- runner


def run_method(dataset: str, name: str) -> dict:
    import faiss

    faiss.omp_set_num_threads(1)
    base, queries, gt = load_data(dataset)
    pq_m = DATASETS[dataset].pq_subquantizers
    ivf_methods = faiss_ivf_methods(base.shape[1], pq_m)
    rss_before = _rss_mb("VmRSS")
    t = time.perf_counter()
    if name in ivf_methods:
        factory, code_bits = ivf_methods[name]
        index = faiss.index_factory(base.shape[1], factory)
        index.train(base)
        index.add(base)
        build_s = time.perf_counter() - t
        size = len(faiss.serialize_index(index))
        points = _faiss_ivf_points(index, code_bits, queries, gt, refine="Refine" in factory)
    elif name == "errball-pq4+adaptive":
        index, error = _errball_build(base, pq_m)
        build_s = time.perf_counter() - t
        size = len(faiss.serialize_index(index)) + error.nbytes + base.nbytes  # + raw vectors for re-rank
        points = _errball_points(index, error, base, queries, gt, 4 * pq_m)
    elif name == "faiss-hnsw32":
        index = _faiss_hnsw(base)
        build_s = time.perf_counter() - t
        size = len(faiss.serialize_index(index))
        points = _faiss_hnsw_points(index, queries, gt)
    else:
        bits = int(name[2])
        ivf, ql = _hull_build(base, bits)
        build_s = time.perf_counter() - t
        size = _hull_size_bytes(ivf, ql)
        points = _hull_points(ivf, ql, queries, gt)
    build_peak = _rss_mb("VmHWM") - rss_before
    return {
        "method": name,
        "build_s": build_s,
        "index_mb": size / 2**20,
        "build_peak_mb": max(build_peak, 0.0),
        "points": [p.__dict__ for p in points],
    }


def _operating_point(points: list[dict], target: float) -> dict | None:
    """Interpolated (cost, codes, exact, ms) at target recall along the blended-cost frontier."""
    frontier, best = [], -1.0
    for p in sorted(points, key=lambda p: p["cost"]):
        if p["recall"] > best:
            frontier.append(p)
            best = p["recall"]
    for prev, cur in zip([None, *frontier], frontier):
        if cur["recall"] < target:
            continue
        frac = 1.0 if prev is None or cur["recall"] == prev["recall"] else (target - prev["recall"]) / (cur["recall"] - prev["recall"])
        base = prev or cur
        return {k: base.get(k, 0.0) + frac * (cur.get(k, 0.0) - base.get(k, 0.0)) for k in ("cost", "codes", "exact", "ms_per_query")}
    return None


def print_table(dataset: str, dim: int, results: list[dict]) -> None:
    title = DATASETS[dataset].title
    print(f"\n{title} {N_BASE // 1000}k (D={dim}), Q={N_QUERIES}, K={K}, single thread")
    print("per query at each recall target: exact = full-precision distances, codes = compressed codes scanned,")
    print("pruned = share of the N vectors never touched at full precision\n")
    print(f"{'method':<22}{'build s':>8}{'idx MB':>8}{'peak MB':>8}" + "".join(f" | R>={t:<4} {'exact':>7} {'codes':>7} {'pruned':>6}" for t in RECALL_TARGETS))
    for r in results:
        row = f"{r['method']:<22}{r['build_s']:8.1f}{r['index_mb']:8.1f}{r['build_peak_mb']:8.0f}"
        for t in RECALL_TARGETS:
            op = _operating_point(r["points"], t)
            if op is None:
                row += f" | {'-':>13} {'-':>7} {'-':>6}"
                continue
            row += f" | {op['exact']:13.0f} {op['codes']:7.0f} {1 - op['exact'] / N_BASE:6.3f}"
        print(row, flush=True)


def reprint(dataset: str) -> None:
    results = json.loads((CACHE_DIR / f"bench_{dataset}.json").read_text())
    print_table(dataset, int(np.load(CACHE_DIR / f"{dataset}.npz", mmap_mode="r")["base"].shape[1]), results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove100k", choices=DATASETS)
    parser.add_argument("--method", help="internal: run one method in this process")
    parser.add_argument("methods", nargs="*", default=list(METHODS))
    args = parser.parse_args()
    if args.method:
        print(json.dumps(run_method(args.dataset, args.method)))
        return
    if args.methods == ["report"]:
        reprint(args.dataset)
        return
    base, _, _ = load_data(args.dataset)  # build the cache once before the subprocesses
    dim = base.shape[1]
    del base
    results = []
    for name in args.methods:
        cmd = [sys.executable, "-m", "hyperhull_ivf.bench", "--dataset", args.dataset, "--method", name]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{name} failed:\n{out.stderr[-2000:]}", flush=True)
            continue
        results.append(json.loads(out.stdout.strip().splitlines()[-1]))
        print(f"  done {name}", flush=True)
    print_table(args.dataset, dim, results)
    (CACHE_DIR / f"bench_{args.dataset}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
