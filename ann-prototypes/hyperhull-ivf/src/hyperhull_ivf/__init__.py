"""Hyper-hull IVF experiment: can geometric list summaries reach the same
Recall@10 as centroid-routed IVF with fewer distance computations?

Two families are compared at equal recall, counting every distance computation:
  * heuristic routers (re-rank lists by estimated NN distance, fixed or adaptive nprobe)
  * sound outer-hull search (PCA box x orthogonal ring x sphere bounds on lists
    and sub-list blocks; exact at eps=0, guaranteed (1+eps)-approximate otherwise)

Usage: uv run hyperhull-ivf [glove100k|glove|dbpedia|synthetic]   (default: glove100k)
"""

import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from .bounded_search import HullIndex, bounded_curve
from .data import DataConfig, make_dataset
from .evaluate import adaptive_curve, build_cache, fixed_nprobe_curve
from .hulls import build_hulls, subdivide_lists
from .ivf import build_ivf
from .real_data import load_dbpedia, load_glove
from .report import print_cost_table, print_curve, print_header
from .routers import centroid_router, pca_box_router, pca_rerank_router, sphere_router
from .summaries import build_summaries
from .va_search import quantize_lists, va_curve

ADAPTIVE_SLACKS = np.round(np.geomspace(0.2, 6.0, 30), 3)
EPS_VALUES = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
CELL_HULL_EPS = (0.0, 1.0)  # cell-level hulls barely prune (see tightness.py); keep as a reference point
VA_BITS = (4, 8)
LIST_HULL_COMPONENTS = 16
BLOCK_HULL_COMPONENTS = 4


@dataclass(frozen=True)
class Experiment:
    title: str
    base: np.ndarray
    query_sets: dict[str, np.ndarray]
    n_lists: int
    block_size: int = 64
    train_size: int = 100_000

    @property
    def nprobes(self) -> list[int]:
        grid = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
        return [n for n in grid if n <= self.n_lists]


def _experiments(name: str) -> list[Experiment]:
    if name == "glove100k":
        d = load_glove(n_queries=200, n_base=100_000)
        return [Experiment(d.name, d.base, {"real": d.queries}, n_lists=256)]
    if name == "glove":
        d = load_glove(n_queries=200)
        return [Experiment(d.name, d.base, {"real": d.queries}, n_lists=1024)]
    if name == "dbpedia":
        d = load_dbpedia(n_queries=200)
        return [Experiment(d.name, d.base, {"real": d.queries}, n_lists=512, train_size=60_000)]
    if name == "synthetic":
        cfg = DataConfig(topic_spread=0.6, noise=0.09)
        d = make_dataset(cfg)
        queries = {"in-distribution": d.queries_in_dist, "boundary": d.queries_boundary}
        return [Experiment("synthetic overlapping topics", d.base, queries, n_lists=48)]
    raise SystemExit(f"unknown experiment {name!r}")


def _log(msg: str, t0: float) -> None:
    print(f"  [{time.perf_counter() - t0:6.1f}s] {msg}", flush=True)


def run(exp: Experiment) -> None:
    t0 = time.perf_counter()
    ivf = build_ivf(exp.base, exp.n_lists, exp.train_size)
    ivf, layout = subdivide_lists(ivf, exp.block_size)
    hx = HullIndex(
        ivf,
        layout,
        build_hulls(ivf.vectors, ivf.offsets, LIST_HULL_COMPONENTS),
        build_hulls(ivf.vectors, layout.block_offsets, BLOCK_HULL_COMPONENTS),
    )
    summaries = build_summaries(ivf)
    routers = [
        centroid_router(ivf),
        sphere_router(ivf, summaries),
        pca_box_router(ivf, summaries, 8),
        pca_rerank_router(ivf, summaries, 8, shortlist=32),
    ]
    sizes = ivf.sizes
    print_header(
        f"{exp.title}: N={len(ivf.vectors)} D={ivf.vectors.shape[1]} lists={ivf.n_lists} "
        f"(size {sizes.min()}-{sizes.max()}) blocks={len(layout.block_offsets) - 1}"
    )
    _log("index + hulls built", t0)

    for set_name, queries in exp.query_sets.items():
        cache = build_cache(ivf, queries)
        _log(f"{set_name}: ground truth for Q={len(queries)}", t0)
        curves = {}
        for r in routers:
            curves[r.name] = fixed_nprobe_curve(ivf, cache, r, queries, exp.nprobes)
        for r in routers[:1] + routers[2:]:
            curves[f"{r.name} adaptive"] = adaptive_curve(ivf, cache, r, queries, ADAPTIVE_SLACKS, ivf.n_lists)
        _log("heuristic routers done", t0)
        curves["cell-hull (eps sweep)"] = bounded_curve(hx, queries, cache.gt, CELL_HULL_EPS)
        _log("cell-hull search done", t0)
        for bits in VA_BITS:
            ql = quantize_lists(ivf, bits)
            eps_curve = va_curve(ivf, ql, queries, cache.gt, EPS_VALUES, ivf.n_lists)
            probe_curve = [
                replace(p, param=n)
                for n in exp.nprobes
                for p in va_curve(ivf, ql, queries, cache.gt, (0.0,), n)
            ]
            curves[f"va{bits}-hull eps-sweep"] = eps_curve
            curves[f"va{bits}-hull nprobe-sweep"] = probe_curve
            print_curve(f"va{bits} eps", eps_curve, "eps")
            print_curve(f"va{bits} nprobe", probe_curve, "np")
        _log("vector-hull (VA) search done", t0)
        print_cost_table(curves, baseline="ivf-centroid", n_vectors=len(ivf.vectors))


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "glove100k"
    for exp in _experiments(name):
        run(exp)
