"""Issue #56 acceptance: TieredQIVFFlatIndex.knn_graph against tuned pynndescent at matched recall@k, 2 seeds.

    uv run compare.py DATA.npy OUT_DIR --suite 100k|1m [--threads 8] [--k 15] [--seeds 0,1] [--truth-rows 0]

Every (configuration, seed) runs in a fresh process pinned to CPUs 0..threads-1 with that many OpenMP/numba
threads, so no run inherits another's caches or JIT. pynndescent is timed after a 2,000-row warm-up (numba JIT); the combined builders run whole on 20,000 rows first.
Recall is against exact neighbours: of every row (--truth-rows 0), or of a fixed seeded sample of rows, each ranked
against all rows (an unbiased estimate where exact all-pairs truth is too slow). Results append to
OUT_DIR/results.jsonl; finished (name, seed) pairs are skipped, so reruns resume.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

import methods

WARMUP_ROWS = 2000
NOVEL_WARMUP_ROWS = 20000
TRUTH_SEED = 12345
TRUTH_THREADS = 16

PYNNDESCENT = {
    "pynndescent": {},
    "pynndescent-t16-i20-c30": dict(n_trees=16, n_iters=20, max_candidates=30),
    "pynndescent-t32-i30-c60": dict(n_trees=32, n_iters=30, max_candidates=60),
    "pynndescent-n30": dict(neighbors=30),
    "pynndescent-n30-t32-i30-c60": dict(neighbors=30, n_trees=32, n_iters=30, max_candidates=60),
}
IVF_FLAT_GRID = {
    "100k": {128: (4, 8, 16, 32), 256: (8, 16, 32, 64), 512: (16, 32, 64, 128)},
    "1m": {512: (8, 16, 32, 64), 1024: (16, 32, 64, 128), 2048: (32, 64, 128, 256)},
}
# Combinations aimed at pynndescent (#56 follow-up), per suite: name -> (builder, options).
NOVEL = {
    "100k": {
        **{f"pca{d}-nnd-c{c}": ("pca_nndescent", dict(dims=d, candidates=c)) for d in (64, 128) for c in (30, 45)},
        **{f"pca{d}-ivf-L256-p{p}-c30": ("pca_ivf", dict(dims=d, candidates=30, lists=256, probes=p))
           for d in (64, 128) for p in (8, 16, 32)},
        **{f"ivfinit-L256-p{p}-nnd": ("ivf_init_nndescent", dict(lists=256, probes=p)) for p in (1, 2, 4)},
        **{f"pca{d}-nnd-c{c}": ("pca_nndescent", dict(dims=d, candidates=c)) for d, c in ((192, 30), (256, 30), (256, 45))},
    },
    "1m": {
        **{f"pca{d}-nnd-c{c}": ("pca_nndescent", dict(dims=d, candidates=c)) for d in (128, 256) for c in (30, 45)},
        **{f"pca128-ivf-L1024-p{p}-c30": ("pca_ivf", dict(dims=128, candidates=30, lists=1024, probes=p))
           for p in (8, 16)},
        "ivfinit-L1024-p1-nnd": ("ivf_init_nndescent", dict(lists=1024, probes=1)),
        **{f"pca{d}-nnd-c{c}": ("pca_nndescent", dict(dims=d, candidates=c)) for d, c in ((192, 30), (192, 40), (256, 40))},
    },
}
HNSW = {"hnsw-M8-efc100-ef64": dict(m=8, ef_construction=100, ef=64),
        "hnsw-M16-efc200-ef32": dict(m=16, ef_construction=200, ef=32)}


def configurations(suite: str) -> dict:
    configs = {name: partial(methods.pynndescent_graph, **options) for name, options in PYNNDESCENT.items()}
    for lists, probe_grid in IVF_FLAT_GRID[suite].items():
        for probes in probe_grid:
            configs[f"ivfflat-L{lists}-p{probes}"] = partial(methods.tieredq_ivf_flat, lists=lists, probes=probes)
    configs.update({name: partial(methods.hnswlib_graph, **options) for name, options in HNSW.items()})
    configs.update({name: partial(getattr(methods, builder), **options)
                    for name, (builder, options) in NOVEL[suite].items()})
    return configs


def truth(X: np.ndarray, out: Path, k: int, threads: int, sample: int):
    path = out / "truth.npz"
    if path.exists():
        saved = np.load(path)
        return saved["rows"], saved["ids"]
    rows = np.arange(len(X)) if not sample else np.sort(
        np.random.default_rng(TRUTH_SEED).choice(len(X), sample, replace=False))
    started = time.perf_counter()
    ids = methods.exact_rows(X, rows, k, threads)
    print(json.dumps({"truth_rows": len(rows), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    np.savez(path, rows=rows, ids=ids)
    return rows, ids


def run_one(args, name: str, seed: int) -> None:
    X = np.ascontiguousarray(np.load(args.data, mmap_mode="r"), dtype=np.float32)
    builder = configurations(args.suite)[name]
    if "nnd" in name:  # numba JIT: the same builder on a subset, so the timed call compiles nothing
        builder(np.ascontiguousarray(X[:NOVEL_WARMUP_ROWS]), args.k, args.threads, seed=seed)
    elif name.startswith("pynndescent"):
        builder(X[:WARMUP_ROWS], args.k, args.threads, seed=seed)
    methods.PHASES.clear()
    started = time.perf_counter()
    graph = builder(X, args.k, args.threads, seed=seed)
    seconds = time.perf_counter() - started
    rows, ids = truth(X, args.out, args.k, args.threads, args.truth_rows)
    row = {"name": name, "seed": seed, "seconds": round(seconds, 3),
           f"recall@{args.k}": round(methods.recall(graph[rows], ids), 5), "threads": args.threads, **methods.PHASES}
    with open(args.out / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def finished(out: Path) -> set:
    path = out / "results.jsonl"
    if not path.exists():
        return set()
    return {(r["name"], r["seed"]) for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--suite", choices=sorted(IVF_FLAT_GRID), required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--truth-rows", type=int, default=0)
    parser.add_argument("--only", default="")
    parser.add_argument("--worker", default="")
    parser.add_argument("--worker-seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_one(args, args.worker, args.worker_seed)
        return
    if not (args.out / "truth.npz").exists():
        truth(np.ascontiguousarray(np.load(args.data), dtype=np.float32), args.out, args.k, TRUTH_THREADS,
              args.truth_rows)
    wanted = [n for n in configurations(args.suite) if not args.only or n in args.only.split(",")]
    done = finished(args.out)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), NUMBA_NUM_THREADS=str(args.threads))
    for seed in map(int, args.seeds.split(",")):
        for name in wanted:
            if (name, seed) in done:
                continue
            command = ["taskset", "-c", f"0-{args.threads - 1}", sys.executable, __file__, str(args.data),
                       str(args.out), "--suite", args.suite, "--threads", str(args.threads), "--k", str(args.k),
                       "--truth-rows", str(args.truth_rows), "--worker", name, "--worker-seed", str(seed)]
            if subprocess.run(command, env=env).returncode:
                print(f"{name} seed {seed} failed", flush=True)


if __name__ == "__main__":
    main()
