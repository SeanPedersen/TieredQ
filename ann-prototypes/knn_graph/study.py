"""Issue #56 study: wall time vs recall@k of all-points k-NN graph builders.

    uv run study.py DATA.npy OUT_DIR [--threads 8] [--k 15] [--only name,...]

Each configuration runs in a fresh process (no shared page cache of an earlier index, no
numba JIT in the timing: pynndescent warms up on 2,000 rows first). Workers are pinned to
CPUs 0..threads-1 with as many OpenMP threads. Results append to
OUT_DIR/results.jsonl; finished configurations are skipped, so reruns resume. The exact
graph is computed first and is the recall reference.
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



def configurations() -> dict:
    configs = {"exact": methods.exact, "pynndescent": methods.pynndescent_graph}
    for m, ef_construction, efs in ((8, 100, (16, 32, 64)), (16, 200, (32, 64))):
        for ef in efs:
            configs[f"hnsw-M{m}-efc{ef_construction}-ef{ef}"] = partial(
                methods.hnswlib_graph, m=m, ef_construction=ef_construction, ef=ef
            )
    for probes in (4, 8, 16, 32):
        configs[f"ivfflat-L1024-p{probes}"] = partial(methods.faiss_ivf_flat, lists=1024, probes=probes)
    for lists, probe_grid in ((256, (2, 4, 8, 16)), (1024, (4, 8, 16, 32, 64))):
        for probes in probe_grid:
            configs[f"cjoin-L{lists}-p{probes}"] = partial(methods.cluster_join, lists=lists, probes=probes)
    for bits in (2, 4):
        for probes in (8, 16, 32, 64):
            for list_sorted in (False, True):
                name = f"tq-ivf-b{bits}-L1024-p{probes}" + ("-sorted" if list_sorted else "")
                configs[name] = partial(methods.tieredq_ivf, bits=bits, lists=1024, probes=probes,
                                        list_sorted=list_sorted)
    for probes in (8, 16, 32, 64):
        configs[f"tq-nested-L1024-p{probes}"] = partial(methods.tieredq_nested, lists=1024, probes=probes)
    for n_trees, n_iters, max_candidates in ((16, 20, 30), (32, 30, 60)):
        configs[f"pynndescent-t{n_trees}-i{n_iters}-c{max_candidates}"] = partial(
            methods.pynndescent_graph, n_trees=n_trees, n_iters=n_iters, max_candidates=max_candidates
        )
    for lists, probe_grid in ((64, (4, 8, 16)), (128, (4, 8, 16, 32)), (256, (4, 8, 16, 32, 64)),
                              (1024, (16, 32, 64, 128))):
        for probes in probe_grid:
            configs[f"ivfjoin-L{lists}-p{probes}"] = partial(methods.ivf_join, lists=lists, probes=probes)
    for base in ("cjoin-L256-p4", "cjoin-L256-p8", "cjoin-L1024-p8", "cjoin-L1024-p16", "tq-ivf-b4-L1024-p8"):
        for rounds in (1, 2):
            configs[f"{base}+r{rounds}"] = methods.refined(configs[base], rounds=rounds)
    return configs


def run_one(name: str, data: Path, out: Path, k: int, threads: int) -> None:
    X = np.ascontiguousarray(np.load(data), dtype=np.float32)
    builder = configurations()[name]
    if name.startswith("pynndescent"):  # numba JIT
        builder(X[:WARMUP_ROWS], k, threads)
    if "+r" in name:
        warm = X[:WARMUP_ROWS]
        methods.refine(warm, np.random.default_rng(0).integers(0, len(warm), (len(warm), k)), 1, 1)
    start = time.perf_counter()
    graph = builder(X, k, threads)
    seconds = time.perf_counter() - start
    np.save(out / f"{name}.npy", graph.astype(np.int32))
    truth_path = out / "exact.npy"
    score = methods.recall(graph, np.load(truth_path)) if truth_path.exists() else 1.0
    row = {"name": name, "seconds": round(seconds, 3), f"recall@{k}": round(score, 5), "threads": threads, **methods.PHASES}
    with open(out / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def finished(out: Path) -> set[str]:
    path = out / "results.jsonl"
    return {json.loads(line)["name"] for line in path.read_text().splitlines()} if path.exists() else set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--only", default="")
    parser.add_argument("--worker", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_one(args.worker, args.data, args.out, args.k, args.threads)
        return
    wanted = [n for n in configurations() if not args.only or n in args.only.split(",")]
    done = finished(args.out)
    # exact first: it is the recall reference for every later row
    for name in sorted(wanted, key=lambda n: n != "exact"):
        if name in done:
            continue
        cpus = f"0-{args.threads - 1}"
        base = ["taskset", "-c", cpus, sys.executable, __file__, str(args.data), str(args.out)]
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), NUMBA_NUM_THREADS=str(args.threads))
        result = subprocess.run([*base, "--threads", str(args.threads), "--k", str(args.k), "--worker", name], env=env)
        if result.returncode:
            print(f"{name} failed: {result.returncode}", flush=True)


if __name__ == "__main__":
    main()
