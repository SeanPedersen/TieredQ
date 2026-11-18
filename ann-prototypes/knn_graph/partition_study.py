"""Issue #45 follow-up: do e-means (Faiss contrib) or s-means lists beat our Lloyd k-means for TieredQIVFFlatIndex?

    uv run partition_study.py DATA.npy OUT_DIR --lists 1024 [--probes 8,16,32] [--seeds 0,1] [--truth-rows 10000]
                              [--emeans-path VENDOR_DIR] [--only name,...]

Every trainer clusters a seeded sample of rows_per_list x lists rows into the same number of lists. Lloyd and the
native e-means port (cpp-emeans-*) run inside the index (Lloyd's default recipe is the baseline); the reference e-means
(Faiss contrib, NumPy) and s-means centroids enter through set_centroids(), so all of them share the index's exact
assignment and knn_graph join. A trainer's budget is its sample size times its passes:
Lloyd at 64 rows per list x 10 iterations equals e-means at 256 x 2.5 epochs or 640 x 1.

Per (trainer, seed), in a fresh process pinned to CPUs 0..threads-1: train_s, the k-means objective of all rows on
the final lists, list-size imbalance (lists x sum(size^2) / rows^2, 1 = balanced: it sets the join's cost), and for
every nprobe the knn_graph join time and recall@k. Recall is measured as in compare.py (shared truth.npz).
Results append to OUT_DIR/partitions.jsonl; finished (name, seed) pairs are skipped.

With --queries Q.fvecs (e.g. the MS MARCO calibration and test queries, which come from another distribution than
the passages), the join is replaced by search: per nprobe, recall@k of the queries against exact neighbours over all
rows (cached in OUT_DIR/query_truth.npz) and the rows scanned per query (the probed lists' sizes, a deterministic
cost). Those results append to OUT_DIR/queries.jsonl.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import methods
from compare import TRUTH_THREADS, truth

SAMPLE_SALT = 45
OBJECTIVE_CHUNK_ROWS = 65536
QUERY_CHUNK_ROWS = 256

# name -> (trainer, rows per list, passes over the sample)
TRAINERS = {
    "lloyd-s64-i10": ("lloyd", 64, 10),  # the index default
    "lloyd-s256-i3": ("lloyd", 256, 3),  # about the default budget on 4x the rows
    "lloyd-s256-i10": ("lloyd", 256, 10),  # 4x Lloyd's budget: what more data alone buys
    "emeans-s64-e10": ("emeans", 64, 10),
    "emeans-s256-e2.5": ("emeans", 256, 2.5),
    "emeans-s640-e1": ("emeans", 640, 1),
    "emeans-s256-e10": ("emeans", 256, 10),
    "smeans-s64-e10": ("smeans", 64, 10),
    "smeans-s256-e2.5": ("smeans", 256, 2.5),
    "smeans-s640-e1": ("smeans", 640, 1),
    "smeans-s256-e10": ("smeans", 256, 10),
    "cpp-emeans-s256-e2.5": ("cpp-emeans", 256, 2.5),
    "cpp-emeans-s640-e1": ("cpp-emeans", 640, 1),
}
NATIVE = {"lloyd", "cpp-emeans"}  # trained inside the index


def sample(X: np.ndarray, rows: int, seed: int) -> np.ndarray:
    picked = np.sort(np.random.default_rng(seed * 1000 + SAMPLE_SALT).choice(len(X), min(rows, len(X)), replace=False))
    return np.ascontiguousarray(X[picked])


def emeans_centroids(rows: np.ndarray, lists: int, epochs: float, seed: int, vendor: Path) -> np.ndarray:
    sys.path.insert(0, str(vendor))
    from e_means import NumpyEmeans

    model = NumpyEmeans(rows.shape[1], lists, nepochs=epochs, seed=seed, capture_assignments=False)
    centroids, _ = model.train(rows)
    return centroids


def smeans_centroids(rows: np.ndarray, lists: int, epochs: float, seed: int, threads: int) -> np.ndarray:
    import torch
    from smeans import smeans

    torch.set_num_threads(threads)
    return smeans(rows, n_clusters=lists, nepochs=epochs, seed=seed, device="cpu", compute_labels=False).numpy()


def partition_quality(X: np.ndarray, index) -> dict:
    """List-size statistics of the stored partition."""
    labels = np.asarray(index.list_assignment())
    sizes = np.bincount(labels, minlength=index.lists)
    return {"imbalance": round(float(index.lists * (sizes.astype(np.float64) ** 2).sum() / len(X) ** 2), 4),
            "empty_lists": int((sizes == 0).sum()), "largest_list": int(sizes.max())}


def objective(X: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared distances of all rows to their list's centroid."""
    total = 0.0
    for start in range(0, len(X), OBJECTIVE_CHUNK_ROWS):
        part = X[start : start + OBJECTIVE_CHUNK_ROWS]
        total += float(((part - centroids[labels[start : start + len(part)]]) ** 2).sum(dtype=np.float64))
    return total


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    width = int(raw[0])
    return np.ascontiguousarray(raw.reshape(-1, width + 1)[:, 1:].view(np.float32))


def load_queries(paths: str) -> np.ndarray:
    return np.concatenate([read_fvecs(Path(path)) for path in paths.split(",")])


def query_truth(X: np.ndarray, queries: np.ndarray, out: Path, k: int) -> np.ndarray:
    """Exact k nearest rows (L2) of every query over all of X."""
    path = out / "query_truth.npz"
    if path.exists():
        return np.load(path)["ids"]
    norms = np.einsum("ij,ij->i", X, X)
    ids = np.empty((len(queries), k), np.int64)
    for start in range(0, len(queries), QUERY_CHUNK_ROWS):
        part = queries[start : start + QUERY_CHUNK_ROWS]
        distances = norms[None, :] - 2 * part @ X.T
        top = np.argpartition(distances, k, axis=1)[:, :k]
        order = np.argsort(np.take_along_axis(distances, top, axis=1), axis=1)
        ids[start : start + len(part)] = np.take_along_axis(top, order, axis=1)
    np.savez(path, ids=ids)
    return ids


def scanned_rows(index, queries: np.ndarray, nprobe: int) -> float:
    """Mean rows per query in its nprobe nearest lists, ranked as the index routes (q.c - |c|^2 / 2)."""
    centroids = index.centroids()
    sizes = np.bincount(np.asarray(index.list_assignment()), minlength=index.lists)
    closeness = queries @ centroids.T - 0.5 * np.einsum("ij,ij->i", centroids, centroids)[None, :]
    probes = np.argpartition(-closeness, nprobe - 1, axis=1)[:, :nprobe]
    return float(sizes[probes].sum(axis=1).mean())


def query_results(args, index, X: np.ndarray) -> dict:
    queries = load_queries(args.queries)
    truth_ids = query_truth(X, queries, args.out, args.k)
    row = {}
    for nprobe in map(int, args.probes.split(",")):
        started = time.perf_counter()
        _, ids = index.search(queries, args.k, nprobe=nprobe)
        row[f"p{nprobe}_search_s"] = round(time.perf_counter() - started, 4)
        row[f"p{nprobe}_scanned"] = round(scanned_rows(index, queries, nprobe), 1)
        row[f"p{nprobe}_recall"] = round(methods.recall(ids.astype(np.int64), truth_ids), 5)
    return row


def graph_results(args, index, X: np.ndarray) -> dict:
    truth_rows, truth_ids = truth(X, args.out, args.k, args.threads, args.truth_rows)
    row = {}
    for nprobe in map(int, args.probes.split(",")):
        started = time.perf_counter()
        _, ids = index.knn_graph(args.k, nprobe=nprobe)
        row[f"p{nprobe}_join_s"] = round(time.perf_counter() - started, 3)
        row[f"p{nprobe}_recall"] = round(methods.recall(ids.astype(np.int64)[truth_rows], truth_ids), 5)
    return row


def run_one(args, name: str, seed: int) -> None:
    from tieredq import TieredQIVFFlatIndex

    X = np.ascontiguousarray(np.load(args.data, mmap_mode="r"), dtype=np.float32)
    trainer, per_list, passes = TRAINERS[name]
    options = (dict(trainer="lloyd", iterations=int(passes)) if trainer == "lloyd"
               else dict(trainer="emeans", epochs=float(passes)) if trainer == "cpp-emeans" else {})
    index = TieredQIVFFlatIndex(X.shape[1], lists=args.lists, metric="l2", seed=seed, sample_rows_per_list=per_list,
                                **options)
    started = time.perf_counter()
    if trainer not in NATIVE:
        rows = sample(X, per_list * args.lists, seed)
        centroids = (emeans_centroids(rows, args.lists, passes, seed, args.emeans_path) if trainer == "emeans"
                     else smeans_centroids(rows, args.lists, passes, seed, args.threads))
        index.set_centroids(np.ascontiguousarray(centroids, dtype=np.float32))
        del rows
    train_s = time.perf_counter() - started
    index.build(X)
    stats = index.build_stats
    row = {"name": name, "seed": seed, "lists": index.lists, "train_s": round(train_s + stats["train_s"], 3),
           "assign_s": round(stats["assign_s"], 3), **partition_quality(X, index)}
    labels = np.asarray(index.list_assignment())
    row["objective"] = objective(X, list_means(X, labels, index.lists), labels)
    row.update(query_results(args, index, X) if args.queries else graph_results(args, index, X))
    with open(args.out / results_file(args), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def list_means(X: np.ndarray, labels: np.ndarray, lists: int) -> np.ndarray:
    """Each list's mean row: the objective of the partition itself, independent of how centroids were rounded."""
    sums = np.zeros((lists, X.shape[1]), np.float64)
    for start in range(0, len(X), OBJECTIVE_CHUNK_ROWS):
        part_labels = labels[start : start + OBJECTIVE_CHUNK_ROWS]
        order = np.argsort(part_labels, kind="stable")
        present, first = np.unique(part_labels[order], return_index=True)
        sums[present] += np.add.reduceat(X[start : start + OBJECTIVE_CHUNK_ROWS][order], first, axis=0)
    counts = np.maximum(np.bincount(labels, minlength=lists), 1)
    return (sums / counts[:, None]).astype(np.float32)


def results_file(args) -> str:
    return "queries.jsonl" if args.queries else "partitions.jsonl"


def finished(args) -> set:
    path = args.out / results_file(args)
    if not path.exists():
        return set()
    return {(r["name"], r["seed"]) for r in map(json.loads, path.read_text().splitlines())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--lists", type=int, required=True)
    parser.add_argument("--probes", default="8,16,32")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--truth-rows", type=int, default=0)
    parser.add_argument("--emeans-path", type=Path, default=Path("vendor"))
    parser.add_argument("--queries", default="", help="comma-separated .fvecs files: search recall instead of the join")
    parser.add_argument("--only", default="")
    parser.add_argument("--worker", default="")
    parser.add_argument("--worker-seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        run_one(args, args.worker, args.worker_seed)
        return
    if not args.queries and not (args.out / "truth.npz").exists():
        truth(np.ascontiguousarray(np.load(args.data), dtype=np.float32), args.out, args.k, TRUTH_THREADS,
              args.truth_rows)
    wanted = args.only.split(",") if args.only else list(TRAINERS)
    unknown = set(wanted) - set(TRAINERS)
    if unknown:
        parser.error(f"unknown trainers: {sorted(unknown)}")
    done = finished(args)
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads))
    for seed in map(int, args.seeds.split(",")):
        for name in wanted:
            if (name, seed) in done:
                continue
            command = ["taskset", "-c", f"0-{args.threads - 1}", sys.executable, __file__, str(args.data),
                       str(args.out), "--lists", str(args.lists), "--probes", args.probes, "--threads",
                       str(args.threads), "--k", str(args.k), "--truth-rows", str(args.truth_rows), "--emeans-path",
                       str(args.emeans_path), "--queries", args.queries, "--worker", name, "--worker-seed", str(seed)]
            if subprocess.run(command, env=env).returncode:
                print(f"{name} seed {seed} failed", flush=True)


if __name__ == "__main__":
    main()
