# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu>=1.8", "numpy>=1.26"]
# ///
"""Train a flat partition on decoded SQ8 rows and write one uint32 leaf label per row.

Methods: `lloyd` (faiss.Kmeans, as in Faiss IVF training) and `emeans` (mini-batch E-means, arXiv 2611.09501).
Both train on the same `--sample-per-centroid` rows per centroid (same rows for the same seed); 0 streams every
row through E-means. Both end with one exact assignment pass of every row against the final centroids.
Labels feed `graph_bench --assignments`.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def requested_threads(argv: list[str]) -> int:
    # BLAS reads its thread count at import, so this must be known before importing faiss.
    for flag, value in zip(argv, argv[1:]):
        if flag == "--threads":
            return int(value)
    return 1


THREADS = requested_threads(sys.argv)
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = str(THREADS)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
from memory_probe import MemorySampler  # noqa: E402

DECODE_CHUNK = 65536  # rows decoded at once: 256 MiB of FP32 at d=1024


class DecodedRows:
    """Per-dimension affine SQ8 decoding of selected rows; never materializes the FP32 corpus."""

    def __init__(self, codes: Path, scales: Path, offsets: Path, rows: int, dimension: int):
        self.codes = np.memmap(codes, dtype="int8", mode="r", shape=(rows, dimension))
        self.scales = np.fromfile(scales, dtype="<f4")
        self.offsets = np.fromfile(offsets, dtype="<f4")
        if self.scales.shape != (dimension,) or self.offsets.shape != (dimension,):
            raise ValueError("scales/offsets must have one value per dimension")
        self.rows, self.dimension = rows, dimension

    def take(self, index: np.ndarray) -> np.ndarray:
        decoded = self.codes[index].astype(np.float32)
        decoded *= self.scales
        decoded += self.offsets
        return decoded

    def chunks(self, index: np.ndarray):
        for begin in range(0, len(index), DECODE_CHUNK):
            yield self.take(index[begin:begin + DECODE_CHUNK])


def initial_rows(rows: int, k: int, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).choice(rows, k, replace=False))


def training_sample(rows: int, k: int, per_centroid: int, seed: int) -> np.ndarray:
    """Identical training rows for every method with the same seed (sorted for sequential memory-mapped reads)."""
    return np.sort(np.random.default_rng(seed).choice(rows, min(rows, per_centroid * k), replace=False))


def train_lloyd(data: DecodedRows, k: int, iterations: int, per_centroid: int, seed: int) -> tuple[np.ndarray, dict]:
    sample = training_sample(data.rows, k, per_centroid, seed)
    kmeans = faiss.Kmeans(data.dimension, k, niter=iterations, seed=seed, max_points_per_centroid=per_centroid, verbose=True)
    kmeans.train(data.take(sample))
    return kmeans.centroids, dict(sample=len(sample), iterations=iterations, objective=[float(v) for v in kmeans.obj])


def load_emeans(emeans_path: Path):
    sys.path.insert(0, str(emeans_path))
    from e_means import NumpyEmeans  # vendored pure-NumPy E-means (not yet in the faiss-cpu wheel)
    return NumpyEmeans


def train_emeans_sample(data: DecodedRows, k: int, epochs: float, per_centroid: int, seed: int, emeans_path: Path) -> tuple[np.ndarray, dict]:
    """E-means on the same in-memory sample Lloyd uses, through its own default `train` path."""
    sample = training_sample(data.rows, k, per_centroid, seed)
    model = load_emeans(emeans_path)(data.dimension, k, nepochs=epochs, seed=seed, verbose=True, capture_assignments=False)
    centroids, _ = model.train(data.take(sample))
    recipe = {key: value for key, value in model.recipe.items() if isinstance(value, (int, float, str, list))}
    return centroids, dict(sample=len(sample), recipe=recipe, splits=int(model.n_splits),
                           objective=[float(s["obj"]) for s in model.iteration_stats])


def train_emeans(data: DecodedRows, k: int, epochs: float, seed: int, emeans_path: Path) -> tuple[np.ndarray, dict]:
    """E-means streamed over every row (decoded chunk by chunk)."""
    model = load_emeans(emeans_path)(data.dimension, k, nepochs=epochs, seed=seed, verbose=True, capture_assignments=False)
    model.begin(data.rows, data.take(initial_rows(data.rows, k, seed)))
    batch = model.recipe["batch_size"]
    for epoch, epoch_rows in enumerate(model.recipe["epoch_rows"]):
        order = np.random.default_rng(seed + 1 + epoch).permutation(data.rows)[:epoch_rows]
        for begin in range(0, epoch_rows, batch):
            # Batch statistics are sums over rows against one centroid snapshot: row order within a batch
            # is irrelevant, and sorting keeps memory-mapped reads sequential.
            rows = np.sort(order[begin:begin + batch])
            stats = None
            for decoded in data.chunks(rows):
                stats = model.accumulate(decoded)
            model.update(stats)
        model.end_epoch()
    centroids = model.finalize()
    recipe = {key: value for key, value in model.recipe.items() if isinstance(value, (int, float, str, list))}
    return centroids, dict(recipe=recipe, splits=int(model.n_splits), objective=[float(s["obj"]) for s in model.iteration_stats])


def assign_all(data: DecodedRows, centroids: np.ndarray) -> tuple[np.ndarray, float]:
    index = faiss.IndexFlatL2(data.dimension)
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    labels = np.empty(data.rows, dtype=np.uint32)
    objective = 0.0
    for begin in range(0, data.rows, DECODE_CHUNK):
        end = min(begin + DECODE_CHUNK, data.rows)
        distances, nearest = index.search(data.take(np.arange(begin, end)), 1)
        labels[begin:end] = nearest[:, 0]
        objective += float(distances.sum(dtype=np.float64))
    return labels, objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("codes", "scales", "offsets", "labels-output", "report"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--method", choices=("lloyd", "emeans"), required=True)
    parser.add_argument("--iterations", type=int, default=4, help="Lloyd iterations")
    parser.add_argument("--sample-per-centroid", type=int, default=40,
                        help="training rows per centroid for both methods; 0 streams all rows (emeans only)")
    parser.add_argument("--epochs", type=float, default=5, help="E-means passes over all rows")
    parser.add_argument("--emeans-path", type=Path, help="directory containing the vendored e_means package")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--centroids-output", type=Path, help="also write the trained centroids (k x dim float32)")
    parser.add_argument("--assign", choices=("exact", "none"), default="exact",
                        help="none skips the exact assignment pass (labels then come from fast_assign)")
    args = parser.parse_args()
    if args.labels_output.exists() or args.report.exists() or (args.centroids_output and args.centroids_output.exists()):
        raise FileExistsError("refusing to overwrite partition outputs")
    if args.assign == "none" and not args.centroids_output:
        parser.error("--assign none requires --centroids-output")
    if args.method == "emeans" and not args.emeans_path:
        parser.error("--emeans-path is required for emeans")
    if args.method == "lloyd" and not args.sample_per_centroid:
        parser.error("lloyd requires a positive --sample-per-centroid")
    faiss.omp_set_num_threads(args.threads)
    data = DecodedRows(args.codes, args.scales, args.offsets, args.rows, args.dim)
    sampler = MemorySampler()
    start = time.perf_counter()
    if args.method == "lloyd":
        centroids, details = train_lloyd(data, args.k, args.iterations, args.sample_per_centroid, args.seed)
    elif args.sample_per_centroid:
        centroids, details = train_emeans_sample(data, args.k, args.epochs, args.sample_per_centroid, args.seed, args.emeans_path)
    else:
        centroids, details = train_emeans(data, args.k, args.epochs, args.seed, args.emeans_path)
    trained = time.perf_counter()
    if args.centroids_output:
        np.ascontiguousarray(centroids, dtype="<f4").tofile(args.centroids_output)
    if args.assign == "none":
        report = dict(method=args.method, k=args.k, rows=args.rows, threads=args.threads, seed=args.seed, train_s=trained - start,
                      assign_s=None, total_s=trained - start, faiss_version=faiss.__version__, details=details, **sampler.stop())
        args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(json.dumps({key: report[key] for key in ("method", "train_s")}))
        return
    labels, objective = assign_all(data, centroids)
    finished = time.perf_counter()
    labels.tofile(args.labels_output)
    used = np.bincount(labels, minlength=args.k)
    report = dict(method=args.method, k=args.k, rows=args.rows, threads=args.threads, seed=args.seed,
                  train_s=trained - start, assign_s=finished - trained, total_s=finished - start,
                  assignment_objective=objective, empty_clusters=int((used == 0).sum()),
                  largest_cluster=int(used.max()), faiss_version=faiss.__version__, details=details, **sampler.stop())
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({key: report[key] for key in ("method", "train_s", "assign_s", "assignment_objective", "empty_clusters")}))


if __name__ == "__main__":
    main()
