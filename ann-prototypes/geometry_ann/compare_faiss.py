# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu>=1.8", "numpy>=1.26"]
# ///
"""Single-thread Faiss calibration with optional frozen-budget replay on independent queries."""
import argparse
import csv
import json
import os
from pathlib import Path
import resource
import sys

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_variable] = "1"

import faiss
import numpy as np
from faiss_reference import K, build_reference, measure, select_targets


def records(path: Path, dtype: str) -> np.ndarray:
    """Validate every row header before exposing the memory-mapped payload."""
    raw = np.memmap(path, dtype="<i4", mode="r")
    if not raw.size or raw[0] <= 0:
        raise ValueError(f"Invalid vecs file: {path}")
    width = int(raw[0])
    if raw.size % (width + 1):
        raise ValueError(f"Truncated vecs file: {path}")
    rows = raw.reshape(-1, width + 1)
    if not np.all(rows[:, 0] == width):
        raise ValueError(f"Inconsistent dimensions: {path}")
    return rows[:, 1:].view(dtype)


def load_queries(queries_path: Path, truth_path: Path, count: int, base: np.ndarray) -> tuple:
    queries = np.ascontiguousarray(records(queries_path, "<f4")[:count])
    truth = records(truth_path, "<i4")[:count, :K]
    if len(base) < K or queries.shape[1] != base.shape[1] or truth.shape != (len(queries), K):
        raise ValueError("Incompatible dataset shapes")
    if not np.isfinite(queries).all() or np.any(truth < 0) or np.any(truth >= len(base)):
        raise ValueError("Invalid coordinates or ground truth")
    return queries, truth


def write_results(output, observations: list[dict], common: dict) -> None:
    writer = csv.DictWriter(output, fieldnames=list(common) + list(observations[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(dict(common, **row) for row in observations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("base", "queries", "truth"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--nq", type=int, default=100)
    parser.add_argument("--nlist", type=int, default=512)
    parser.add_argument("--sample", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=("ivf", "ivf-hnsw", "hnsw"), required=True)
    parser.add_argument("--budgets", help="comma-separated fixed nprobe/efSearch values")
    parser.add_argument("--evaluation-queries", type=Path)
    parser.add_argument("--evaluation-truth", type=Path)
    parser.add_argument("--evaluation-output", type=Path)
    parser.add_argument("--evaluation-nq", type=int, default=1000)
    args = parser.parse_args()
    if min(args.nq, args.nlist, args.sample, args.evaluation_nq) <= 0:
        parser.error("counts must be positive")
    evaluation = (args.evaluation_queries, args.evaluation_truth, args.evaluation_output)
    if any(evaluation) and not all(evaluation):
        parser.error("evaluation requires queries, truth, and output")
    if args.evaluation_output and args.evaluation_output.exists():
        raise FileExistsError(args.evaluation_output)
    faiss.omp_set_num_threads(1)
    base = np.ascontiguousarray(records(args.base, "<f4"))
    if not np.isfinite(base).all():
        raise ValueError("Nonfinite base")
    queries, truth = load_queries(args.queries, args.truth, args.nq, base)
    test = load_queries(args.evaluation_queries, args.evaluation_truth, args.evaluation_nq, base) if all(evaluation) else None
    index, budgets, build_seconds = build_reference(base, args)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak if sys.platform == "darwin" else peak * 1024)
    if args.budgets:
        budgets = sorted({int(value) for value in args.budgets.split(",")})
        if not budgets or min(budgets) <= 0 or (args.variant != "hnsw" and max(budgets) > args.nlist):
            raise ValueError("Invalid replay budgets")
    calibration = [measure(index, args.variant, queries, truth, budget) for budget in budgets]
    selection = select_targets(calibration)
    test_results = []
    if test:
        frozen = sorted({row["budget"] for row in selection.values() if row})
        test_results = [measure(index, args.variant, *test, budget) for budget in frozen]
    # Serialization happens after ALL timed queries, avoiding its large temporary buffers during warmup.
    serialized_bytes = faiss.serialize_index(index).nbytes
    common = dict(variant=args.variant, build_s=build_seconds, raw_bytes=base.nbytes,
                  serialized_overhead_bytes=serialized_bytes - base.nbytes, process_peak_rss_bytes=peak)
    write_results(sys.stdout, calibration, common)
    if test_results:
        with args.evaluation_output.open("x", newline="") as output:
            write_results(output, test_results, common)
        with args.evaluation_output.with_suffix(".selection.json").open("x") as output:
            json.dump(dict(faiss_version=faiss.__version__, selection=selection, nlist=args.nlist, sample=args.sample,
                           seed=args.seed, calibration_queries=str(args.queries), test_queries=str(args.evaluation_queries)), output, indent=2)
            output.write("\n")
    print(f"Calibration selections: {selection}", file=sys.stderr)


if __name__ == "__main__":
    main()
