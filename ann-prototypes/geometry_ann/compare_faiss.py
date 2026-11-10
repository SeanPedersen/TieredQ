# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu>=1.8", "numpy>=1.26"]
# ///
"""Faiss calibration with optional frozen-budget replay on independent queries.

Build may use several threads (--build-threads); every search runs single-threaded,
optionally pinned to one core (--search-cpu). Peak anonymous/file-backed RSS is sampled
during the build."""
import argparse
import csv
import json
import os
import shlex
import math
import time
from pathlib import Path
import resource
import sys

def requested_build_threads(argv: list[str]) -> int:
    # BLAS reads its thread count at import, so this must be known before importing faiss.
    for flag, value in zip(argv, argv[1:]):
        if flag == "--build-threads":
            return int(value)
    return 1


BUILD_THREADS = requested_build_threads(sys.argv)
for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_variable] = str(BUILD_THREADS)

import faiss
import numpy as np
from faiss_reference import K, build_reference, measure, select_targets
from memory_probe import MemorySampler, read_memory


def records(path: Path, dtype: str) -> np.ndarray:
    """Validate every row header before exposing the memory-mapped payload."""
    if path.suffix == ".rawview":
        magic, rows, dimension, offset, filename = shlex.split(path.read_text())
        rows, dimension, offset = int(rows), int(dimension), int(offset)
        backing = Path(filename)
        if not backing.is_absolute(): backing = path.parent / backing
        if magic != "geometry_raw_f32_v1" or dtype != "<f4" or min(rows, dimension) <= 0 or offset < 0 or offset % 4:
            raise ValueError("Invalid rawview descriptor")
        if offset + rows * dimension * 4 > backing.stat().st_size:
            raise ValueError("Rawview exceeds backing file")
        return np.memmap(backing, dtype="<f4", mode="r", offset=offset, shape=(rows, dimension))
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


def load_queries(queries_path: Path, truth_path: Path, count: int, base_shape: tuple[int,int]) -> tuple:
    queries = np.ascontiguousarray(records(queries_path, "<f4")[:count])
    truth = records(truth_path, "<i4")[:count, :K]
    if base_shape[0] < K or queries.shape[1] != base_shape[1] or truth.shape != (len(queries), K):
        raise ValueError("Incompatible dataset shapes")
    if not np.isfinite(queries).all() or np.any(truth < 0) or np.any(truth >= base_shape[0]):
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
    parser.add_argument("--evaluation-repeats", type=int, default=1)
    parser.add_argument("--targets", default="0.95,0.99")
    parser.add_argument("--int8-base",type=Path)
    parser.add_argument("--int8-scales",type=Path)
    parser.add_argument("--int8-offsets",type=Path)
    parser.add_argument("--build-threads", type=int, default=1)
    parser.add_argument("--search-cpu", type=int)
    parser.add_argument("--memory-output", type=Path)
    parser.add_argument("--centroids", type=Path, help="use these coarse centroids (nlist x dim float32) instead of training")
    parser.add_argument("--assignments", type=Path, help="with --centroids: one uint32 list label per row instead of quantizer search")
    args = parser.parse_args()
    if min(args.nq, args.nlist, args.sample, args.evaluation_nq, args.evaluation_repeats, args.build_threads) <= 0:
        parser.error("counts must be positive")
    targets = [float(t) for t in args.targets.split(",")]
    if not targets or any(not math.isfinite(t) or not 0 < t <= 1 for t in targets):
        parser.error("targets must be in (0,1]")
    if args.int8_offsets and not args.int8_base: parser.error("SQ8 offsets require codes")
    if args.assignments and not args.centroids:
        parser.error("--assignments requires --centroids")
    if args.centroids and (args.variant != "ivf" or not args.int8_base):
        parser.error("--centroids requires the flat-quantizer IVF-SQ8 variant")
    if bool(args.int8_base) != bool(args.int8_scales) or (args.int8_base and args.variant == "hnsw"):
        parser.error("Shared SQ8 requires codes, scales, and an IVF variant")
    evaluation = (args.evaluation_queries, args.evaluation_truth, args.evaluation_output)
    if any(evaluation) and not all(evaluation):
        parser.error("evaluation requires queries, truth, and output")
    if args.evaluation_output and args.evaluation_output.exists():
        raise FileExistsError(args.evaluation_output)
    faiss.omp_set_num_threads(args.build_threads)
    sampler = MemorySampler()
    base_owner = records(args.base, "<f4")
    base = np.ascontiguousarray(base_owner)
    base_bytes = base.nbytes
    base_shape = base.shape
    validation_start=time.perf_counter()
    if not args.int8_base:
        for start in range(0, len(base), 8192):
            if not np.isfinite(base[start:start + 8192]).all():
                raise ValueError("Nonfinite base")
    validation_seconds=time.perf_counter()-validation_start
    queries, truth = load_queries(args.queries, args.truth, args.nq, base_shape)

    index, budgets, build_seconds = build_reference(base, args)
    build_seconds += validation_seconds
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = int(peak if sys.platform == "darwin" else peak * 1024)
    build_memory = sampler.stop()
    del base
    if isinstance(base_owner, np.memmap): base_owner._mmap.close()
    del base_owner
    # Single-query searches with one OpenMP thread never reach the BLAS path (nq < 20).
    faiss.omp_set_num_threads(1)
    if args.search_cpu is not None: os.sched_setaffinity(0, {args.search_cpu})
    if args.memory_output:
        serving = read_memory()
        with args.memory_output.open("x") as output:
            json.dump(dict(build_s=build_seconds, build_threads=args.build_threads, nlist=args.nlist, **build_memory,
                           serving_rss=serving.get("VmRSS", 0), serving_anon=serving.get("RssAnon", 0),
                           serving_file=serving.get("RssFile", 0)), output, indent=2)
    if args.budgets:
        budgets = sorted({int(value) for value in args.budgets.split(",")})
        if not budgets or min(budgets) <= 0 or (args.variant != "hnsw" and max(budgets) > args.nlist):
            raise ValueError("Invalid replay budgets")
    calibration = [measure(index, args.variant, queries, truth, budget) for budget in budgets]
    selection = select_targets(calibration, targets)
    test_results = []
    if all(evaluation):
        frozen = {row["budget"] for row in selection.values() if row}
        diagnostics = []
        if any(row is None for row in selection.values()):
            diagnostic=max(calibration,key=lambda row:(row["recall10"],-row["p50_us"]))["budget"]
            frozen.add(diagnostic);diagnostics.append(diagnostic)
        frozen=sorted(frozen)
        with args.evaluation_output.with_suffix(".selection.json").open("x") as output:
            json.dump(dict(faiss_version=faiss.__version__, selection=selection, targets=targets, nlist=args.nlist,
                           sample=args.sample, seed=args.seed, diagnostic_budgets=diagnostics, calibration_queries=str(args.queries),
                           test_queries=str(args.evaluation_queries)), output, indent=2)
            output.write("\n")
        test = load_queries(args.evaluation_queries,args.evaluation_truth,args.evaluation_nq,base_shape)
        for repeat in range(args.evaluation_repeats):
            for budget in frozen:
                before = resource.getrusage(resource.RUSAGE_SELF).ru_majflt
                row = measure(index, args.variant, *test, budget)
                if args.evaluation_repeats > 1:
                    row = dict(repeat=repeat, **row)
                    row["major_faults"] = resource.getrusage(resource.RUSAGE_SELF).ru_majflt - before
                    memory = {}
                    try:
                        for line in Path("/proc/self/status").read_text().splitlines():
                            if line.startswith(("VmRSS:","VmSwap:")):
                                key, kb, _ = line.split(); memory[key.rstrip(":")] = int(kb) * 1024
                    except FileNotFoundError: pass
                    row["serving_rss_bytes"] = memory.get("VmRSS",0)
                    row["swap_bytes"] = memory.get("VmSwap",0)
                test_results.append(row)
    if args.evaluation_output:
        with Path(str(args.evaluation_output)+".observations.json").open("x") as checkpoint:
            json.dump(dict(calibration=calibration,heldout=test_results,build_s=build_seconds,peak_build_rss_bytes=peak),checkpoint,indent=2)
    # Serialization happens after ALL timed queries, avoiding its large temporary buffers during warmup.
    # Avoid a second full index allocation for large corpora. Faiss's native
    # writer counts actual serialized bytes with a bounded streaming callback.
    if base_bytes > (1 << 30):
        # The callback's chunks are bounded by the writer's streaming buffer.
        total = [0]
        def count_bytes(data):
            total[0] += len(data)
            return len(data)
        faiss.write_index(index, faiss.PyCallbackIOWriter(count_bytes))
        serialized_bytes = total[0]
    else:
        serialized_bytes = faiss.serialize_index(index).nbytes
    stored_bytes = base_bytes // 4 if args.int8_base else base_bytes
    common = dict(variant="ivf-sq8" if args.int8_base else args.variant, build_s=build_seconds, raw_bytes=stored_bytes,
                  serialized_overhead_bytes=serialized_bytes - stored_bytes, process_peak_rss_bytes=peak,
                  vector_storage="int8_sq_per_dim" if args.int8_base else "fp32", original_fp32_bytes=base_bytes, validation_s=validation_seconds)
    write_results(sys.stdout, calibration, common)
    if test_results:
        with args.evaluation_output.open("x", newline="") as output:
            write_results(output, test_results, common)
    print(f"Calibration selections: {selection}", file=sys.stderr)


if __name__ == "__main__":
    main()
