"""One L2 scan-cap measurement (#44 finding, centroidq): TieredQ's automatic max_scan_bits is 8 for L2, which sends
single queries to the generic scan; a 4-bit cap keeps them in the nibble-lookup kernel. Builds one index on BUILD_CORES,
measures held-out recall@10 with one batch call on those threads, then pins itself to QUERY_CORE with one thread and
times single queries (p50 / p99 / QPS).

- flat: TieredQIndex (fast mode, full scan), code_bits = bits_per_dim * d, metric l2. A seed > 0 permutes the rows
  before add(), which trains the codec on a different sample (the binding's sample seed is fixed).
- ivf: TieredQIVFIndex, metric l2, 2,048 lists (the automatic count at 1M); a seed > 0 trains the lists on a sample
  drawn with it.

Usage: BUILD_CORES=.. QUERY_CORE=2 l2_scan_cap.py DATASET_DIR OUT.json KIND BITS_PER_DIM MAX_SCAN_BITS SEED [NPROBES]
DATASET_DIR holds base.fvecs (or base.i8 + scales.f32 + offsets.f32 and ROWS), test_queries.fvecs, truth.ivecs."""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

K = 10
TIMED_QUERIES = 200
MIN_LIST_SAMPLE, SAMPLE_ROWS_PER_LIST = 32768, 64
AUTOMATIC_LISTS_1M = 2048  # TieredQIVFIndex::automatic_lists(1M) at 86f003d; given explicitly, as train_lists needs it


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    width = raw[0]
    table = raw.reshape(-1, width + 1)[:, 1:]
    return np.ascontiguousarray(table.view(dtype) if dtype != np.int32 else table)


def load_base(directory: Path):
    if (directory / "base.fvecs").exists():
        return read_vecs(directory / "base.fvecs", np.float32)
    rows = int(os.environ["ROWS"])
    scales = np.fromfile(directory / "scales.f32", dtype=np.float32)
    offsets = np.fromfile(directory / "offsets.f32", dtype=np.float32)
    codes = np.memmap(directory / "base.i8", dtype=np.int8, mode="r", shape=(rows, len(scales)))
    return np.ascontiguousarray(codes.astype(np.float32) * scales + offsets)


def single_thread(core):
    for library, setter in (("libgomp.so.1", "omp_set_num_threads"), ("libopenblas.so.0", "openblas_set_num_threads")):
        try:
            getattr(ctypes.CDLL(library), setter)(1)
        except OSError:
            pass
    os.sched_setaffinity(0, {core})


def recall(found, truth):
    return float(np.mean([len(set(row[:K]) & set(true[:K])) for row, true in zip(found, truth)])) / K


def timed(search, queries):
    latencies = []
    for query in queries[:TIMED_QUERIES]:
        started = time.perf_counter()
        search(query[None, :])
        latencies.append(time.perf_counter() - started)
    latencies = np.array(latencies) * 1e3
    return dict(p50_ms=float(np.percentile(latencies, 50)), p99_ms=float(np.percentile(latencies, 99)),
                qps=float(len(latencies) / latencies.sum() * 1e3))


def main():
    directory, out = Path(sys.argv[1]), Path(sys.argv[2])
    kind, bits, cap, seed = sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    nprobes = json.loads(sys.argv[7]) if len(sys.argv) > 7 else []
    from tieredq import TieredQIndex, TieredQIVFIndex

    os.sched_setaffinity(0, {int(c) for c in os.environ["BUILD_CORES"].split(",")})
    base = load_base(directory)
    queries = read_vecs(directory / "test_queries.fvecs", np.float32)
    truth = read_vecs(directory / "truth.ivecs", np.int32)[:, :K]
    order = np.random.default_rng(seed).permutation(len(base)) if seed and kind == "flat" else None
    code_bits = round(bits * base.shape[1])
    started = time.time()
    if kind == "flat":
        index = TieredQIndex(base.shape[1], code_bits, mode="fast", metric="l2", max_scan_bits=cap)
        index.add(base[order] if order is not None else base)
    else:
        index = TieredQIVFIndex(dimensions=base.shape[1], code_bits=code_bits, lists=AUTOMATIC_LISTS_1M, metric="l2",
                                max_scan_bits=cap)
        if seed:
            rows = max(MIN_LIST_SAMPLE, SAMPLE_ROWS_PER_LIST * AUTOMATIC_LISTS_1M)
            pick = np.sort(np.random.default_rng(seed).choice(len(base), rows, replace=False))
            index.train_lists(np.ascontiguousarray(base[pick]))
        index.build(base)
    build_s = time.time() - started
    row = dict(kind=kind, bits_per_dim=bits, max_scan_bits=cap, seed=seed, rows=len(base), dim=base.shape[1],
               build_s=build_s, storage_bytes=index.storage_bytes, scalar=getattr(index, "uses_scalar_fast_scan", None))
    to_ids = (lambda ids: order[ids]) if order is not None else (lambda ids: ids)
    del base
    settings = [None] if kind == "flat" else nprobes
    search = {s: (lambda q, s=s: index.search(q, k=K)) if s is None else (lambda q, s=s: index.search(q, k=K, nprobe=s))
              for s in settings}
    batch = {s: recall(to_ids(search[s](queries)[1]), truth) for s in settings}
    single_thread(int(os.environ["QUERY_CORE"]))
    row["points"] = [dict(nprobe=s, recall=batch[s], **timed(search[s], queries)) for s in settings]
    out.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
