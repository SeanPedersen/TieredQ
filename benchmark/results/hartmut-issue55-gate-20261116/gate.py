"""#55 non-regression gate: build or serve one TieredQ IVF index (flat or nested) for the A/B of aae87c7 (A) against the
step-1 library change (B, fixed-capacity tiles and TieredQ::map_scores).

build KIND DATASET INDEX.tq [TIER.rows] OUT.json: builds on BUILD_CORES threads and saves.
  flat qwen:   Qwen/MS MARCO 8.84M x 1024 from the SQ8 store, 1 bit/dim, automatic lists (64K, router), inner product.
  flat arxiv:  VIBE arxiv-nomic-768-normalized, 1.75 bits/dim, 2,048 lists, inner product.
  nested qwen / arxiv: TieredQNestedIVFIndex.build defaults (1-bit prefix, 8-bit fine tier) with the tier on NVMe.
serve KIND DATASET INDEX.tq [TIER.rows] OUT.json: loads, then per nprobe single queries alone on QUERY_CORE with one
  thread (best of PASSES: QPS, p50 / p95 of that pass), then one 32-thread batch at BATCH_NPROBE (best of PASSES).
  Anonymous RSS after load and after serving; ids and scores of the single-query pass go next to OUT.json as .npz.
Qwen serves k = 10, arxiv k = 100 (the issue's operating points)."""
import ctypes
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

PASSES = 3
QWEN = Path(os.environ.get("QWEN", "/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107"))
ARXIV = Path(os.environ.get("ARXIV", "/home/sean/experiments/TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5"))
QWEN_ROWS, QWEN_QUERIES = 8840823, 750
K = {"qwen": 10, "arxiv": 100}
NPROBES = {("flat", "qwen"): (64, 128, 256, 1024), ("nested", "qwen"): (128, 256, 512, 1024),
           ("flat", "arxiv"): (8, 16, 32, 48), ("nested", "arxiv"): (32, 64, 128)}
BATCH_NPROBE = {("flat", "qwen"): 256, ("nested", "qwen"): 256, ("flat", "arxiv"): 16, ("nested", "arxiv"): 64}
NESTED_CANDIDATES = {"qwen": None, "arxiv": 100}  # qwen: the index's default (40 reads); arxiv: k


def read_fvecs(path, count=None):
    raw = np.fromfile(path, dtype=np.int32)
    rows = raw.reshape(-1, raw[0] + 1)[:count, 1:]
    return np.ascontiguousarray(rows.view(np.float32))


def queries(dataset):
    if dataset == "qwen":
        return read_fvecs(QWEN / "data/test_queries.fvecs", QWEN_QUERIES)
    import h5py
    with h5py.File(ARXIV, "r") as f:
        return np.ascontiguousarray(f["test"][:], dtype=np.float32)


def sq8_store():
    scales = np.fromfile(QWEN / "scales.f32", dtype=np.float32)
    offsets = np.fromfile(QWEN / "offsets.f32", dtype=np.float32)
    codes = np.memmap(QWEN / "base.i8", dtype=np.int8, mode="r", shape=(QWEN_ROWS, len(scales)))
    return dict(codes=codes, scales=scales, offsets=offsets)


def arxiv_rows():
    import h5py
    with h5py.File(ARXIV, "r") as f:
        return np.ascontiguousarray(f["train"][:], dtype=np.float32)


def anonymous_rss():
    for line in open("/proc/self/status"):
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    return 0


def build(kind, dataset, index_path, tier_path, out):
    from tieredq import TieredQIVFIndex, TieredQNestedIVFIndex

    store = sq8_store() if dataset == "qwen" else dict(rows=arxiv_rows())
    started = time.perf_counter()
    if kind == "flat":
        if dataset == "qwen":
            index = TieredQIVFIndex(dimensions=len(store["scales"]), code_bits=len(store["scales"]), metric="ip")
            index.build(**store)
        else:
            rows = store["rows"]
            index = TieredQIVFIndex(dimensions=rows.shape[1], code_bits=round(1.75 * rows.shape[1]), lists=2048, metric="ip")
            index.build(rows)
        built = tiered = time.perf_counter()
    else:
        index = (TieredQNestedIVFIndex.build(**store) if dataset == "qwen"
                 else TieredQNestedIVFIndex.build(store["rows"], metric="ip"))
        built = time.perf_counter()
        if dataset == "qwen":
            index.write_tier(tier_path, **store)
        else:
            index.write_tier(tier_path, store["rows"])
        tiered = time.perf_counter()
    index.save(str(index_path))
    Path(out).write_text(json.dumps(dict(kind=kind, dataset=dataset, lists=index.lists, storage_bytes=index.storage_bytes,
                                         build_s=built - started, tier_s=tiered - built, total_s=tiered - started,
                                         file_bytes=Path(index_path).stat().st_size,
                                         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                                    indent=2) + "\n")


def single_thread(core):
    # A warm multi-thread OpenMP pool slows single queries through wake-ups; queries run on one thread.
    for library, setter in (("libgomp.so.1", "omp_set_num_threads"), ("libopenblas.so.0", "openblas_set_num_threads")):
        try:
            getattr(ctypes.CDLL(library), setter)(1)
        except OSError:
            pass
    os.sched_setaffinity(0, {core})


def serve(kind, dataset, index_path, tier_path, out):
    from tieredq import TieredQIVFIndex, TieredQNestedIVFIndex

    test, k = queries(dataset), K[dataset]
    build_cores = {int(c) for c in os.environ["BUILD_CORES"].split(",")}
    os.sched_setaffinity(0, build_cores)
    if kind == "flat":
        index = TieredQIVFIndex.load(str(index_path))
        search = lambda q, nprobe: index.search(q, k=k, nprobe=nprobe)
    else:
        index = TieredQNestedIVFIndex.load(str(index_path))
        index.attach_tier(str(tier_path), reads="direct")
        extra = {} if NESTED_CANDIDATES[dataset] is None else dict(candidates=NESTED_CANDIDATES[dataset])
        search = lambda q, nprobe: index.search(q, k=k, nprobe=nprobe, **extra)
    loaded_rss = anonymous_rss()
    batch_nprobe = BATCH_NPROBE[(kind, dataset)]
    search(test, batch_nprobe)
    batch_qps = 0.0
    for _ in range(PASSES):
        started = time.perf_counter()
        search(test, batch_nprobe)
        batch_qps = max(batch_qps, len(test) / (time.perf_counter() - started))
    single_thread(int(os.environ["QUERY_CORE"]))
    points, arrays = [], {}
    for nprobe in NPROBES[(kind, dataset)]:
        search(test[:1], nprobe)
        best = None
        for _ in range(PASSES):
            micros = np.empty(len(test))
            ids = np.empty((len(test), k), dtype=np.uint32); scores = np.empty((len(test), k), dtype=np.float32)
            for q in range(len(test)):
                started = time.perf_counter()
                s, i = search(test[q:q + 1], nprobe)
                micros[q] = (time.perf_counter() - started) * 1e6
                scores[q], ids[q] = s.reshape(-1), i.reshape(-1)
            if best is None or micros.sum() < best.sum():
                best = micros
        arrays[f"ids_np{nprobe}"], arrays[f"scores_np{nprobe}"] = ids, scores
        points.append(dict(nprobe=nprobe, single_qps=len(test) / (best.sum() / 1e6), p50_us=float(np.percentile(best, 50)),
                           p95_us=float(np.percentile(best, 95))))
    np.savez(Path(out).with_suffix(".npz"), **arrays)
    Path(out).write_text(json.dumps(dict(kind=kind, dataset=dataset, index=str(index_path), k=k,
                                         storage_bytes=index.storage_bytes, loaded_rss_bytes=loaded_rss,
                                         served_rss_bytes=anonymous_rss(), batch_nprobe=batch_nprobe, batch_qps=batch_qps,
                                         points=points), indent=2) + "\n")


if __name__ == "__main__":
    mode, kind, dataset, index_path = sys.argv[1:5]
    tier_path, out = (sys.argv[5], sys.argv[6]) if kind == "nested" else (None, sys.argv[5])
    (build if mode == "build" else serve)(kind, dataset, index_path, tier_path, out)
