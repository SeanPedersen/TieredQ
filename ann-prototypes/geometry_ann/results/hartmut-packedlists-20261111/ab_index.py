"""Build or serve one flat TieredQ IVF index for the #53 packed-list A/B.

build DATASET OUT.tq: builds on BUILD_CORES threads and saves; prints storage_bytes, list count and build seconds.
  arxiv: VIBE arxiv-nomic-768-normalized (hdf5), 1.75 bits/dim, 2,048 lists, inner product.
  qwen: Qwen/MS MARCO 8.84M x 1024 from the SQ8 store (base.i8, scales.f32, offsets.f32), 1 bit/dim (the nested
  default's prefix), automatic lists (64K), inner product.
serve DATASET IN.tq OUT.json: loads, then for k in K_VALUES and every nprobe: single queries alone on QUERY_CORE with one
  thread (best of PASSES), one batch call on BUILD_CORES threads (best of PASSES), and the batch ids and scores
  (written next to OUT.json as .npz) for the identity check."""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PASSES = 3
K_VALUES = (10, 100)
NPROBES = {"arxiv": (8, 16, 32, 48), "qwen": (64, 128, 256, 1024)}
QWEN = Path(os.environ.get("QWEN", "/home/sean/experiments/TieredQ-issue44-msmarco8m-20261107"))
ARXIV = Path(os.environ.get("ARXIV", "/home/sean/experiments/TieredQ-vibe-arxiv-20261106/data/arxiv-nomic-768-normalized.hdf5"))
QWEN_ROWS = 8840823


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    return np.ascontiguousarray(raw.reshape(-1, raw[0] + 1)[:, 1:].view(np.float32))


def queries(dataset):
    if dataset == "qwen":
        return read_fvecs(QWEN / "data/test_queries.fvecs")
    import h5py
    with h5py.File(ARXIV, "r") as f:
        return np.ascontiguousarray(f["test"][:], dtype=np.float32)


def build(dataset, out):
    from tieredq import TieredQIVFIndex

    started = time.time()
    if dataset == "qwen":
        scales = np.fromfile(QWEN / "scales.f32", dtype=np.float32)
        offsets = np.fromfile(QWEN / "offsets.f32", dtype=np.float32)
        codes = np.memmap(QWEN / "base.i8", dtype=np.int8, mode="r", shape=(QWEN_ROWS, len(scales)))
        index = TieredQIVFIndex(dimensions=len(scales), code_bits=len(scales), metric="ip")
        index.build(codes=codes, scales=scales, offsets=offsets)
    else:
        import h5py
        with h5py.File(ARXIV, "r") as f:
            rows = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        index = TieredQIVFIndex(dimensions=rows.shape[1], code_bits=round(1.75 * rows.shape[1]), lists=2048, metric="ip")
        index.build(rows)
    build_s = time.time() - started
    index.save(str(out))
    print(json.dumps(dict(dataset=dataset, lists=index.lists, storage_bytes=index.storage_bytes, build_s=build_s,
                          file_bytes=Path(out).stat().st_size)))


def single_thread(core):
    for library, setter in (("libgomp.so.1", "omp_set_num_threads"), ("libopenblas.so.0", "openblas_set_num_threads")):
        try:
            getattr(ctypes.CDLL(library), setter)(1)
        except OSError:
            pass
    os.sched_setaffinity(0, {core})


def serve(dataset, path, out):
    from tieredq import TieredQIVFIndex

    test = queries(dataset)
    build_cores = {int(c) for c in os.environ["BUILD_CORES"].split(",")}
    os.sched_setaffinity(0, build_cores)
    index = TieredQIVFIndex.load(str(path))
    batch, arrays = [], {}
    for k in K_VALUES:
        for nprobe in NPROBES[dataset]:
            index.search(test, k=k, nprobe=nprobe)
            best = 0.0
            for _ in range(PASSES):
                started = time.perf_counter()
                scores, ids = index.search(test, k=k, nprobe=nprobe)
                best = max(best, len(test) / (time.perf_counter() - started))
            arrays[f"ids_k{k}_np{nprobe}"], arrays[f"scores_k{k}_np{nprobe}"] = ids, scores
            batch.append(dict(k=k, nprobe=nprobe, batch_qps=best))
    single_thread(int(os.environ["QUERY_CORE"]))
    for point in batch:
        best = 0.0
        for _ in range(PASSES):
            total = 0.0
            for query in test:
                started = time.perf_counter()
                index.search(query[None, :], k=point["k"], nprobe=point["nprobe"])
                total += time.perf_counter() - started
            best = max(best, len(test) / total)
        point["single_qps"] = best
    np.savez(Path(out).with_suffix(".npz"), **arrays)
    Path(out).write_text(json.dumps(dict(dataset=dataset, index=str(path), storage_bytes=index.storage_bytes, points=batch),
                                    indent=2) + "\n")
    print(json.dumps(batch))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        serve(sys.argv[2], sys.argv[3], sys.argv[4])
