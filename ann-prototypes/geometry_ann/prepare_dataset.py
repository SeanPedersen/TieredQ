# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu>=1.8", "numpy>=1.26"]
# ///
"""Export a deterministic held-out L2 benchmark from an existing numeric .npy matrix."""
import argparse
import os
import json
from pathlib import Path
import time

for thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[thread_variable] = "1"

import faiss
import numpy as np
from vecs_io import write_records

K = 10
CHUNK_ROWS = 4096
QUERY_BATCH = 32



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = np.load(args.source, mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or not np.issubdtype(source.dtype, np.number):
        raise ValueError("Source must be a numeric vector matrix")
    if args.nq <= 0 or len(source) - args.nq < K:
        raise ValueError("Query split leaves too few base vectors")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [args.output / name for name in ("base.fvecs", "queries.fvecs", "truth.ivecs", "manifest.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output already contains benchmark artifacts")
    start = time.perf_counter()
    query_ids = np.sort(np.random.default_rng(args.seed).choice(len(source), args.nq, replace=False))
    keep = np.ones(len(source), dtype=bool)
    keep[query_ids] = False
    queries = np.ascontiguousarray(source[query_ids], dtype=np.float32)
    base = np.empty((len(source) - args.nq, source.shape[1]), dtype=np.float32)
    position = 0
    for offset in range(0, len(source), CHUNK_ROWS):
        block = np.asarray(source[offset : offset + CHUNK_ROWS], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Source contains nonfinite values or overflows FP32")
        selected = block[keep[offset : offset + CHUNK_ROWS]]
        base[position : position + len(selected)] = selected
        position += len(selected)
    print(f"Prepared FP32 base={base.shape}, queries={queries.shape}; computing exact L2 truth", flush=True)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatL2(source.shape[1])
    index.add(base)
    truth = np.empty((len(queries), K), dtype=np.int32)
    for offset in range(0, len(queries), QUERY_BATCH):
        _, ids = index.search(queries[offset : offset + QUERY_BATCH], K)
        truth[offset : offset + len(ids)] = ids
        print(f"Exact truth {offset + len(ids)}/{len(queries)} ({time.perf_counter() - start:.1f}s)", flush=True)
    del index
    hashes = {}
    for path, data, dtype in zip(paths[:3], (base, queries, truth), ("<f4", "<f4", "<i4"), strict=True):
        hashes[path.name] = write_records(path, data, dtype)
        print(f"Wrote {path} ({time.perf_counter() - start:.1f}s)", flush=True)
    manifest = dict(source=str(args.source.resolve()), source_shape=source.shape, source_dtype=str(source.dtype),
                    source_size=args.source.stat().st_size, source_mtime_ns=args.source.stat().st_mtime_ns,
                    base_shape=base.shape, query_shape=queries.shape, seed=args.seed,
                    held_out_source_ids=query_ids.tolist(), metric="squared L2", normalization="none",
                    truth="Faiss IndexFlatL2, FP32, single thread", faiss_version=faiss.__version__, sha256=hashes)
    paths[3].write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
