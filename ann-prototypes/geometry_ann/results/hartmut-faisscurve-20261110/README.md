# Faiss IVF-SQ8 baselines: full held-out curves and 32-thread batch QPS (#51, #52)

Remote root: `/home/sean/experiments/TieredQ-issue44-faisscurve-20261110`. Faiss 1.15.1, shared SQ8 codes, each index
trained by Faiss on 40 rows per list with 4 Lloyd iterations. Every budget is replayed on the 750 held-out queries, one
at a time on pinned core 2 (`heldout.csv`), then as one 32-thread batch call (`heldout.csv.batch.json`).

| Configuration | Build | Serving RSS (anon) | Held-out recall @ p50 |
|---|---:|---:|---|
| IVF-HNSW, 64K | 1,621 s | 13.6 GB | 0.942 @ 2.66, 0.949 @ 3.33, 0.960 @ 4.86, 0.974 @ 9.55, 0.985 @ 24.9, 0.991 @ 73.5 ms |
| IVF-HNSW, 128K | 3,052 s | 13.9 GB | 0.943 @ 1.98, 0.954 @ 2.94, 0.963 @ 3.87, 0.972 @ 7.52, 0.982 @ 14.9, 0.988 @ 29.4 ms |
| IVF flat, 64K | 1,634 s | 14.4 GB | 0.951 @ 19.9, 0.958 @ 20.7, 0.972 @ 23.4, 0.985 @ 34.0, 0.990 @ 60.5 ms |

## Faiss 1.15.1 bug: flat quantizer, `parallel_mode` 0, multi-threaded batch

The batch pass of `ivf-64k` reported recall 0 at every budget. Debugged in `debug/faiss-batch-debug.py` on an IVF-SQ8
index built from the given 64K Lloyd partition (`results/debug-ivf-64k-given`, 10 s build). The quantizer agrees between
batch and single calls. But with the default `parallel_mode` 0 and more than a few threads, the IVF scan returns one
repeated distance (1.607) and one id for every slot. That happened at 2 threads with 2 queries, and at 32 threads with
64 or 750 queries; one thread is always correct. Modes 1, 2 and 3 are correct at every thread count. A 500K-row,
4,096-list real-data build was correct in mode 0 (`debug/faiss-batch-repro-real.py`), as was synthetic data
(`debug/faiss-batch-repro.py`). IVF-HNSW is unaffected, and single-query numbers run on one thread.

The flat quantizer's batch curve was therefore remeasured on the given-partition 64K index in modes 1–3
(`results/debug-ivf-64k-given/flat-batch-modes.txt`). The best mode (2 or 3) gives: 3,880 QPS @ 0.940,
3,091 @ 0.950, 2,591 @ 0.960, 1,979 @ 0.967, 1,598 @ 0.971, 1,133 @ 0.977, 887 @ 0.979, 613 @ 0.983, 477 @ 0.986,
325 @ 0.988, 247 @ 0.991. The study's own `ivf-64k` batch rows are invalid.

`ivf-64k.stopped-223746` is the first flat-64K attempt, stopped when the nested study moved ahead of it. The exact
quantizer at 128K is in the finelists study.
