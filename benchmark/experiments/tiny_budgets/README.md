# Tiny total-code budgets on SIFT1M

This tests **TieredQ at budgets highlighted by Brandt**, not a faithful Brandt baseline. Production quantization/allocation remains unchanged. The codec includes cluster residuals, tier rotations, Gaussian tables, row scaling, and centroid correction; SIFT uses the existing query-metric L2 adapter. A direct Brandt comparison requires its own implementation and same-data evaluation.

The study uses the existing C++20 codec and Python 3.12/uv benchmark environment. Experimental bindings live in `../native/`; the learned-codebook experiment reuses the same bridge while retaining its original module name and training interface.

From the repository root:

```sh
cmake -S benchmark/experiments/tiny_budgets -B build-tiny \
  -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PWD/benchmark/.venv/bin/python"
cmake --build build-tiny -j8
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 benchmark/.venv/bin/python \
  benchmark/experiments/tiny_budgets/quality.py \
  --data /path/to/existing/benchmark/data --native build-tiny --output /path/to/new/results
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 benchmark/.venv/bin/python \
  benchmark/experiments/tiny_budgets/timing.py \
  --data /path/to/existing/benchmark/data --native build-tiny --results /path/to/results
benchmark/.venv/bin/python benchmark/experiments/tiny_budgets/summarize.py /path/to/results
```

Quality: full 1M SIFT base, existing 1,000 held-out queries and exact top-10 truth, training samples of 32,768 base vectors, 256 centroids, rotations enabled, Gaussian tables, seeds 0/1/2, and 64/80/128/256/512 total residual-code bits. Code budgets span 129 lifted dimensions; bits per original dimension refer to the original 128. The stored centroid ID and correction/scaling metadata are additional to these code bits.

`quality.py` retrieves 1,000 candidates to evaluate true-nearest-neighbor Recall@R for R=1/10/20/50/100/200/500/1000, plus ordinary Recall@10 from its first ten results. For each model, a separate top-10 call on 32 queries checks identical IDs and scores to the first ten from the top-1,000 call. Save/load canonicalizes allocations and excludes lazy encoder buffers from serving model memory. Both post-build and post-load counters are recorded. Seed-0 indexes remain for timing; other seeds share a scratch reload file. Raw recipes and per-query recall/rank artifacts are retained.

Serving storage counts `index.storage()` + `codec.memory_bytes()` + stored adapter bytes after reload, divided by one million. This includes required shared cache/model state and padding, and excludes caller-owned original vectors, transient query/selection scratch, allocator bookkeeping, and process RSS. Serialized index sizes are recorded separately. `vector_stride` describes fixed row layout; the difference between total memory and stride × row count includes padding and shared state, rather than shared state alone.

Timing: seed 0, top-10 searches; one and eight threads for single queries (first 128 queries), and eight threads/batch 32 for all 1,000 queries. Include query lifting/map multiplication in each timed call. Use three warmups per case and three repeats in seeded shuffled order across all budget/scorer/thread/batch cases. Report true batch response latency and throughput separately. Last batches can be partial. Single-query results cover a different query count from the main recall study; timed recall is recorded independently because float32 adapter matmul can round differently for different batch shapes.

For phase diagnostics, copy `src`, `include`, `cmake`, `CMakeLists.txt`, and `benchmark/experiments` plus `benchmark/l2_transform.py` into a **fresh separate source root**, then:

```sh
benchmark/.venv/bin/python benchmark/experiments/tiny_budgets/instrument.py /path/to/profile-source
cmake -S /path/to/profile-source/benchmark/experiments/tiny_budgets -B build-tiny-profile \
  -DCMAKE_BUILD_TYPE=Release -DEXPERIMENT_PROFILE=ON \
  -DPython_EXECUTABLE="$PWD/benchmark/.venv/bin/python"
cmake --build build-tiny-profile -j8
OPENBLAS_NUM_THREADS=1 benchmark/.venv/bin/python \
  benchmark/experiments/tiny_budgets/timing.py \
  --data /path/to/existing/benchmark/data --native build-tiny-profile \
  --results /path/to/results --profile
```

The profiler instruments query preparation, scan, selection, and final merge in that copy only. It checks warmup/profiled IDs match. Diagnostic runs use one thread and batches 1/32; the query adapter runs before phase timing. Preparation is wall time; scan, selection, and merge are accumulated worker elapsed times. At one thread they can be compared to the overall wall duration, but clocks/counters add overhead and uncaptured orchestration remains. These figures are diagnostics, not the uninstrumented latency measurements.

The report bundle retains source and binary hashes, compiler option audits (`-O3 -march=native -ffp-contract=off`, nanobind `NOMINSIZE`), machine metadata, raw rows, per-query artifacts, phase instrumentation patch, source snapshot, and plotting output. Remote data caches are reused without download or changes. No production source or file-format change is required.
