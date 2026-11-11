# TieredQ query speed from first principles

Status: Phase 0 analysis, validated against the single-thread Apple M1 benchmark in
`examples/scan_benchmark.cpp` (100k x 384 real embeddings, fixed model, p50). The
implementation results and the final "what limits us now" section are appended after
each verified change.

## 1. Dataflow from query vector to top-k

```
q[d]
  |  (query-dependent)
  v
transform[j] = sum_i q[i] * basis[i,j]                 d*m FMA
  |
  +--> weight[j] = transform[j] * sigma[j]             m mul
  |      |
  |      v
  |    lut[rank][level] = weight[j] * levels[bits][level]     sum(2^bits) mul
  |      |
  |      v
  |    scan_tables[group][entry] = sum of member LUT entries  sum(group entries) add
  |
  +--> centroid_scores[c] = q . centroid[c] + transform . mean       k*d FMA + m
  +--> correction_weights[c] = (q . centroid[c]) / (2 * |centroid[c]|^2)  k*d FMA + k
  |
  v  (document-dependent, 16-document byte planes)
for each 16-doc block:
    for each scan group:
        load 1-2 plane bytes per doc
        extract variable-width code
        lookup float value (vqtbl4 or scalar gather for 6-8 bit groups)
        accumulate per-doc float sum
    for each doc:
        load centroid id, fp16 factor, fp16 correction
        score = centroid_scores[cid] + correction_weights[cid]*corr + factor*sum
  |
  v  (top-k)
threshold + partial selection, heap, final sort
```

| Operation | Class |
|---|---|
| PCA projection, LUT entries, grouped tables, centroid dots, correction weights | query-dependent |
| basis / mean / sigma, centroid matrix, tier layout, bit offsets, level codebooks, scan-group geometry, kernel dispatch, table widths, block size | model-dependent |
| plane bytes, codes, centroid id, fp16 factor, fp16 correction, per-doc score | document-dependent |

The model-dependent part is fixed at load time. The query-dependent part is ~0.09 ms
and cannot be precompiled. The document-dependent scan is the KPI.

## 2. Per-document cost model on M1

Measured tier mix for the real 100k x 384 model (greedy allocation, 256 centroids):

| budget | code bytes | stride | tiers (dims@bits) | scan groups | 16-doc table lookups | scan p50 |
|---|---|---|---|---|---|---|
| 808 | 101 | 106 | 45@4, 135@3, 88@2, 47@1 | 234 | 234 x 4 | 4.81 ms |
| 1176 | 147 | 152 | 31@5, 142@4, 99@3, 53@2, 50@1 | 309 | 309 x 4 | 6.68 ms |
| 1560 | 195 | 200 | 23@6, 146@5, 105@4, 62@3, 41@2, 4@1 | 358 (23 wide) | 335 x 4 + 23x16 scalar loads | 9.16 ms |

At 100k docs and 808 bits: 48.1 ns/doc, i.e. ~154 cycles/doc at the M1's nominal
3.2 GHz, and 2.2 GB/s of payload. Per group of 16 docs the current NEON loop is
roughly:

* 1-2 16-byte plane loads, 2 widening ops, 0-4 straddle ops;
* 2 variable shifts + 2 masks (scalar shift/mask construction each group);
* 4 widening ops to 32-bit lanes + 4 `*4` multiplies;
* 4 byte-offset adds + 4 `vqtbl4` lookups + 4 `fadd`;
* ~5 scalar loads/arithmetic ops for the group descriptor and table pointer.

That is about 25-30 issued ops per group per 16 docs, or ~370-440 ops/doc. The
metadata epilogue adds ~2 ops/doc: two 16-lane fp16 decodes, four strided centroid-id
loads, one gather for `centroid_scores`/`correction_weights`, and three FMA.

## 3. Roofline: which bound binds first?

Calibration on this machine: a single-thread 64 MiB copy reaches 23.7 GB/s. The
payload is read once per query and is L2-resident (10.6 MB) but is streamed by DRAM on
the first touch.

| Bound | Formula | 808-bit limit |
|---|---|---|
| Memory bandwidth | 23.7 GB/s / 106 B/doc | 224M docs/s (2240 QPS @100k) |
| NEON vector issue (ideal 4 ops/cycle) | 12.8 Gops/s / ~400 ops/doc | 32M docs/s (320 QPS @100k) |
| `vqtbl4` throughput (assume 1/cycle) | 3.2G tbl/s / 58.5 tbl/doc | 55M docs/s (550 QPS @100k) |
| Top-k selection | measured 0.077 ms / 100k docs | independent; 1.5% of total |
| Scan (measured) | 4.81 ms / 100k docs | 20.8M docs/s (208 QPS @100k) |

The scan is **instruction-issue and dependency bound**, not bandwidth bound: it uses
only ~9% of the streaming bandwidth and ~55-65% of the ideal vector issue rate. The
lookup throughput alone would allow ~2.6x more, so decode and per-group bookkeeping,
not `vqtbl4`, are the first-order cost. Top-k and query prep are noise (1.5% and 1.8%
of end-to-end).

The second observed law is more useful than any cycle estimate: **scan time tracks the
number of scan groups, not bytes**. 1176 bits has 32% more groups than 808 and is 39%
slower; 1560 bits has 53% more groups and is 90% slower. Halving per-group instructions
is therefore the direct route to throughput.

## 4. Ideal circuit vs current NEON

Ideal per group per 16 docs:

1. load the minimum whole number of plane bytes;
2. extract codes with one variable shift and one constant mask per code vector;
3. scale codes by 4 while widening (a single `ushll #2`);
4. one `add` for the byte offsets `{0,1,2,3}`;
5. one `vqtbl4` per 4 docs, one `fadd`.

Current divergence:

* codes are widened and then multiplied by 4 separately (4 extra ops/group);
* shift and mask are rebuilt from scalar tier fields every group (5-9 scalar ops);
* tables and group descriptors are reloaded every 16-doc block;
* 6-8 bit groups fall back to 16 scalar loads per 4 docs because `vqtbl4` indexes
  64 bytes (16 floats), not 64 floats;
* the scan has no reuse across queries: batch search assigns one query per worker and
  re-decodes the same codes for every query.

`vdotq_s32` is attractive but the Lloyd-Max levels are not linear in the code. Using
int8/int16 dot products changes the score and violates the current exact-parity
contract unless followed by an exact rerank. It is kept as design 3, not the first
change.

## 5. What "pre-compile the query transform" means precisely

Query values are dynamic, so `transform`, `lut`, `scan_tables`, `centroid_scores` and
`correction_weights` must be computed per query. What can be specialized once at model
load:

* tier layout, group boundaries, table widths, byte offsets, shifts and masks;
* kernel selection and the number of docs processed per iteration;
* fused per-centroid constants (`centroid_scores`/weights stay query-dependent, the
  gather and fp16 epilogue shape do not);
* template variants keyed by group bit width (1-5 and wide 6-8) and block count.

Template specialization is enough. A JIT would only remove the per-group dispatch and
scalar shift/mask construction, which is a small, bounded set of variants (five narrow
widths plus three wide widths) and would add a compiler/runtime dependency. The
remaining unavoidable per-query work is the `d*m` projection, ~2.2k LUT multiplies,
`k*d` centroid dots, and ~3.9k group-table adds: 0.09 ms on this machine, under 2% of
the scan.

## 6. Designs ranked by expected speedup and risk

### Design 1 - tighter exact NEON scan (low risk)

Fuse the `*4` into widening, use one constant-folded mask per group width, process 32
docs per iteration to halve table/descriptor loads, and dispatch narrow groups by
width. No arithmetic change: per-document accumulation order stays group order, so
scores remain bit-identical.

* Expected: 1.4-1.7x single-query scan, most of it from issue-slot reduction.
* Falsified if: `vqtbl`/load latency, not issue, dominates; then 32-doc blocking will
  not help and the group-width dispatch will not move the needle.
* Cheapest test: variant microbenchmark on the fixed 808-bit real model with an exact
  score-parity assertion.

### Design 2 - multi-query tiled scan (medium risk)

One worker scans the same 16-doc block for up to four queries, decoding each group's
codes once and doing four sets of lookups. Feed a chunked top-k per query. At batch
32 with 8 threads this gives each worker four queries without changing the thread
count or the shared index.

* Expected: 1.2-1.4x additional batch QPS over design 1; 2x total is plausible.
* Falsified if: decoded-code cost is under ~20% of the group cost, or 16 accumulator
  vectors plus tables spill registers and erase the reuse.
* Cheapest test: same-thread batch QPS for 1, 2, 4 queries per worker with exact id/score
  equality against the single-query path.

### Design 3 - approximate int8/fp16 scan with exact rerank (higher risk)

Replace the float table with an 8-bit code table (`vqtbl1`) or an int16 dot product to
process 16 docs per lookup, then rerank the top ~64 candidates with the exact float
scorer.

* Expected: 2-3x scan throughput, but only if recall@10 is preserved and rerank stays
  below ~10% of the scan.
* Falsified by: recall loss on real embeddings, score tolerance changes, or rerank cost
  that erases the gain.
* Cheapest test: offline quantization-error simulation and top-100 rerank on the real
  100k x 384 fixture, before writing a kernel. This changes the kernel contract and
  would need explicit sign-off, so it is not the first change.

## 7. Profile confirmation

Phase 1 benchmark and `sample`/Instruments evidence are recorded in Section 8 after
the benchmark is added. The single-thread serial benchmark above already confirms the
relative sizes: prep ~1.8%, scan ~96.7%, top-k ~1.5% at 808 bits.

## 8. Implementation results

Measured on Apple M1, 100k x 384 real embeddings, 808-bit model, recall@10 against
the exact NEON scan. These are full scans: no reranking and no candidate generation.

| Path | Threads | Latency / throughput | Recall vs exact |
|---|---:|---:|---:|
| exact single query | 1 | 4.90 ms p50 | 1.000 |
| int8 cache single query | 1 | 0.82 ms p50 | 0.990 |
| exact batch, 32 rows | 8 | 868 QPS | 1.000 |
| int8 cache batch, 32 rows | 8 | 2416 QPS | 0.990 |
| exact batch, 32 rows | 1 | 262 QPS | 1.000 |
| int8 cache batch, 32 rows | 1 | 908 QPS | 0.990 |

The level cache stores one byte per coded coordinate (315 coordinates padded to
316 B/doc, 31.6 MB for this fixture, about 3.0x the 106-byte packed payload, excluding cache metadata) and uses `vdotq_s32`. It is opt-in via
`TieredQIndex::build_fast_scan()`; `search_fast` and `search_batch_fast` use it,
while the exact `search`/`search_batch` paths remain unchanged. Cache build is
about 66 ms for 100k rows.

Why the earlier bit-sliced design was dropped: TieredQ codes are Lloyd-Max labels,
not linear levels. Raw-code bitplanes with a linear level fit gave 0.378 recall
against the exact scan, uniform-level reinterpretation gave 0.396, and quantizing
the query to 1/2/3/4 bits gave 0.611/0.724/0.822/0.878 against ground truth
(exact scan: 0.936). Int8 q/level quantization preserves recall: 6-bit gave 0.935
and 8-bit gave 0.933 against ground truth.

The x86 AVX-512BW implementation and compact-cache experiment are measured below.
The level-lookup compact cache below remained prototype-only. Section 10 describes
the subsequently integrated query-specific lookup design; exact search remains the default.


## 9. ARM verification and Skylake-X transfer (2026-10-23)

Rebuilt the current library on Apple M1 and linked fresh benchmark executables.
100k documents, 384 input dimensions, **315 coded dimensions** (316 padded cache bytes/doc), k=10; single-query
medians across 100 queries at embedding rows 108000–108099:

| Path | Threads | Exact | Fast | Overlap@10 vs exact |
|---|---:|---:|---:|---:|
| M1 single query, p50 | 1 | 4.886 ms | 0.809 ms | 0.990 |
| M1 batch, 32 rows | 8 | 1054 QPS | 2902 QPS | 0.990 |
| i9-7980XE single query, p50 | 1 | 2.031 ms | 2.148 ms | 0.989 |

The M1 batch timing is the first timed repetition; recall is averaged across ten
batches. Single-query timings include query preparation, scan and top-k. Cache
build took 67.4 ms on M1 and 125.2 ms on hartmut. Both machines pass ctest (2/2).

ARM quantizes the Lloyd-Max levels once into the cache, then quantizes query weights
with per-tier level scales. Search uses signed native `vdotq_s32`, followed by the
centroid/factor/correction epilogue and top-k. It scans every document without
reranking. Four-query batching reuses each loaded level vector.

Hartmut has AVX-512F/BW but no VNNI or VBMI. The existing transfer uses biased
7-bit levels and signed 8-bit queries with `vpmaddubsw` followed by `vpmaddwd`.
Its 31.6 MB expanded level payload exceeds the 24.8 MB L3. Earlier perf measurements
showed about 29.5M LLC load misses for the expanded cache versus 84k for exact.
The native ARM dot instruction and cache behavior therefore do not transfer simply
by widening vectors. The x86 expanded-cache path is **not a speedup on this CPU**.

### Compact x86 prototype

`/tmp/tq_nibble_batch_x86.cpp` on both machines extends the nibble prototype with
8/16-query tiling, OpenMP over query tiles, query preparation and top-k. It uses
full scans, 7-bit levels and 8-bit queries. Packed codes plus level tables occupy
15.073 MiB (15.805 MB); padded metadata adds 0.9 MB. The benchmark uses three
alternating exact/fast timed repetitions after warmup and reports median throughput.
Its top-k uses partial_sort over document ids, unlike the library's chunked TopK;
these are prototype end-to-end numbers, not integrated-library performance.

The repeat below used tile=16. Queries are contiguous embedding rows starting at
108000. Timings exclude cache construction and reuse allocated score buffers.

| Threads | Rows | Exact QPS | Prototype QPS | Speedup | Overlap@10 |
|---:|---:|---:|---:|---:|---:|
| 8 | 32 | 4481 | 2841 | 0.63x | 0.9813 |
| 8 | 128 | 4395 | 7868 | 1.79x | 0.9859 |
| 8 | 256 | 4404 | 7443 | 1.69x | 0.9859 |
| 8 | 512 | 4622 | 8285 | 1.79x | 0.9869 |
| 16 | 32 | 8187 | 2428 | 0.30x | 0.9813 |
| 16 | 128 | 8532 | 8473 | 0.99x | 0.9859 |
| 16 | 256 | 8545 | 14709 | 1.72x | 0.9859 |
| 16 | 512 | 8667 | 13133 | 1.52x | 0.9869 |
| 32 | 32 | 2462 | 1164 | 0.47x | 0.9813 |
| 32 | 128 | 7220 | 5653 | 0.78x | 0.9859 |
| 32 | 256 | 8908 | 11679 | 1.31x | 0.9859 |
| 32 | 512 | 8912 | 14065 | 1.58x | 0.9869 |

Tile=8 also failed to establish 2x: a uniform-codebook lookup variant measured
1.56–1.72x for rows 128/256/512 at 8/16 threads. Small batches underutilize cores
because work is divided over query tiles. One initial 2.08x result at 256 rows,
16 threads, tile=16 did not reproduce (repeat: 1.72x). Do not treat it as a win.
The original single-query nibble scan measured 2.049 ms scan-only, overlap 0.989,
and score cosine correlation 0.999923. It did not establish sub-millisecond latency.

No integration of that prototype was made: the requested repeatable >=2x batch target
was not met. These measurements establish a limitation of this implementation on
Skylake-X, not a proof that all possible AVX designs are hardware-limited. Approximate
recall is reported explicitly; equal recall across paths has not been established.

Remote reproduction (existing data and library snapshot):

```sh
cd ~/experiments/TieredQ-int8-x86-20261023/src
g++ -std=c++20 -O3 -DNDEBUG -mavx512f -mavx512bw -Iinclude -Isrc \
  /tmp/tq_nibble_batch_x86.cpp build/libtieredq.a -fopenmp -llapack -lblas \
  -o /tmp/tq_nibble_batch_x86
OMP_NUM_THREADS=8 /tmp/tq_nibble_batch_x86 128 16
```


## 10. Single-core query-specific byte lookups (integrated)

The next experiment targeted single-core latency explicitly. It removed the
nibble-to-level expansion and query/level dot product. Models with tiers of at
most four bits now use this design through the existing `build_fast_scan()`,
`search_fast()` and `search_batch_fast()` API on AVX-512F/BW x86. Models with wider
tiers retain the previous expanded-level dot path; ARM retains its native int8 dot
path. Exact search remains the default. No reranking or candidate generation occurs.

### Representation and arithmetic

* Whole coordinates form lookup groups containing at most four code bits. This
  fixture has 236 groups. Two groups share a byte per document; cache blocks hold
  256 documents. These are combined-code labels, not quantized level values.
* Each query builds signed byte lookup scores from its coordinate weights and the
  original Lloyd-Max levels. Each section of up to 32 groups has its own scale.
  A single global scale measured only 0.975 overlap@10 on the first 100 queries;
  section scales recovered 0.992 on those queries and 0.9897 across 1,000 queries.
* `vpshufb` scores two groups for 64 documents, and `vpmaddubsw` sums their signed
  byte values using unsigned ones. Pair sums are bounded by 254, far below signed
  int16 saturation. A 32-group section is bounded by 4064, so int16 accumulation
  cannot overflow. Section sums are converted to float and scaled before the
  centroid/factor/correction epilogue.
* A derived query projection folds sigma and the centroid mean/correction terms
  into SIMD panels. This avoids building unused exact float lookup tables.
  Folding constants changes FP32 rounding, so it is confined to the approximate
  path. Persisted model and index formats are unchanged.

Cache payload for this 100k-document fixture:

| Component | Bytes |
|---|---:|
| Packed lookup codes, including tail padding | 11,811,328 |
| Padded centroid/factor/correction metadata | 900,864 |
| Derived query projection | 1,277,952 |
| Above components combined | 13,990,144 (13.34 MiB) |

Small group descriptors, level books, vector/object overhead and per-query buffers
are additional. This is an opt-in auxiliary cache; the original exact payload is
still retained. The old expanded-level cache is not allocated for this representation.
The earlier 31.6 MB figure was correct for padded level bytes, but the fixture
actually has **315** coded coordinates: `coded_dimensions()` reports 315 and the
four-coordinate grouping pads the level-cache stride to 316.

### Public API measurements on hartmut

Intel i9-7980XE, CPU affinity fixed to logical CPU 2, one OpenMP thread and one
OpenBLAS thread. 100k documents, k=10, 1,000 held-out queries (embedding rows
108000–108999), 64 warmup queries. The executable links the normal CMake-built
library without globally enabling AVX-512. Timings include query preparation,
score-buffer allocation, complete scan and top-k. Cache construction is excluded
from search latency and took about 93–101 ms across the recorded runs.

| Timing mode / run | Exact p50 | Fast p50 | Fast p95 | p50 speedup | Overlap@10 |
|---|---:|---:|---:|---:|---:|
| Separate warm paths, 1 | 1.719 ms | 0.748 ms | 0.769 ms | 2.30x | 0.9897 |
| Separate warm paths, 2 | 1.722 ms | 0.751 ms | 0.818 ms | 2.29x | 0.9897 |
| Separate warm paths, 3 | 1.722 ms | 0.751 ms | 0.843 ms | 2.29x | 0.9897 |
| Alternating exact/fast | 1.974 ms | 0.998 ms | 1.125 ms | 1.98x | 0.9897 |

After the final integration changes and successful tests, a verification run measured
1.787 ms exact / 0.794 ms fast p50 (2.25x), fast p95 0.839 ms. Alternating mode
measured 1.909 ms exact / 0.962 ms fast p50 and fast p95 1.104 ms. Both retained
0.9897 overlap@10. Final ctest passes on hartmut and ARM: 2/2 on each.

Thus warm single-core sub-millisecond latency reproduces through the public API.
Cache competition matters: alternating exact and fast paths is slower and its p95
exceeds 1 ms. This is not a worst-case latency guarantee. Recall is top-10 overlap
against TieredQ exact search, not recall against ground truth. No claim of the
original >=2x **batched** QPS objective follows from these single-core results.

ARM regression benchmark with the same executable and 1,000 queries: 4.827 ms
exact p50, 0.744 ms fast p50, 0.952 ms fast p95, overlap@10 0.9864. This uses a
larger query set than the earlier 100-query overlap of 0.990.

### Validation and reproduction

Tests compare the AVX byte-table output with an independent scalar sum from the
original encoded rows. Coverage includes odd group counts, section boundaries,
positive/negative accumulator bounds, zero queries, document counts around 16/64/256
boundaries, guarded tail stores, linear projection terms, public single/batch
parity, wider-tier fallback, invalid query rejection and deterministic cache rebuild.
Query byte vectors in the existing ARM and wide x86 dot paths now pad their final
four-coordinate group rather than reading past the query's coded dimensions.
The shared batch preparation/selection orchestration is deduplicated, and worker
preparation errors are rethrown outside the OpenMP region.

```sh
cd ~/experiments/TieredQ-int8-x86-20261023/src
cmake --build build -j36
ctest --test-dir build --output-on-failure
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 \
  build/tieredq_fast_scan_benchmark ../data/tq_808.index \
  ../data/scale_embeddings.f32 1000 separate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 2 \
  build/tieredq_fast_scan_benchmark ../data/tq_808.index \
  ../data/scale_embeddings.f32 1000 alternating
```


## 11. GloVe200 validation

[Single-core real-data measurements and precision tradeoffs](glove200-fast-scan.md)
compare the old exact scorer and opt-in fast mode on 100k and all 1,183,514
GloVe200 base vectors. Scan-only speedup at 100k is 2.13–2.18x with 0.9818
top-10 overlap; end-to-end speedup on the full base is 1.76–1.85x for the new
byte-lookup path. Exact remains default, fast remains approximate, and ARM is
unchanged by these precision experiments.


## 12. Python opt-in API and lifecycle validation

Python now exposes `build_fast_scan()`, read-only `has_fast_scan`, and
`search_fast(queries, k=10)`. The latter accepts a single vector or a matrix;
`search()` remains exact even after the cache is built. Shared binding validation,
result ownership and native routing are reused rather than duplicated. Single
queries call the serial native search, and matrices use native batch search.

A successful nonempty append invalidates the fast cache. Empty/failed appends
preserve it. Loaded indexes start without a derived cache, and fast calls safely
use exact search until rebuilt. Python's exclusive lock protects cache creation
and append against searches. Regression tests first reproduced the stale-cache
append bug, then verified invalidation, fallback and rebuild.

Python validation also exposed an existing exact-batch worker-exception abort.
Exact and fast batch query preparation now share an exception-safe helper that
rethrows outside OpenMP. Invalid input raises a Python exception without aborting
the process. This changes error handling, not valid-query scoring arithmetic.

Final API validation on Apple M1 and hartmut: **25 Python tests and 2/2 C++ tests
pass on each platform**. Coverage includes both query ranks, k=0 and k>size,
exact-default preservation, unavailable-cache fallback, append/load lifecycle,
owned outputs, invalid inputs, worker exceptions and concurrent build/search/add.
Keep exact as the cross-platform default: high ARM overlap does not establish
exact equivalence or make its additional memory free. Fast remains explicit.
