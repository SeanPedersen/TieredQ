# Compact narrow fast-query projection

This bundle measures a focused serving-memory change from HEAD
`3c9dbf590209dabf8ecccd3385fd56254779c70d`. The implementation regenerates
scaled basis coefficients from the model during narrow lookup queries, retaining
the centroid projection panels. Original code budgets, training, threshold-5
routing, packed high-bit codes, FP32 operation order and persisted formats are
preserved. Results below distinguish unchanged-budget savings from the separate
larger-budget quality follow-up.

The primary result at unchanged code budget is **57.01632 B/vector less serving
RAM (14.18%)**, with byte-identical saved indexes and identical full-query IDs.
Spending the saving on 2,048 code bits produces **Recall@10 0.9192**, versus a
fresh native RaBitQ 2-bit result of **0.9041**, while fitting both its disk and
RAM budgets. This is a recall/storage result, with a throughput tradeoff.

## Separate disk and serving RAM

Disk is actual self-contained serialized file size. RAM includes allocated code,
metadata, model, rotation and cache storage plus core C++ objects. The common
RAM boundary adds TieredQ's 56-byte index shell (excluding its already counted
embedded codec) to the historical counter. RaBitQ RAM includes its 136-byte core
object, 96-byte FHT/Kac rotation object, 768 allocated sign-flip bytes, 6,144
center bytes and native binary/extra code-factor buffers. The FHT callable is
verified to be a free-function pointer stored without a heap target. A file
header is disk state, not a RAM allocation. Adapter/Python wrapper objects,
allocator bookkeeping and transient query/top-k scratch are excluded on both
sides. RAM is component allocation accounting, not RSS.

| OpenAI point | Code bits/vector | Disk bits/vector | Serving RAM bits/vector | Recall@10 |
|---|---:|---:|---:|---:|
| TieredQ baseline | 1695 | 2507.67360 | 3217.59808 | 0.8991 |
| TieredQ compact | 1695 | 2507.67360 | 2761.46752 | 0.8991 |
| TieredQ compact, larger codes | 1920 | 2800.20608 | 3054.65536 | 0.9121 |
| TieredQ compact, closer aligned | 2048 | 2968.46016 | 3224.05632 | 0.9192 |
| Native RaBitQ | 3072 (2-bit) | 3232.55680 | 3232.57152 | 0.9041 |

The compact 2,048-bit point uses 403.00704 RAM B/vector and 371.05752 disk
B/vector, versus RaBitQ's 404.07144 RAM and 404.06960 disk B/vector. Its RAM is
0.263% lower, disk 8.17% lower, and recall is 1.51 percentage points higher.
This is a discrete within-both-budgets comparison, not exact equality or an
optimal-code-allocation claim. The 2,048-bit selection used storage alone;
`dual-budget.json` retains its calibration and the two independently measured
limits. No recall was used to choose the point. The earlier 1,920-bit follow-up
remains separate. Fresh baseline builds at both larger budgets reproduce their
IDs exactly; the old representation needs 443.76288/468.87016 B/vector at
1,920/2,048 bits (plus the same 56-byte shell). The new feasibility is attributable
to the removed projection, not a threshold-5 routing switch: both larger points
still have a widest tier of four bits.

The fix itself does **not** shrink disk files. The removed projection was never
serialized. It creates RAM headroom that can be spent on codes; the selected
larger-code file still fits the separate RaBitQ disk limit. At unchanged budgets,
all seven persisted index SHA-256 hashes match before/after.

## Unchanged-budget construction and serving results

The following bytes use the original `max(serialized, reloaded cache + model)`
metric, retained so the starting 402.19920 B/vector can be reconciled exactly.
Add 0.00056 B/vector to resident totals for the common core-object RAM boundary
above. Disk and RAM are not collapsed in the comparison table above.

| Code bits | Serving budget B/vector before → after | Build seconds before → after | Encode vectors/s before → after | Recall before = after | 8-thread/batch-32 QPS before → after |
|---:|---:|---:|---:|---:|---:|
| 1695 | 402.19920 → 345.18288 | 14.209 → 14.573 | 99,287 → 95,379 | 0.8991 | 3,672 → 3,250 |
| 5344 | 787.79958 → 787.79958 | 15.295 → 15.261 | 52,689 → 50,962 | 0.9793 | 1,075 → 1,057 |
| 11488 | 1555.81058 → 1555.81058 | 21.864 → 22.307 | 12,777 → 12,562 | 0.9982 | 942 → 827 |
| 1536 | 372.06256 → 318.97840 | 14.063 → 15.013 | 118,325 → 99,477 | storage only | — |
| 3072 | 493.90534 → 493.90534 | 14.825 → 15.763 | 74,607 → 70,807 | storage only | — |
| 6144 | 887.80474 → 887.80474 | 16.232 → 16.066 | 43,568 → 44,732 | storage only | — |
| 12288 | 1655.92464 → 1655.92464 | 26.154 → 26.286 | 11,227 → 11,185 | storage only | — |

Construction includes training and encode/pack; encode throughput excludes
training. Serialization/reload diagnostics are outside build/search timers.
Code allocation, serialized sizes and full query IDs are invariant across
scheduling and the unchanged-budget A/B. Additional original-grid points were
storage-only; no new recall claim is made for those rows.

| Low-memory serving point | 8-thread/batch-32 QPS | Batch-1 p50 / p95 ms, 1 thread | Batch-1 p50 / p95 ms, 8 threads | Build s | Encode vectors/s |
|---|---:|---:|---:|---:|---:|
| TieredQ baseline, 1695 | 3,672 | 2.125 / 2.459 | 0.884 / 0.971 | 14.209 | 99,287 |
| TieredQ compact, 1695 | 3,250 | 2.343 / 2.594 | 1.037 / 1.092 | 14.573 | 95,379 |
| TieredQ compact, 1920 | 2,717 | 2.725 / 3.044 | 1.266 / 1.434 | 14.372 | 90,090 |
| TieredQ compact, 2048 | 2,821 | 2.768 / 3.036 | 1.194 / 1.255 | 14.526 | 82,607 |
| Native RaBitQ, 2-bit | 4,718 | 1.600 / 1.886 | 1.577 / 1.854 | 2.386 | 42,857 |

The 1,695-bit compact point measured 11.5% lower throughput and 17.3% higher
8-thread single-query p50 than baseline (1-thread p50 +10.3%). These are observed
run differences, not an isolated causal estimate: even the untouched high-bit
11,488-bit path measured 12.2% lower throughput and 5.3% higher p50. Background
services and sequential, non-interleaved A/B timing limit attribution. The final
2,048-bit point has 0.60× RaBitQ batch throughput, lower 8-thread single-query
latency, and higher 1-thread single-query latency. Do not claim an across-the-board
speed win. The initial strided regeneration candidate had worse single-query
latency; its raw results and patch are retained as `candidate-strided*`. The final
kernel traverses basis rows sequentially while preserving each output's reduction
order. No further kernel, pruning or allocation-objective experiments were run.

## Concrete 1,695-bit allocation breakdown

| Component | Before bytes | After bytes | Scaling / role |
|---|---:|---:|---|
| Lookup codes, including padded documents | 23,222,272 | 23,222,272 | Per-vector, native nibble group expansion |
| FP32 factor array | 400,384 | 400,384 | Per-vector, including padded documents |
| FP32 correction array | 400,384 | 400,384 | Per-vector, including padded documents |
| Centroid IDs | 100,096 | 100,096 | Per-vector, including padded documents |
| Query projection | 8,847,360 | 3,145,728 | Shared D × output-width FP32 map |
| Lookup groups | 8,192 | 8,192 | Shared rank/tier geometry, allocated capacity |
| Lookup codebooks | 2,256 | 2,256 | Shared duplicated codebooks, retained |
| Cache objects | 384 | 384 | Shared objects |
| Model basis | 5,640,192 | 5,640,192 | Shared D × coded-ranks; required for save/append |
| Model centroids | 1,572,864 | 1,572,864 | Shared D × centroids |
| Other model state | 25,536 | 25,536 | Mean/sigma, tiers, half norms, scan plan, books, model/encoder objects |
| Historical resident counter total | 40,219,920 | 34,518,288 | Complete existing counted state |
| Additional common-boundary index shell | 56 | 56 | Core object, avoiding embedded-codec double count |
| Common-boundary resident total | 40,219,976 | 34,518,344 | Used for separate RAM comparison |

The original 88.81328 B/vector beyond the 241-byte vector stride consists of
88.47360 B/vector of query projection, 0.23136 B/vector of document padding, and
0.10832 B/vector of lookup groups/codebooks/cache objects. Model serving state
adds 72.38592 B/vector. This separates required live state, retained duplication,
representation expansion and padding; they are not all removable shared bytes.

## Allocation attribution and choice

At the original OpenAI 1,695-bit point (918 coded PCA ranks), the fast-vector
stride is 241 bytes: 232 lookup-code bytes and two FP32 metadata arrays plus one
centroid byte (9 bytes). Its nominal residual codes need 212 whole bytes; the
lookup grouping uses another 20 bytes per vector because groups stop at tier
boundaries and each occupies a nibble. This expansion scales with vector count;
it is not a shared projection or a second copy of the original packed row.
Lookup blocks pad the document count from 100,000 to 100,096, adding 23,136 bytes
across codes and metadata. The resident counter uses capacities, including spare
append capacity when present; reloading makes these allocations canonical.

The old dense query projection has 1,440 output columns: 918 scaled basis
weights, 512 centroid/correction terms and ten alignment columns. At 1,536 input
dimensions and FP32 precision it owns 8,847,360 bytes, or 88.4736 B/vector at
100K rows. It duplicates derived knowledge from the serving model; the model's
basis itself is still required for serialization and append, and cannot simply
be excluded from accounting. The model also retains centroids, mean, sigma,
codebooks, half norms, scan geometry, tier metadata and the lazy encoder object.
After append it additionally retains the lazy encoder's buffers; these are
counted by `memory_bytes()` and are absent after save/load.

Options considered before implementation:

| Approach | Benefit | Cost / reason for choice |
|---|---|---|
| Repack narrow lookup groups | Reduce the 20-byte per-row expansion and block padding | Changes group boundaries/LUT quantization or scan/persistence layout; larger correctness surface |
| Remove model basis or reduce its precision | Large shared saving | Basis is required by save/append; reducing precision risks codes and recall |
| Regenerate the entire query map | Remove all 88.474 B/vector of dense projection | Repeats centroid normalization and transforms; larger preprocessing cost |
| Regenerate scaled basis coefficients, retain centroid panels | Remove duplicate basis coefficients and their padding | Adds one FP32 multiply per basis coefficient per query; selected focused change |

The selected map owns only the 512 centroid columns (3,145,728 bytes).
The removed 5,701,632 bytes comprise 5,640,192 bytes of scaled basis coefficients
and 61,440 bytes of output-alignment padding. These are shared dimensional
savings: 57.01632 B/vector at 100K rows, 5.701632 B/vector at 1M rows. They do not
reduce the per-vector code stride. The retained centroid panels intentionally
trade memory for ordered SIMD reductions. Codebooks/groups/cache objects remain
fully counted. Packed-code paths selected by tiers of five or more bits do not
own this map and receive no memory saving.

Weights are computed as `query[i] * (basis[i, rank] * sigma[rank])`, followed by
the same ordered FP32 additions as the old materialized coefficients. Moving
sigma outside the reduction would change rounding and is deliberately avoided.
Query calls borrow the current codec's basis/sigma through an internal view;
the cache holds no pointers into a codec, preserving copies, moves and append
ownership. No new public API or file format is introduced. The code uses the
existing C++20/CMake/OpenMP stack and Python 3.12/uv evidence harness.

## Protocol and reproduction

Full OpenAI3 base (100,000 × 1,536), 1,000 held-out queries, seed 0 only. Three
warmups and three complete timing repeats, pinned to CPUs 0–7 on hartmut
(i9-7980XE, AVX-512BW). BLAS uses one thread; construction uses eight OpenMP
workers. Primary throughput uses eight threads/batch 32. All three aligned
points check eight-thread single-query response latency; the low-memory point
also checks one-thread single-query latency. Additional original-grid points
are storage-only, with construction/encode costs recorded. No normal-mode
benchmark, seed sweep or broad optimization experiment is included.

Both variants use fresh isolated builds and environments under
`/home/sean/experiments/TieredQ-serving-memory-20261026-OxwQTjbU/{before,after,sequential}`.
The baseline exports HEAD; the implementation applies `implementation.patch`
and adds `test_tieredq_projection.cpp`. Both apply the preserved exact-budget
binding patch from the aligned bundle. `instrument_memory.py` adds experiment-only
allocation access and reload checks to the binding, with no representation or
scoring changes. It changes C++ access control solely in the diagnostic binding;
no production introspection API is added. Capacity totals are checked against
the real counters, including model/cache objects and lazy encoder state.

Use the preserved `build-job.sh` and `measurement-job.sh` with a fresh root,
then run `run_memory.py before`, `run_memory.py after`, and
`run_memory.py after --quality` from their respective `benchmark/` directories
with the extension on `PYTHONPATH`. The quality follow-up chooses budgets using
serving storage alone before evaluating recall. Reuse the verified original
cache via a symlink; copy the pinned RaBitQ dependency into the fresh workspace
so overlay preparation cannot modify historical dependencies. All effective
compiler commands are retained and audited for `-O3 -march=native
-ffp-contract=off`; nanobind uses `NOMINSIZE`.

The other session's learned-codebook jobs started after the initial job check.
This measurement waited for them rather than stopping or modifying their work.
Background services remain a timing limitation; this is not an exclusive-host
or statistical-confidence claim. Historical result bundles and the three named
remote workspaces remain unchanged. Dataset hashes/manifests and source/binary
hashes are retained in the JSON evidence. Allocator bookkeeping, shared-pointer
control blocks and transient query/top-k scratch follow the existing exclusions;
measured serving bytes are not RSS. Required persistent state remains counted.

## Validation, artifacts and limits

- `audit.py` / `audit.json`: allocation sum identities, full-data manifests/truth
  hashes, seven identical persisted indexes, three identical 1,000-query ID
  arrays, loaded score/ID equality, both larger-budget baseline checks and the
  separate disk/RAM constraints. Run `python3 audit.py` in this bundle.
- `before.json`, `after.json`, `quality.json`, `dual-budget.json`,
  `quality-baseline.json`, `dual-budget-baseline.json`: raw measurements,
  component allocations, training tiers, build/encode costs and focused timings.
- `rabitq-memory.json`: fresh 2/4/8-bit disk/RAM components, construction/encode
  costs and fresh 2-bit seed-0 recall/timings. Higher RaBitQ anchors are
  storage-only in this follow-up. Original native bounds and HACC are retained.
- Fresh x86 Release C++ suites and all 16 benchmark validation tests pass for
  baseline, the initial candidate and the final implementation. Final local ARM
  C++ tests and 44 rebuilt Python API tests pass. New projection tests compare
  old materialized coefficients with regenerated results, including SIMD tails,
  918 ranks, zero norms and AVX2/AVX-512. Persistence/ownership tests compare
  scores through reload and append; high-bit scalar parity remains covered.
- Compiler commands, source/binary hashes, diagnostic patches, machine details
  and job logs are retained. The benchmark-only RaBitQ rotation accessor exposes
  actual allocated capacity; `rabitq-memory-introspection.patch` and its overlay
  hash record that diagnostic change. Codec kernels are unchanged.

Existing 1,695-bit recall is not a new regression. This is one full-data seed-0
run; there is no statistical-confidence claim, seed sweep or optimality claim.
The change reduces shared dimensional RAM overhead, not per-row code storage or
serialized bytes. It also affects narrow SIFT/GloVe models, but their new memory,
quality-at-larger-budget and timing results were not measured here. Packed paths
with a tier wider than four bits are unchanged. Encoder buffers remain resident
after append and are counted; serving values here describe saved/reloaded
indexes. Historical outputs and concurrently created codebook experiment files
are preserved and excluded from this implementation.
