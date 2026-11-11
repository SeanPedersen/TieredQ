# Heap a TieredQ build leaves resident, and what storage_bytes misses (#43, #41)

Remote root: `/home/sean/experiments/TieredQ-issue43-heapretention-20261111`. `heap_probe.py` builds one index on the
full VIBE arxiv base (1,344,643 × 768, 32 threads, THP host default):
- flat: 1.75 bits/dim, 2,048 lists, the #43 configuration;
- nested: the default, then `write_tier` to the NVMe SSD.

After each step it records USS growth, the private KiB of `[heap]` and of anonymous regions, and glibc's per-arena
`malloc_info`. A last step calls `malloc_trim(0)` once more: whatever that returns was freed after the library's own
trim. Warm runs build a 65,536-row index first, so one-time process state precedes the baseline. Cold runs
(`WARMUP=0`) measure the way VIBE does. `study.py` ran the first round on 86f003d. `study_fix.py` ran the fix against
86f003d, plus heaptrack over the fixed warm builds; heaptrack ended right after the last snapshot, so its "leaked"
allocations are the ones still live (`results/*-heaptrack/live-sites.txt`).

## Causes

1. **The trim ran too early.** `TieredQIVFIndex::build` and `write_tier` called `malloc_trim` before their own
   buffers were destroyed. In the warm flat run, a second trim after `build()` returned 36.6 MiB of `[heap]`
   (50.3 → 13.7 MiB). With glibc's trim and mmap thresholds pinned (`GLIBC_TUNABLES`, which turns off their dynamic
   growth), nothing was left to return. With one arena (`MALLOC_ARENA_MAX=1`) the second trim was still needed.
2. **Thread-arena tops.** The refinement encoder allocated its per-block buffers (~2.3 MiB: rows, projections,
   panel dots) inside each OpenMP worker. Freed, they merged into the top chunk of that worker's arena, and
   `malloc_trim` never shrinks a thread arena's top. A cold nested build kept +67.5 MiB after `write_tier` (30 arenas
   × 2,316 KiB). A warm process reuses the same arenas, so the cost is paid once per process, about 2.3 MiB × build
   threads. VIBE's cold process charges all of it to its one index.

**Fix.** `build()` trims after a private `encode_lists()` returns, and `write_tier` trims after its buffers' scope.
`TieredQRefinement::encode` takes its per-worker buffers from a vector the calling thread allocates
(`encode_scratch`), so they live in the caller's arena. Tests passed on hartmut: CTest 2/2, pytest 220.

| Cold process, USS after the last step | 86f003d | fix |
|---|---:|---:|
| Flat | 330.1 MiB (318.1 after a second trim) | **314.6 MiB** (a second trim returns nothing) |
| Nested, after `write_tier` | 300.3 MiB (`write_tier` +67.5) | **230.5 MiB** (`write_tier` +1.0) |

## What stays live, and storage_bytes

Warm and fixed, flat sits 10.5 MiB above `storage_bytes` and nested 16.8 MiB. heaptrack places it here (its MB):

| Live allocation | Flat | Nested | In storage_bytes at 86f003d |
|---|---:|---:|---|
| Codes (`reserve_rows`) | 239.6 | 146.1 | yes |
| Coarse binary16 panels (`CoarseQuantizer::from`) | 3.2 | 3.6 | yes |
| Elias-Fano ids | 2.2 | 2.2 | yes |
| Codec encoder cache (`TieredQ::build_encoder`; serving never reads it) | 3.3 | 5.4 | **no** |
| Codec model (`train_model`: basis, centroids, levels) | ~1.5 | ~1.6 | **no** |
| Nested id → document map (`id_documents`) | – | 5.6 | **no** |
| Thread-local candidate gather scratch (`document_rows`) | – | 1.8 | no (per-thread scratch) |
| Python and h5py objects of the harness | ~5.9 | ~11.7 | – |

`storage_bytes` now counts that state:
- `TieredQIVFIndex::storage_bytes` adds the codec's `memory_bytes()`, its encoder cache included once built. The router
  already counted its own member codec that way.
- The new `TieredQNestedIVFIndex::storage_bytes` adds the refinement's `memory_bytes()` (beyond the prefix arrays it
  shares) and the id → document map. Python's nested `storage_bytes` and `TieredQNestedIndex` use it.

Thread-local scratch and the one-time process state stay uncounted: the 20.7 MiB region, OpenBLAS 4.1 MiB and the
module 1.8 MiB that `../hartmut-arxivfollowup-20261111` found in every cold process.

Not changed: the encoder cache could be dropped after a build instead of counted (3–5 MB here). It is rebuilt on
first use and `call_once` cannot reset it, so dropping it needs a resettable cache, which these 1–2% of storage do
not justify.
