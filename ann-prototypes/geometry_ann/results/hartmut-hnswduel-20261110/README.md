# Coarse router duel: Faiss HNSW vs the two-level router (#50) — parked, never run

Remote root: `/home/sean/experiments/TieredQ-issue44-hnswduel-20261110`. Status: **not run**. `launch.sh` was to start
`study.py` after the Faiss curve study released the shared lock. The study was marked `PARKED` on 2026-11-10 at
20:39 before it started, so there is no `results/` and no `status.json`. #50 was closed as not planned on 2026-11-11.

## What it would have measured

`hnsw_duel.py` routes 750 held-out Qwen/MS MARCO queries one at a time on pinned CPU 2. Builds run on 32 threads. It
compares three coarse routers over the Lloyd list centroids at 32K, 64K and 128K lists, plus 1M random SQ8-decoded
base rows standing in for 1M lists:

- Faiss `IndexHNSWFlat`: 32 neighbours and construction ef 100, as in the scorecard's IVF-HNSW quantizer. efSearch
  runs from 64 to 4,096.
- Faiss `IndexHNSWSQ` (8-bit), with an exact re-rank of its first 256 lists.
- TieredQ's two-level router: `TieredQIVFIndex.probe_lists` on an IVF whose lists are fixed to the centroids. 8 to
  1,024 probed supers.

Quality at depths L = 64 … 2,048 lists:
- **Coverage:** the share of each query's true top-10 whose list is among the first L. This needs the row labels,
  so the 1M case has none.
- **Routing recall:** the share of the exact first L lists that the router finds.

## Why it was parked

It mattered only for the low-recall tie with Faiss IVF-HNSW 128K (about 0.945) and for 1M+ lists. The tie went away
without it: with the router's rerank default at 64 and faster head packing (2881397, `../hartmut-routerprofile-20261111`),
TieredQ takes 1.71 ms at 0.945, against Faiss's 1.98 ms at 0.943. An HNSW router over compressed centroids may still
pay off at 1M+ lists. That case is untested.
