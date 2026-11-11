# Where the router's fixed cost goes, and the re-rank depth (#47 follow-up, #52)

Remote root: `/home/sean/experiments/TieredQ-issue47-routerprofile-20261111`. `profile_router.cpp` loads the router
study's from-scratch 64K index and times one pinned core over the 750 held-out queries (p50). `perf` is unavailable on
hartmut (`perf_event_paranoid` 2 refuses every event, software clocks included), so the cost is decomposed by timing.

**Decomposition** (`results/decompose.csv`, `probe_lists` at nprobe 64 and a fixed number of probed supers, rerank 256
vs 0). At the default 32 supers the router took 1.04 ms. About 475 us was the exact binary16 re-rank of the top 256
lists, 350 us was fixed (query preparation for the member codec and exact ranking of the 512 supers), and 210 us was the
member scan (about 5.4 us per super).

**Head packing.** The re-rank packed 256 row-major binary16 rows into column panels one element at a time.
`pack_panel_rows` copies cache-line blocks instead, with the same panels, so scores stay bit-identical (router tests
pass). It saves about 130 us (`results/after.csv`): `probe_lists` 1.03 → 0.91 ms at nprobe 32 and 1.08 → 0.88 ms at
nprobe 64.

**Re-rank depth** (`depths.sh`, `results/depths.csv`): full queries (router, 2-bit scan, SQ8 re-rank of the top 40),
router seed 0. Most of the remaining head cost came from asking the member search for 256 hits instead of 64:

| nprobe | rerank 256 | rerank 128 | rerank 64 |
|---:|---|---|---|
| 64 | 0.9311 @ 1.67 ms | 0.9311 @ 1.48 | 0.9295 @ 1.40 |
| 96 | 0.9463 @ 2.02 | 0.9461 @ 1.83 | 0.9456 @ 1.75 |
| 128 | 0.9565 @ 2.37 | 0.9544 @ 2.18 | 0.9544 @ 2.09 |
| 192 | 0.9641 @ 3.05 | 0.9641 @ 2.85 | 0.9641 @ 2.80 |
| 256 | 0.9685 @ 3.72 | 0.9685 @ 3.57 | 0.9685 @ 3.48 |

Rerank 64 is the faster curve from 0.93 up, so it is now the library default (commit 2881397). It beats Faiss
IVF-HNSW 128K at the bottom of the curve too: 0.9456 @ 1.75 ms vs 0.943 @ 1.98 ms, and 0.9544 @ 2.09 vs 0.954 @ 2.94.
That removes the low-recall tie of #52. The router studies used seed 42, which gives 0.9544 at nprobe 128 with rerank
256; seed 0 here gives 0.9565.
