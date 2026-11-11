# Nested RAM after the heap fixes, VIBE arxiv k = 100 (#41)

Remote root: `/home/sean/experiments/TieredQ-issue41-nestedram-20261111`. `vibe_arxiv.py` is the
`../hartmut-arxivfollowup-20261111` harness (USS growth across `fit()`, cold process, THP off, single queries on pinned
CPU 2, best of 5 passes, VIBE's 1e-3 tolerance). It adds `sample_seed` for nested builds and glibc's `mallinfo2` after
`fit()`. The nested default runs at 86f003d's automatic 2,048 lists: 1-bit prefix, 8-bit fine tier on the NVMe SSD
with direct reads, 200 candidates (`candidates_for(100)`). Seeds are the builder's own (`-default`) and a seeded
sample of 131,072 rows passed as `sample=` (`-s1`), which also trains the codec on more rows. The `-fix` models rerun
both on the heap fix of `../hartmut-heapretention-20261111`. `REPORT.md` comes from `summarize.py`.

## The estimate of 0.19 GiB

The estimate for 86f003d is refuted: it measures **0.241 / 0.250 GiB**. `write_tier`'s trim did not release the 30
thread arenas of 2.3 MiB (67.9 MiB). With the fix (encoder buffers from the calling thread; trims after the buffers
are freed) it measures **0.178 / 0.183 GiB**.

| Recall@100 ≥ 0.98, best setting | USS | QPS/GB |
|---|---:|---:|
| nested 86f003d, seed 42 / sample 1 | 0.241 / 0.250 GiB | 1,293 / 1,257 (np48: r 0.985 at 312 / 314 QPS) |
| nested with the fix, seed 42 / sample 1 | 0.178 / 0.183 GiB | 1,614 / 1,482 (np64: r 0.991 at 287 / 271 QPS) |
| rabitq-ivf 4 bits, 2,048 lists (`../hartmut-arxivdefaults-20261111`) | 0.760 GiB | 1,342 (r 0.983, 1,020 QPS) |

Recall is identical with and without the fix at every nprobe, and the two seeds agree within 0.0005.

**QPS caveat.** The `-fix` serves overlapped other load on hartmut: the embeddinggemma worker was at 130% CPU and IO
pressure averaged 3%. This path makes 200 direct SSD reads per query, and nprobe 48 measured below nprobe 64 (271 and
206 QPS against 287 and 271). At the 86f003d serves' 312 QPS, the fixed USS would give about 1,750 QPS/GB at r 0.985.
Either way nested passes rabitq-ivf at ≥ 0.98 (1,482–1,614 against 1,342). A quiet-host rerun of the `-fix` serves
goes with the #52 quiet-host window.

What remains of the 0.178 GiB: `storage_bytes` 0.135 GiB (the prefix IVF; the refinement adds 4.9 MB serialized),
live uncounted state of about 16 MiB (the id → document map, encoder caches, the refinement model; now counted by
`storage_bytes`), and about 27 MiB of one-time process state that VIBE charges to the first index (the 20.7 MiB region,
OpenBLAS, the module).

Seed 1's USS-per-step figures are negative because its 384 MiB training sample was allocated before the "start"
reading and freed during `build`. Its total growth is measured from the baseline before `fit()`, so it is correct.
