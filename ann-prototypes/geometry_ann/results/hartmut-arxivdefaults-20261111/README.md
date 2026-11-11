# b0148dd defaults on VIBE arxiv, k = 100 (#41, #43, #42, #34)

Remote root: `/home/sean/experiments/TieredQ-issue41-arxivdefaults-20261111`, TieredQ b0148dd (Python package,
Release). VIBE `arxiv-nomic-768-normalized`, 1,344,643 × 768, 1000 queries, k = 100. `vibe_arxiv.py` is the #41 VIBE-like
harness updated for the b0148dd API. Index size is USS growth across `fit()` (GiB, VIBE's divisor). Each query is a
separate call on pinned CPU 2 (sibling idle), QPS is the best of 5 passes, and Recall@100 uses VIBE's 1e-3 distance
tolerance. Builds run on 32 threads, with THP disabled per process unless marked `host`. `study.py` runs one fresh
process per model under the shared lock. `REPORT.md` (from `summarize.py`) has every curve. The host had other load
(load average 2-5) but nothing else on CPUs 2/20.

Models: flat TieredQ IVF at 1.75 and 1.25 bits/dim. Each is run with automatic lists (`lists=None`, which picks 4,096
here) and with the #43 best counts (1,024 and 2,048). There are second partitions via `train_lists` on a sample seeded
1 (`-s1`), the pre-b0148dd 6-bit scan cap (`-cap6`), and 4,096 lists with the router forced on. Then the nested
default (1-bit prefix, 8-bit fine tier written to the NVMe SSD and read with direct I/O, 32 in flight), rabitq-ivf
(4 bits, 2,048 lists, VIBE's module), and the automatic build with THP `always`.

## Best QPS/GB at fixed Recall@100 (VIBE-style: the best setting that reaches the target)

| model | r ≥ 0.90 | r ≥ 0.95 | ceiling |
|---|---|---|---|
| flat 1.75, 2,048 lists | **7,303** (np16, 1,989 QPS) | **3,864** (np48, r 0.953, 1,053 QPS) | 0.965 |
| flat 1.75, 2,048, seed 1 | 7,151 | 3,795 | 0.964 |
| flat 1.75, 1,024 | 7,291 | 3,714 | 0.961 (np48) |
| flat 1.75, automatic (4,096) | 5,372 (np24, 1,618 QPS) | 2,628 (np96, r 0.958, 792 QPS) | 0.966 |
| flat 1.75, 4,096, seed 1 | 5,221 | 2,532 | 0.966 |
| flat 1.75, 4,096 + router | 4,309 | 2,228 | 0.966 |
| flat 1.25, 1,024 | **7,501** (np24, r 0.913) | — | 0.930 |
| flat 1.25, automatic (4,096) | 5,877 | — | 0.935 |
| nested default (1-bit + 8-bit tier on SSD) | 1,393 (np128, 100 cand.) | 1,097 (np64, 200 cand., r 0.980, 296 QPS) | 0.998 |
| rabitq-ivf 4 bits, 2,048 | 2,411 (np16, r 0.920) | 1,897 (np32, r 0.962, 1,443 QPS) | 0.983 (np64) |

## Findings

- **The automatic list count is wrong for arxiv at k = 100.** `automatic_lists(1.34M)` picks 4,096 lists, which
  is 26-32% below 2,048 lists on QPS/GB at both targets, in both seeds. Seed spread is about 2%. At equal recall
  (0.953) it is about 15% slower: about 900 QPS interpolated between np64 and np96, against 1,053. It also takes about
  11% more USS (0.301 vs 0.272 GiB). The difference is mostly fixed cost: ranking 4,096 binary16 centroids instead of
  2,048 adds about 150 us per query. #43 measured the same before b0148dd. The rule came from Qwen 1M at recall@10,
  where 4K lists won. Whether k or the dataset makes the difference is open (next study).
- **The router does not help at 4,096 lists.** It is 1-9% slower (most at low nprobe) and adds 47 MiB, so the 8K threshold stands.
- **The 4-bit scan cap is neutral.** Against the 6-bit cap it loses 0.0006-0.0012 recall at matched nprobe and is up to
  2% faster, for +1.7% QPS/GB at 0.95 (2,048 lists). Keep it.
- **rabitq-ivf**, rerun on the same day, matches #41 (0.760 GiB with THP off). Flat TieredQ at 2,048 lists leads on
  QPS/GB by 3.0× at 0.90 and 2.0× at 0.95. rabitq is 1.37× faster per query at 0.95 and reaches 0.98, where flat
  TieredQ at 1.75 bits stops at 0.965.
- **Nested on SSD at k = 100.** The 100 candidates of `candidates_for(100)` cap recall at about 0.905, so k = 100
  needs more candidates. 200 candidates reach 0.980 at np64 and 0.994 at np128, within 0.001 of 400 candidates and
  1.5-2.6× faster. The default of 4k = 400 is twice what is needed. It is SSD-bound: 200 direct reads put a query at
  about 3.4 ms (296 QPS), against about 1 ms for flat. At ≥ 0.98, nested reaches 1,097 QPS/GB against rabitq's 1,342
  (np64, 1,020 QPS).
- **Nested RAM is not what `storage_bytes` says.** USS growth is 0.269 GiB, but `storage_bytes` (prefix IVF only) is
  0.141 GiB. The remaining 0.128 GiB is most likely the refinement model (8-bit tier codebooks), which
  `storage_bytes` leaves out. If it does not grow with rows, it is about 8% of the 1.72 GB serving RAM at 8.84M, but
  half the index here. Not yet confirmed.
- **Flat USS sits about 45 MiB above `storage_bytes`** at every list count with THP off (0.272 vs 0.228 GiB at 2,048).
  #43 measured 0.254 GiB for the same configuration at 2249246. The difference is unattributed: module, OpenMP or
  allocator.
- **#42: THP `always` no longer inflates this build.** The automatic build measured 0.308 GiB under the host's THP
  `always` against 0.301 GiB with THP off (+7 MiB, against +60 MiB in #42). AnonHugePages grew by 264 MiB. This is not
  yet the issue's exact configuration (2 bits, 2,048 lists).
- **#34 list padding:** 1.2% / 2.4% / 4.7% of rows at 1,024 / 2,048 / 4,096 lists. That fits 16 × lists ÷ rows
  (about 15.4 padding rows per list).

## Not measured here

Recall@10 on arxiv (to separate k from dataset in the list-count rule), the #42 original configuration, a breakdown
of the nested and flat USS gap, nested at 150 candidates, and the quiet-host Qwen rerun.
