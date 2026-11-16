# #55 acceptance: IndexIVFTieredQ (faiss-tieredq) on hartmut

Remote roots: `/home/sean/experiments/TieredQ-issue55-faiss-20261116` (source 6f75020) and
`TieredQ-issue55-highbits-20261116` (source ec7beec, `high-bits/`), `TieredQ-issue55-minram-20261116` (source
614b87e, `minram/`) and `TieredQ-issue55-bisect-20261116` (source 3623bea, `bisect/`), 2026-11-16, under the shared
lock. Faiss v1.16.0 was built from source there (`FAISS_OPT_LEVEL=avx512`, OpenBLAS) and the plugin links
`faiss_avx512`. The driver is `faiss-tieredq/benchmarks/acceptance.cpp`; `study.py` runs it and `summarize.py` writes
`REPORT.md`. Run it as `summarize.py results high-bits/results minram/results`. `bisect/bisect.py` wrote
`bisect/bisect.json`. `ram-levers/` is `TieredQ-issue55-ram-20261116` (source ed853a0). The per-query hit files
stayed on hartmut, and the indexes were removed.

## Setup

- **One partition per seed, shared by every index.** Faiss k-means (L2, seeded) on a sample of 64 rows per list for
  arxiv (2,048 lists, 10 iterations) and 40 rows per list for Qwen (65,536 lists, 4 iterations). `TieredQIVFIndex`
  takes the centroids with `set_lists` and builds. Every other index gets TieredQ's row assignment: the plugin through
  `add_core`, Faiss's codecs through a quantizer that hands those lists out. Encoders train on the same 32,768 rows,
  the sample TieredQ's own Python build uses. The partition reproduces exactly: a high-bit-study rerun of arxiv seed 1
  at 1.75 bits matched the base study's recall at every nprobe.
- **Indexes.** `TieredQIVFIndex` and `IndexIVFTieredQ` at 1.75, 2.5, 3 and 4 bits/dim (arxiv), and 1, 2 and 4 bits/dim
  (Qwen). Faiss's `IndexIVFRaBitQ` at 1, 2 and 4 bits (default query bits 4), and `IndexIVFPQFastScan` with 4-bit PQ
  at 1 and 2 bits/dim (M = d/4, d/2). No refinement for any of them.
- **Coarse ranking.** Faiss-side indexes (plugin and codecs alike) rank lists on binary16 centroids, as TieredQ does:
  `IndexScalarQuantizer(QT_fp16)` on arxiv and `IndexHNSWSQ(QT_fp16, M = 32)` on Qwen, with efSearch max(64,
  2 × nprobe) and no other value tried. TieredQ uses its own exact ranking (arxiv) or its router (Qwen, 64K lists).
- **Serving.** Each index runs in a fresh process. RAM is the growth of RssAnon across load and coarse setup, after
  `malloc_trim`. Then the process runs alone on CPU 2 with one thread. Each query is a separate call, QPS is the best
  of 3 passes, and recall is id overlap with the true top k: arxiv recall@100 over 1,000 queries, Qwen recall@10 over
  750 held-out queries. Recall is not VIBE's distance-tolerant recall, so arxiv values sit below #41's.
- **Parity.** 50 queries per nprobe and at full probe: `TieredQIVFIndex` with exact coarse ranking against the plugin
  with its exact fp16 quantizer.
- **Seeds.** Arxiv uses 3 partition seeds and Qwen 2 (1 for the 2- and 4-bit runs). `results-v1-fp32-quantizer/` is
  the first arxiv round with `IndexFlatL2` quantizers and 64-bit list ids; it is superseded, see below.

## Acceptance (issue #55, step 4)

- **Same results as `TieredQIVFIndex`.** At every nprobe up to full probe, at least 49 of 50 queries return identical
  ids and scores (arxiv). On Qwen all 50 do, at every bit width. The odd query out is a coarse near-tie that probes a
  different list. Same-nprobe recall agrees to 0.0001 on arxiv and 0.0033 on Qwen, where the plugin's coarse ranking
  is the HNSW graph instead of TieredQ's router.
- **RAM within 5%.** Plugin over TieredQ: 1.018-1.026 on arxiv (all bit widths), and 0.921 / 0.977 / 1.013 on Qwen at
  1 / 2 / 4 bits. Lists sit in blocks at their size rounded to 16 rows (TieredQ pads to 32), ids take 32 bits, and the
  quantizer is fp16.
- **QPS within 10% at equal recall.** Arxiv: 1.03-1.07 at 1.75 bits and 1.06-1.14 at 2.5-4 bits (equal recall at
  equal nprobe), over all seeds. Qwen at 1 bit and the same nprobe, seeds 1 and 2: 1.36 at nprobe 64, 1.14 at 128,
  1.03-1.04 at 256, 0.94-0.95 at 512, and 0.89 at 1024, where recall agrees within 0.0015. At 2 and 4 bits:
  0.96-1.42 and 1.23-1.52. Interpolated to TieredQ's exact recall, one Qwen point reads 0.84 (seed 2, nprobe 512),
  because recall changes by 0.0034 between nprobe 512 and 1024. The only same-nprobe point outside 10% is Qwen 1-bit
  at nprobe 1024 (0.89), where HNSW at efSearch 2,048 costs more than TieredQ's router. efSearch is the plugin user's
  knob and was not tuned.

## Against Faiss's codecs: recall per RAM (#41)

Best QPS (QPS/GiB) among settings reaching the target, means over seeds. Arxiv uses k = 100:

| index | RAM GiB | ceiling | r ≥ 0.90 | r ≥ 0.95 |
|---|---|---|---|---|
| RaBitQ 1-bit / PQ fast scan 1 bit/dim | 0.14 | 0.81 / 0.57 | — | — |
| **plugin 1.75 bits** | 0.230 | 0.939 | **1,517 (6,585)** | — |
| RaBitQ 2-bit | 0.281 | 0.905 | 331 (1,177) | — |
| **plugin 2.5 bits** | 0.320 | 0.969 | 1,255 (3,917) | **762 (2,378)** |
| plugin 4 bits | 0.503 | 0.983 | 933 (1,854) | 684 (1,359) |
| RaBitQ 4-bit | 0.523 | 0.966 | 530 (1,015) | 294 (562) |

Qwen 8.84M uses k = 10; the 2- and 4-bit rows are seed 1 only:

| index | RAM GiB | ceiling | r ≥ 0.85 | r ≥ 0.90 | r ≥ 0.95 |
|---|---|---|---|---|---|
| RaBitQ 1-bit | 1.344 | 0.849 | — | — | — |
| **plugin 1 bit** | 1.352 | 0.874 | **755 (558)** | — | — |
| **plugin 2 bits** | 2.465 | 0.942 | 1,084 (440) | **627 (254)** | — |
| RaBitQ 2-bit | 2.497 | 0.915 | 867 (347) | 183 (73) | — |
| PQ fast scan 2 bits/dim | 2.711 | 0.844 | — | — | — |
| RaBitQ 4-bit | 4.618 | 0.964 | 558 (121) | 558 (121) | 78 (17) |
| **plugin 4 bits** | 4.691 | 0.976 | 832 (177) | 832 (177) | **259 (55)** |

- **Arxiv: TieredQ holds the frontier from recall 0.85 up.** At every target the plugin needs the least RAM and is
  the fastest. Recall 0.95 takes 0.320 GiB at 762 QPS, against RaBitQ 4-bit's 0.523 GiB at 294 QPS. At 4 bits the
  plugin is smaller than RaBitQ 4-bit and reaches 0.983 against 0.966. Faiss is smaller only below recall 0.81.
- **Qwen: TieredQ needs the least RAM at 0.85 and 0.90, with one near-tie at 0.95.** At 0.85 the 1-bit plugin needs 46%
  less RAM than the next index that reaches it, RaBitQ 2-bit, at 13% fewer QPS (755 against 867). At equal RAM (2
  bits) it is 1.25× faster. At 0.90 the 2-bit plugin needs the least RAM and is 3.4× faster than RaBitQ 2-bit. At 0.95, RaBitQ 4-bit (4.62 GiB, 78 QPS) and the plugin at 4 bits (4.69 GiB, 259 QPS) are within
  1.3% of RAM, and the plugin is 3.3× faster. The 2.5- and 3-bit runs and the bisection below close this: 0.95 takes
  2.07 bits/dim (2.54 GiB). TieredQ's
  shipping low-RAM configuration (#52: 1-bit IVF with the 8-bit tier on SSD, about 1.7 GB) belongs to phase 2 here
  (`IndexRefine`).
- **Under recall 0.81, Faiss's 1-bit codecs use about 40% less RAM on arxiv.** Below the lowest TieredQ budget
  measured, RaBitQ 1-bit is the smallest index.

## RAM at RaBitQ's recall: bisection over TieredQ's bit budget (`bisect/`, `minram/`)

Every Faiss RaBitQ setting sets a target: its best recall over the nprobe grid, with 8-bit queries (its most
accurate setting; `minram/` shows 8-bit queries reach the same ceiling as the default 4, at 10-25% fewer QPS). The
grid goes to nprobe 256 of 2,048 lists on arxiv (recall@100) and 4,096 of 65,536 on Qwen (recall@10), on partition
seed 1. The Qwen runs here reuse the seed-1 centroids (`CENTROIDS`), which gives a byte-identical partition. The
bisection builds `TieredQIVFIndex` at the budget that interpolation over every measured point predicts, until the
smallest budget that reaches the target is within 1/8 bit of the largest that misses it. At that budget the plugin was
built and served too, and it reached the target every time.

| dataset | RaBitQ | target | RaBitQ RAM GiB | TieredQ bits/dim (misses at) | TieredQ GiB (ratio) | plugin GiB (ratio) |
|---|---|---|---|---|---|---|
| Qwen | 1 bit | 0.855 | 1.344 | 0.875 (0.75) | 1.345 (1.00) | **1.221 (0.91)** |
| Qwen | 2 bits | 0.917 | 2.497 | 1.5 (1.4375) | 1.995 (0.80) | **1.909 (0.76)** |
| Qwen | 3 bits | 0.950 | 3.555 | 2.0625 (2.0) | 2.589 (0.73) | **2.535 (0.71)** |
| Qwen | 4 bits | 0.971 | 4.618 | 3.25 (3.125) | 3.840 (0.83) | **3.857 (0.84)** |
| arxiv | 1 bit | 0.817 | 0.144 | 0.6875 (0.5625) | 0.095 (0.66) | **0.100 (0.69)** |
| arxiv | 2 bits | 0.908 | 0.281 | 1.25 (1.1875) | 0.164 (0.59) | **0.169 (0.60)** |
| arxiv | 3 bits | 0.950 | 0.402 | 2.0 (1.875) | 0.255 (0.63) | **0.261 (0.65)** |
| arxiv | 4 bits | 0.970 | 0.523 | 2.625 (2.5) | 0.329 (0.63) | **0.336 (0.64)** |

- **TieredQ IVF needs 31-40% less RAM on arxiv and 16-29% less on Qwen from 2 bits up.** At Qwen 1 bit only the
  plugin is smaller (9%), through its 32-bit ids and 16-row list padding, not through the codec.
- **It is also faster there.** Both reach their ceiling at the largest nprobe. At that point the matched plugin serves
  18-26 QPS against RaBitQ's 9-14 (Qwen) and 213-505 against 71-110 (arxiv).
- RaBitQ's arxiv ceilings vary by at most 0.0006 over 3 partition seeds (`minram/`), so one seed sets the targets. The
  TieredQ budgets come from seed 1 alone.
- `REPORT.md` merges `minram/` into the frontier tables. There the Qwen rows of `minram/` use the grid up to 4,096, and
  rabitq4's "3 seeds" mixes 2 runs to nprobe 1,024 with 1 to 4,096.

## RAM levers: residual encoding and training rows (`ram-levers/`)

Two plugin options, off by default, tested for more recall at the same RAM. With `by_residual` (`res`), the codec
encodes each row minus its list's fp16 centroid and scores add the query's inner product with that centroid. With
`training_rows` (`rows`), the codec trains on 262,144 rows instead of 32,768. `res-rows` uses both. Each ran at the
bisection's matched budgets, on partition seed 1 and the same nprobe grids, plugin only. The table gives best recall
against the bisection's plugin (`base`). RAM is equal within 0.1%.

| dataset | bits/dim | base | res | rows | res-rows |
|---|---|---|---|---|---|
| arxiv (recall@100) | 1.25 | 0.9113 | 0.8512 | 0.9126 | 0.8548 |
| | 2 | 0.9512 | 0.9210 | 0.9518 | 0.9228 |
| | 2.625 | 0.9719 | 0.9530 | 0.9720 | 0.9545 |
| Qwen (recall@10) | 1.5 | 0.9252 | 0.9219 | 0.9217 | 0.9221 |
| | 2.0625 | 0.9503 | 0.9468 | 0.9471 | 0.9508 |
| | 3.25 | 0.9740 | 0.9740 | 0.9712 | 0.9765 |

- **Residual encoding costs recall.** It loses 0.019-0.060 on arxiv and 0.003-0.004 on Qwen at 1.5 and 2.0625 bits,
  and ties at 3.25. So the smaller in-IVF saving on Qwen (see below) is not a missing residual mode. A likely cause, not
  measured: TieredQ's allocation exploits the data's few high-variance directions, which the list centroids remove.
  RaBitQ's random rotation does not depend on them.
- **More training rows do not help.** They gain at most 0.0014 on arxiv and lose 0.003 on Qwen.
- **Both together are mixed on Qwen.** They give −0.0031 / +0.0005 / +0.0025. The gain at 3.25 bits is worth about
  1/8 bit (3% of RAM) at the RaBitQ 4-bit match only. They did not reach the study's 0.002 mean gain, so no
  bisection ran.
- **The Qwen differences are not routing noise (`control.json`).** At 2.0625 bits the base plugin, rebuilt and served
  twice with a serially built HNSW graph and once with 32 threads, gave the same recall at every nprobe as the
  bisection. Rebuilt res and rows reproduced their screen values and stay below base at every nprobe (−0.0026 to
  −0.0040).

## Against the earlier RaBitQ comparisons

| study | search | data | TieredQ at RaBitQ's 1 / 2 / 4-bit recall, against RaBitQ |
|---|---|---|---|
| #4 ([issue4-current](../issue4-current-20261105/benchmark-report.md)), official RaBitQ | flat | 6 datasets, 100K-1M | disk 0.48-0.71×; Gemini-768 0.67 / 0.60 / 0.63, MS MARCO-1M (Cohere-1024) 0.59 / 0.53 / 0.56 |
| #6 ([issue6-ivf](../issue6-ivf-20261030/README.md)), official RaBitQ IVF | IVF, 64 lists, 1,560 rows/list | OpenAI-1536 100K | at full probe 1,424 code bits (0.93 bits/dim) reach 0.887, above RaBitQ 1-bit (0.826) and near 2-bit (0.912) |
| this study, Faiss `IndexIVFRaBitQ` | IVF, 2,048 lists, 650 rows/list | arxiv-768 1.34M | RAM 0.66 / 0.59 / 0.63 (3 bits 0.63); bits 0.69 / 0.63 / 0.66 |
| this study, Faiss `IndexIVFRaBitQ` | IVF, 65,536 lists, 135 rows/list | Qwen-1024 8.84M | RAM 1.00 / 0.80 / 0.83 (3 bits 0.73); bits 0.88 / 0.75 / 0.81 |

- **Arxiv matches the flat results.** Its ratios (0.59-0.66) fall in the flat codec range and next to the flat
  Gemini-768 rows (0.60-0.67). The in-IVF codec saving holds at 2,048 lists.
- **Qwen saves less than any flat dataset.** The matched code bits are 0.69-0.88× RaBitQ's, against 0.53-0.59× for
  flat MS MARCO-1M at the same width. Fixed per-row costs explain part of the RAM ratio: the fp16 centroids
  (about 121 bits per row at 64K lists), ids, factors and list padding add about 290 bits per row to the plugin and
  about 280 to RaBitQ (1-bit: 1,187 and 1,306 bits per row in total, against 896 and 1,024 code bits). The rest is
  in the codes. Faiss's RaBitQ encodes residuals to the list centroid and TieredQ does not, but a residual TieredQ
  codec loses recall on both datasets (see RAM levers above), so that is not what is missing. List size against
  dataset remains untested.
- **#41's larger RAM gap is a different comparison.** There, nested TieredQ (1-bit prefix in RAM, 8-bit tier on SSD)
  took 0.18 GiB at recall@100 ≥ 0.98 on arxiv, against 0.760 GiB for VIBE's rabitq-ivf (4 bits, RAM only). Faiss's
  RaBitQ 4-bit on the same data takes 0.523 GiB here. In RAM only, the codec saves about 35-40% on arxiv and
  20-30% on Qwen. The larger savings come from the SSD tier (phase 2, `IndexRefine`).

## Superseded first round (`results-v1-fp32-quantizer/`)

With `IndexFlatL2` quantizers and 64-bit list ids, the plugin missed both bars on arxiv: RAM 1.061 and QPS
0.79-0.98. The flat fp32 quantizer cost about 90 us per query more than TieredQ's binary16 ranking (hartmut,
2,048 x 768: 246 us against 134 us; `IndexScalarQuantizer` QT_fp16, 131 us). The ids took 8 bytes per row against
TieredQ's compact ids. The Qwen seed-1 build of that round was stopped. The Faiss codecs' results there are
consistent with the final round.
