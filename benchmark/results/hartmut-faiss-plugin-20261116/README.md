# #55 acceptance: IndexIVFTieredQ (faiss-tieredq) on hartmut

Remote roots: `/home/sean/experiments/TieredQ-issue55-faiss-20261116` (source 6f75020) and
`TieredQ-issue55-highbits-20261116` (source ec7beec, `high-bits/`), 2026-11-16, under the shared lock. Faiss v1.16.0
was built from source there (`FAISS_OPT_LEVEL=avx512`, OpenBLAS) and the plugin links `faiss_avx512`. The driver is
`faiss-tieredq/benchmarks/acceptance.cpp`; `study.py` runs it and `summarize.py` writes `REPORT.md`. Run it as
`summarize.py results high-bits/results`. The per-query hit files stayed on hartmut, and the indexes were removed.

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
  1.3% of RAM, and the plugin is 3.3× faster. A 3-bit Qwen configuration, likely below both, was not run. TieredQ's
  shipping low-RAM configuration (#52: 1-bit IVF with the 8-bit tier on SSD, about 1.7 GB) belongs to phase 2 here
  (`IndexRefine`).
- **Under recall 0.81, Faiss's 1-bit codecs use about 40% less RAM on arxiv.** Below the lowest TieredQ budget
  measured, RaBitQ 1-bit is the smallest index.

## Superseded first round (`results-v1-fp32-quantizer/`)

With `IndexFlatL2` quantizers and 64-bit list ids, the plugin missed both bars on arxiv: RAM 1.061 and QPS
0.79-0.98. The flat fp32 quantizer cost about 90 us per query more than TieredQ's binary16 ranking (hartmut,
2,048 x 768: 246 us against 134 us; `IndexScalarQuantizer` QT_fp16, 131 us). The ids took 8 bytes per row against
TieredQ's compact ids. The Qwen seed-1 build of that round was stopped. The Faiss codecs' results there are
consistent with the final round.
