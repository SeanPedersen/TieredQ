# Shipping configuration through the library (#47, #49, #52)

Remote root: `/home/sean/experiments/TieredQ-issue52-shipping-20261111`, TieredQ 2881397: router at the library
defaults (expansion 64, at least 32 supers, rerank 64, `pack_panel_rows`), the exact stage
`TieredQNestedIVFIndex::attach_exact` over `TieredQSQ8Rows` on a `TieredQFileRows` copy of the SQ8 store (NVMe, direct
I/O, 32 reads in flight). From-scratch 64K index, router seed 42, 750 held-out queries alone on pinned core 2.
`shipping_qwen.cpp` is the hybrid driver with the library's exact stage in place of its own.

| nprobe | SQ8 re-rank in RAM, 12.2 GB | 6-bit tier alone, 7.84 GB | 6-bit tier + 16 exact SQ8 from SSD, 7.84 GB |
|---:|---|---|---|
| 64 | 0.9275 @ 1.36 ms (p99 1.61) | 0.9239 @ 1.65 | 0.9271 @ 2.08 (p99 2.65) |
| 96 | 0.9445 @ 1.71 (1.94) | 0.9409 @ 2.02 | 0.9443 @ 2.44 (3.48) |
| 128 | 0.9525 @ 2.07 (2.32) | 0.9484 @ 2.37 | 0.9520 @ 2.81 (3.34) |
| 192 | 0.9640 @ 2.73 (3.08) | 0.9599 @ 3.05 | 0.9635 @ 3.51 (4.35) |
| 256 | 0.9697 @ 3.43 (3.88) | 0.9656 @ 3.74 | 0.9692 @ 4.17 (4.92) |
| 512 | 0.9788 @ 6.01 (6.74) | 0.9747 @ 6.30 | 0.9783 @ 6.80 (7.68) |
| 1024 | 0.9856 @ 10.9 (12.4) | 0.9816 @ 11.2 | 0.9852 @ 11.7 (13.2) |

Against the rerank-256 router of the earlier studies, the SQ8 curve moved left. 0.954 now takes about 2.16 ms
(interpolated) instead of 2.48 ms, and 0.9697 takes 3.43 instead of 3.82 ms.

**The exact stage's 0.0002-0.0005 gap is summation order, not the design.** 24 and 32 finals give the same recall as
16 (`results/shipping-64k-finals`). Scoring the same tier top 16 with float sums, as the SQ8 sweep does (`hybrid-float`
in `results/shipping-64k-precision`), reproduces the SQ8 recall exactly at every nprobe. The library sums in double.
The truth is exact on the fp32 originals, so neither summation is more correct: the difference is 3-4 of 7,500 hits,
decided by near-ties at rank 10 against fp32 neighbours. The library keeps double sums.
