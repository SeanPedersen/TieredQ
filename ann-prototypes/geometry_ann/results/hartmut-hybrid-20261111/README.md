# Two-stage re-rank: TieredQ tier in RAM, then exact SQ8 rows from SSD (#49, #52)

Remote root: `/home/sean/experiments/TieredQ-issue49-hybrid-20261111` (TieredQ 2bec400). From-scratch 64K index,
router at expansion 64 / at least 32 supers / rerank 256, 750 held-out queries alone on pinned core 2.
`hybrid_qwen.cpp` (the nested driver plus `sq8_rows=`) writes the SQ8 store as a `TieredQFileRows` file on the NVMe
SSD. The 6- and 8-bit fine tiers in RAM rescore the 2-bit scan's top 40, then their best 12/16/20/30 are rescored
with exact SQ8 rows read with direct I/O, 32 in flight. The library now ships this stage as
`TieredQNestedIVFIndex::attach_exact` (measured in `hartmut-shipping-20261111`).

**Result: SQ8 recall at every nprobe, from 7.84 GB.** With 16 finals, the 6-bit tier recovers the in-RAM SQ8 re-rank's
recall exactly. 12 finals lose at most 0.0001. The 8-bit tier gives the same recall from 10.1 GB, so it buys nothing.

| nprobe | Recall (SQ8 = two-stage) | SQ8 in RAM, 12.2 GB | 6-bit tier alone, 7.84 GB | 6-bit + 16 exact from SSD, 7.84 GB |
|---:|---:|---:|---:|---:|
| 96 | 0.9457 | 2.15 ms | 2.47 ms (0.9421) | 2.90 ms |
| 128 | 0.9544 | 2.48 ms | 2.83 ms (0.9503) | 3.18 ms |
| 256 | 0.9697 | 3.82 ms | 4.17 ms (0.9656) | 4.55 ms |
| 512 | 0.9788 | 6.53 ms | 6.87 ms (0.9747) | 7.20 ms |
| 1024 | 0.9856 | 11.8 ms | 12.0 ms (0.9816) | 12.3 ms |

The SQ8 baseline's 12.2 GB counts its memory-mapped store (9.05 GB) plus 3.13 GB anonymous; the two-stage path keeps
only the anonymous 7.84 GB (IVF, router, 4.5 GB tier) resident. The exact stage costs about 0.35 ms over the tier
alone. A few p99 values reach 13-34 ms, which are SSD tail latencies, against 3-14 ms otherwise.
