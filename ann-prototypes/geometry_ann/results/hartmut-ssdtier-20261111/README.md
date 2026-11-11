# Low-RAM TieredQ tiers: on SSD at fewer reads, and the tier-only ceiling (#49)

Remote root: `/home/sean/experiments/TieredQ-issue49-ssdtier-20261111`, TieredQ 2881397: router at the library
defaults (rerank 64), from-scratch 64K index, router seed 42, 750 held-out queries alone on pinned core 2.
`ssdtier_qwen.cpp` is the shipping driver with lists of SSD candidate counts and read modes. Tiers on the NVMe SSD
(Samsung 960 EVO behind LUKS) use direct I/O with 32 reads in flight. Nothing SQ8 is resident: anonymous RSS is
3.35 GB (IVF codes, router and scan buffers).

**20 reads are enough.** Recall is identical at 20, 30 and 40 candidates for both tiers (`results/ssd{6,8}-64k`), and
every 10 reads cost about 0.14 ms. 8-bit fine tier on SSD, 20 candidates, against Faiss IVF-HNSW 128K (1.15.1, 13.6 GB):

| nprobe | Recall | p50 / p99 | Faiss at that recall |
|---:|---:|---:|---:|
| 96 | 0.9436 | 2.36 / 5.72 ms | 1.98 ms (0.943) |
| 128 | 0.9513 | 2.67 / 2.93 ms | ~2.83 ms at 0.954 vs Faiss 2.94 ms |
| 192 | 0.9628 | 3.35 / 3.72 ms | 3.87 ms (0.963) |
| 256 | 0.9685 | 4.04 / 4.50 ms | 5.69 ms (0.968) |
| 512 | 0.9776 | 6.63 / 10.0 ms | ~10.5 ms (interpolated) |
| 1024 | 0.9843 | 11.6 / 13.1 ms | ~17.9 ms (interpolated) |

The 6-bit tier on SSD is 0.003 lower at the same nprobe (0.9484 @ 2.69 ms at 128, 0.9816 @ 11.7 ms at 1024) with
2.3 GB less disk.

**Tier-only ceiling** (`results/ceiling-64k`, tiers in RAM, 40 candidates):

| nprobe | SQ8 re-rank | 8-bit tier | 6-bit tier |
|---:|---|---|---|
| 2048 | 0.9889 @ 19.2 ms | 0.9880 @ 19.5 | 0.9849 @ 19.5 |
| 4096 | 0.9927 @ 35.2 | 0.9916 @ 35.7 | 0.9885 @ 35.6 |

The 8-bit tier passes 0.99 between nprobe 2048 and 4096 (about 28.5 ms interpolated, against Faiss's 0.9903 @ 44.4 ms).
The 6-bit tier does not reach 0.99 by nprobe 4096.

**Low-RAM configuration:** 8-bit tier on SSD, 20 reads. Serving RAM is 3.35 GB, about 4× less than Faiss. Disk is
about 9.7 GB (2.9 GB IVF, 6.8 GB tier), against about 9.4 GB for Faiss. Build is about 370 s (232 s IVF plus 130 s tier
encoding). Its p50 is level with Faiss at 0.95-0.954 and 1.16-1.6× faster from 0.963 to 0.99. It is 1.19× slower at
0.944. Batch QPS was not measured.
