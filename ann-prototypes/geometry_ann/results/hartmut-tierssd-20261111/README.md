# 8-bit TieredQ tier: more candidates, and on SSD (#49)

Remote root: `/home/sean/experiments/TieredQ-issue49-tierssd-20261111` (TieredQ 2bec400). From-scratch 64K index of the
router study, router at expansion 64 / at least 32 supers, 750 held-out queries alone on pinned core 2. `nested_qwen.cpp`
is the nested study's driver; `study.py` runs one model: the SQ8 re-rank and the 8-bit fine tier in RAM at 40/60/100
candidates, then the same tier written to the NVMe SSD (Samsung 960 EVO behind LUKS) and read through the page cache
and with direct I/O at queue depth 1 and 32 (60 candidates).

**More candidates do not help.** Recall is identical at 40, 60 and 100 candidates for both the SQ8 re-rank and the tier
(0.9457 / 0.9544 / 0.9697 / 0.9788 / 0.9856 for SQ8 at nprobe 96-1024; 0.9448 / 0.9532 / 0.9685 / 0.9776 / 0.9844 for
the tier). The 2-bit scan's top 40 already holds every true neighbour it finds, so the tier's 0.0012 loss is its
rescoring precision alone. That led to the exact final stage (`hartmut-hybrid-20261111`).

**Tier on SSD** (anonymous RSS 3.35 GB: IVF codes, router and the scan):

| nprobe | Recall | RAM tier, 60 candidates | Direct, depth 1 | Direct, depth 32 |
|---:|---:|---:|---:|---:|
| 128 | 0.9532 | 2.89 ms | 13.7 ms | 4.94 ms (p99 12.7) |
| 256 | 0.9685 | 4.28 ms | 15.2 ms | 5.20 ms (p99 6.0) |
| 1024 | 0.9844 | 11.7 ms | 20.7 ms | 13.2 ms (p99 15.0) |

One read at a time costs about 180 us per candidate on this encrypted NVMe; 32 in flight cost 20-35 us. Page-cached
reads ran at RAM speed only because the freshly written 6.8 GB file was still cached, so they are not an SSD result.
