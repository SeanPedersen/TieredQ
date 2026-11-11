# Default thresholds and the 1-bit RAM lever (#52)

Remote root: `/home/sean/experiments/TieredQ-issue52-defaults-20261111`, TieredQ 9b158c6 through its explicit calls
(`defaults_qwen.cpp`). Nested 8-bit fine suffix tier, 750 held-out queries alone on pinned core 2. Raw rows are in
`results/*/run.csv` and build times in `run.json`.

## threshold-1m: first 1M Qwen rows (truth: exact L2 over them), 20 candidates, tier in RAM

Recall@10 @ p50 (ms). current = 20 Lloyd iterations, exact assignment, 64 rows/list; fast = 4 iterations, 256 x 16
restricted assignment, 40 rows/list.

| Lists | current, exact ranking | fast, exact ranking | fast + router | list training current -> fast |
|---|---|---|---|---|
| 4K | 0.9483@1.42, 0.9680@2.10, 0.9817@3.41 | 0.9404@1.48, 0.9676@2.16, 0.9807@3.50 | 0.1 ms slower | 33 -> 2.6 s |
| 8K | 0.9328@1.70, 0.9561@2.23, 0.9736@3.26 | 0.9329@1.70, 0.9571@2.21, 0.9732@3.26 | 0.9299@1.43, 0.9575@2.08, 0.9736@3.00 | 136 -> 5.5 s |
| 16K | 0.9449@3.16, 0.9656@4.00, 0.9784@5.70 | 0.9455@3.17, 0.9648@3.94, 0.9765@5.66 | 0.9435@2.18, 0.9652@3.49 | 500 -> 13 s |
| 32K | 0.9387@5.39, 0.9595@6.11, 0.9735@7.50 | 0.9387@5.36, 0.9577@6.10, 0.9727@7.56 | 0.9335@1.86, 0.9565@2.80, 0.9728@4.61 | 1067 -> 25 s |

Decisions: router and fast training from 8,192 lists. From 8K lists up, the router matched recall and was faster, and
fast training stayed within 0.002 from recall 0.93. At 4K the router was slower and fast training lost up to 0.008.
4K lists gave the best curve at 1M, so the automatic list count uses 256 rows/list at 1M, falling log-linearly to 135 (superseded: 2K beat 4K at 1M in `../hartmut-qwen1mlists-20261111`, so the rule now uses 560 rows/list at 1M)
at 8.84M.

## onebit-64k: 1-bit prefix, 64K lists, all 8.84M rows, 8-bit tier on NVMe (direct, 32 in flight)

Build 290 s (87 s lists, 112 s IVF, 90 s tier). Anonymous RSS 2.03 GB (2-bit winner: 3.35 GB).

| nprobe | 1-bit, 20 / 30 / 40 candidates | 2-bit winner, 20 (hartmut-ssdtier) |
|---:|---|---|
| 96 | 0.9389@1.95, 0.9435@2.04, 0.9437@2.19 | 0.9436@2.36 |
| 128 | 0.9463@2.15, 0.9511@2.27, 0.9513@2.38 | 0.9513@2.67 |
| 192 | 0.9572@2.65, 0.9625@2.78, 0.9627@2.89 | 0.9628@3.35 |
| 256 | 0.9625@3.14, 0.9683@3.26, 0.9684@3.37 | 0.9685@4.04 |
| 512 | 0.9708@4.89, 0.9771@4.98, 0.9773@5.14 | 0.9776@6.63 |
| 1024 | 0.9776@8.06, 0.9836@8.23, 0.9843@8.36 | 0.9843@11.6 |
| 2048 / 4096 | 0.9876@13.1 / 0.9912@22.1 (40) | 0.99 near nprobe 3000 |

The 1-bit prefix at 30-40 candidates matches the 2-bit winner's recall at every nprobe, uses 39% less RAM, and is
15-28% faster. It passes 0.99 by nprobe 4096 (22 ms). At 20 candidates it loses about 0.005, so it needs 30-40 reads.
It wins on recall per RAM at competitive latency, so it is the default (`TieredQNestedConfig::prefix_bits` 0 = 1
bit/dim; candidates max(40, 4k)). The 2-bit prefix stays available (`prefix_bits = 2 * d`, 20 candidates).
