# #45: e-means revival threshold 0.3 vs 0.1, full TieredQNestedIVFIndex builds (hartmut, 2026-11-18)

`bench.sh` (as the emeans-stream study's): #52 defaults, 8.84M x 1024 Qwen/MS MARCO, 64K lists, automatic e-means,
32 build threads from the SQ8 memmap, 750 held-out queries on pinned CPU 2. Source `dd4999d` plus the experiment-only
`TQ_EMEANS_UNUSED_THRESHOLD` override. Order ABBA: 0.1 s0, 0.3 s0, 0.3 s7, 0.1 s7.

| | 0.1 s0 | 0.3 s0 | 0.1 s7 | 0.3 s7 |
|---|---:|---:|---:|---:|
| Build (train / assign) | 179.0 s (62.5 / 55.0) | 176.2 (62.3 / 53.6) | 174.8 (62.1 / 51.8) | 176.7 (62.4 / 53.5) |
| Peak build RSS | 15.35 GB | 15.35 | 15.38 | 15.38 |
| R@10 @ p50, nprobe 96 | 0.9472 @ 2.27 ms | 0.9424 @ 2.26 | 0.9448 @ 2.42 | 0.9411 @ 2.30 |
| nprobe 128 | 0.9556 @ 2.56 | 0.9499 @ 2.49 | 0.9535 @ 2.71 | 0.9493 @ 2.55 |
| nprobe 256 | 0.9699 @ 3.55 | 0.9645 @ 3.45 | 0.9675 @ 3.88 | 0.9629 @ 3.54 |
| nprobe 1024 | 0.9836 @ 8.98 | 0.9813 @ 8.50 | 0.9852 @ 9.50 | 0.9819 @ 8.98 |

0.3 loses 0.002-0.006 recall per nprobe on both seeds for 1-5% lower latency. At equal p50 it is -0.004 (seed 0)
and +0.0007 (seed 7, where the 0.1 run was slower across the board). Not a win: the 0.1 default stays. The 5% lower
imbalance in the train-only check did not outweigh the 0.2-0.3% worse objective.

## Final-assignment pruning in full builds (`bench-prune.sh`, `results/prune*`)

Same harness, revival 0.1, `TQ_PRUNE_ASSIGN_DIMS` 0 vs 224 (the pruning prototype of
`../hartmut-pruning-20261118`, final assignment only). Order: 0 s0, 224 s0, 224 s7, 0 s7.

| | 0 s0 | 224 s0 | 0 s7 | 224 s7 |
|---|---:|---:|---:|---:|
| Build (tier write excluded) | 175.3 s | **160.4** | 174.3 | **160.4** |
| Train / assign / encode | 61.9 / 53.7 / 49.4 | 62.6 / **38.5** / 48.8 | 62.2 / 51.6 / 49.5 | 62.2 / **37.6** / 50.2 |
| Peak build RSS | 15.35 GB | 15.48 | 15.38 | 15.52 |
| R@10 nprobe 96 / 256 / 1024 | 0.9472 / 0.9699 / 0.9836 | identical | 0.9448 / 0.9675 / 0.9852 | identical |

Recall@10 is identical at every nprobe on both seeds (p50 within host noise). Builds are 8.5% faster (assignment -28%).
Peak RSS +0.13 GB is the prototype's row-major binary16 member copy; rescoring from the existing panels instead
would remove it.
