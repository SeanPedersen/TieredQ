# Quick TieredQ normal-mode control

Full corpora, first 100 held-out queries, seed 0, 8 threads, batches 1/32, budgets 1/2/4. Both modes are freshly timed on the identical query subset, with three warmups and three repeats. This small control does not replace the main 1,000-query benchmark or its three-seed recall results.

Both builds use the same seed/training inputs and matching learned tiers. Normal mode uses the sixth native Index constructor argument `fast=False`; fast mode uses `fast=True`. Storage includes native serving state plus the SIFT metric-preserving adapter. Build times exclude the SIFT transform shared by both modes in this control.

| Dataset | Bits | Batch | Fast recall | Normal recall | Fast bytes | Normal bytes | Fast QPS | Normal QPS | Fast/normal QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sift | 1 | 1 | 0.4890 | 0.4920 | 27.57 | 21.30 | 2,108 | 212 | 9.96× |
| sift | 1 | 32 | 0.4890 | 0.4920 | 27.57 | 21.30 | 2,675 | 1,539 | 1.74× |
| sift | 2 | 1 | 0.7660 | 0.7660 | 41.32 | 37.32 | 710 | 110 | 6.46× |
| sift | 2 | 32 | 0.7660 | 0.7660 | 41.32 | 37.32 | 2,335 | 938 | 2.49× |
| sift | 4 | 1 | 0.9350 | 0.9350 | 73.34 | 69.34 | 392 | 62 | 6.30× |
| sift | 4 | 32 | 0.9350 | 0.9350 | 73.34 | 69.34 | 1,566 | 531 | 2.95× |
| glove | 1 | 1 | 0.4010 | 0.4040 | 42.88 | 33.71 | 9,586 | 1,574 | 6.09× |
| glove | 1 | 32 | 0.4010 | 0.4040 | 42.88 | 33.71 | 24,087 | 7,359 | 3.27× |
| glove | 2 | 1 | 0.6930 | 0.6940 | 67.94 | 58.74 | 6,801 | 925 | 7.35× |
| glove | 2 | 32 | 0.6930 | 0.6940 | 67.94 | 58.74 | 16,369 | 4,931 | 3.32× |
| glove | 4 | 1 | 0.9140 | 0.9120 | 117.04 | 108.77 | 950 | 538 | 1.76× |
| glove | 4 | 32 | 0.9140 | 0.9120 | 117.04 | 108.77 | 10,696 | 3,430 | 3.12× |
| openai | 1 | 1 | 0.8840 | 0.8850 | 318.98 | 265.20 | 1,096 | 222 | 4.94× |
| openai | 1 | 32 | 0.8840 | 0.8850 | 318.98 | 265.20 | 3,955 | 1,376 | 2.87× |
| openai | 2 | 1 | 0.9470 | 0.9470 | 493.91 | 489.83 | 388 | 117 | 3.31× |
| openai | 2 | 32 | 0.9470 | 0.9470 | 493.91 | 489.83 | 1,185 | 898 | 1.32× |
| openai | 4 | 1 | 0.9840 | 0.9840 | 887.80 | 883.71 | 316 | 66 | 4.76× |
| openai | 4 | 32 | 0.9840 | 0.9840 | 887.80 | 883.71 | 917 | 480 | 1.91× |
| gemini | 1 | 1 | 0.8490 | 0.8450 | 154.76 | 121.73 | 3,402 | 449 | 7.57× |
| gemini | 1 | 32 | 0.8490 | 0.8450 | 154.76 | 121.73 | 8,818 | 1,632 | 5.40× |
| gemini | 2 | 1 | 0.9330 | 0.9330 | 228.47 | 224.43 | 935 | 235 | 3.98× |
| gemini | 2 | 32 | 0.9330 | 0.9330 | 228.47 | 224.43 | 2,735 | 1,415 | 1.93× |
| gemini | 4 | 1 | 0.9820 | 0.9820 | 425.25 | 421.20 | 546 | 132 | 4.13× |
| gemini | 4 | 32 | 0.9820 | 0.9820 | 425.25 | 421.20 | 2,160 | 986 | 2.19× |

See results.csv for p50/p95 latency and build/encode metrics. Per-cell diagnostics, input hashes and top-10 IDs are retained; normal versus fast rankings are reported without requiring equality because their scoring representations can differ.
