# TieredQ-compressed centroid routing with a 4-bit scan cap

Status: complete (exit 0); see `REPORT.md`. Remote root: `/home/sean/experiments/TieredQ-issue44-compressed4-20261110`.

Setup as `hartmut-compressed-20261110`, but with `--routing-scan-bits 4`, which keeps every tier inside the nibble-lookup kernel. Models:
2 bits/dim with 1,024 and with 2,048 candidates, and 1 bit/dim with 1,024. Exact re-rank R ∈ {256, 512}.

Held-out highlights (2 bits/dim, 1,024 candidates, R=256):

| Setting | Recall | Routing ms | p50 / p99 ms |
|---|---:|---:|---:|
| Fixed cap 35K | 0.9580 | 2.75 | 7.6 / 8.7 |
| Patience W=48 | 0.9505 | 2.63 | 5.2 / 10.3 |
| Patience W=96 | 0.9665 | 2.74 | 7.4 / 15.9 |
| Patience W=768 | 0.9865 | 3.01 | 34.2 / 47.3 |
| FP32 flat, cap 35K (same build) | 0.9580 | 14.07 | 20.8 / 25.2 |
| Faiss IVF-SQ8 32K nprobe 96 / 1024 | 0.9557 / 0.9880 | – | 12.7 / 49.9 (p99 14.1 / 55.0) |

Findings:
- Routing is 5× faster than FP32 flat routing, and recall and vectors are unchanged. Near 0.987, p50 and p99 both beat Faiss 32K.
- The candidate limit doubles as a tail cap: patience W=768 reaches p99 328K vectors, against 540K for flat routing, at a cost of 0.0015 recall.
- 2,048 candidates add about 1.4 ms of routing and only help W=768. 1 bit/dim saves about 0.3 ms and loses up to 0.001 recall at deep caps with R=256.
- The remaining routing cost depends on depth: the pruned scan's threshold loosens as k grows, so more rows are fully scored.
  At 2 bits/dim locally, k=10 takes 1.6 ms, k=1,024 3.4 ms and k=4,096 7.3 ms. Sorting and heaps take only microseconds, so an unsorted top-k
  would not help. Lazy candidate expansion (`--routing-initial`) ranks only as deep as each query scans.
