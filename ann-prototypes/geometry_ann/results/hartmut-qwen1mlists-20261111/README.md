# Fewer lists at 1M: 1K/2K/4K on Qwen 1M (#43, #52)

Remote root: `/home/sean/experiments/TieredQ-issue43-qwen1mlists-20261111`. `defaults_qwen.cpp` (from
`hartmut-defaults-20261111`) against TieredQ b0148dd. First 1M rows of Qwen/MS MARCO × 1024, truth is exact L2 over
them, 750 held-out queries alone on pinned CPU 2. Shipping prefix (1 bit/dim, 4-bit cap), 8-bit fine tier in RAM,
40 candidates, exact recipe for lists below 8K (20 Lloyd iterations, 64 rows per list), exact ranking. Partition
seeds 42 and 7. Raw rows are in `results/seed*/run.csv`.

`hartmut-defaults-20261111` picked 4K lists at 1M as the best of 4K-32K, but 4K was the smallest count it tried.
arxiv preferred 2K (`../hartmut-arxivfollowup-20261111`).

Recall@10 @ p50 (us), seed 42 / seed 7:

| nprobe (lists/512 .. /16) | 1K | 2K | 4K |
|---|---|---|---|
| 1/512 | 0.6532@517 / 0.6673@545 | 0.7443@617 / 0.7520@632 | 0.8231@803 / 0.8112@821 |
| 1/256 | 0.7692@595 / 0.7777@590 | 0.8336@673 / 0.8345@687 | 0.8773@882 / 0.8724@896 |
| 1/128 | 0.8511@636 / 0.8573@678 | 0.8853@762 / 0.8920@749 | 0.9168@978 / 0.9124@1006 |
| 1/64 | 0.9068@771 / 0.9116@782 | 0.9267@920 / 0.9328@935 | 0.9477@1162 / 0.9464@1190 |
| 1/32 | 0.9428@983 / 0.9432@1030 | 0.9584@1163 / 0.9572@1206 | 0.9671@1541 / 0.9648@1565 |
| 1/16 | 0.9661@1486 / 0.9680@1529 | 0.9745@1713 / 0.9737@1809 | 0.9807@2272 / 0.9781@2332 |

At equal recall (interpolated, seed 42): 0.958 takes about 1.16 ms at 2K, 1.31 ms at 1K and 1.37 ms at 4K. 0.9745
takes 1.71 ms at 2K and about 1.94 ms at 4K. Below about 0.95, 1K is fastest. Both seeds agree.

**Decision:** `automatic_lists` targets 560 rows per list at 1M (was 256), still falling log-linearly to 135 at 8.84M.
It now gives 2K lists at 1M and at arxiv's 1.34M, keeps 64K at 8.84M, and uses codec lists below about 143K rows.
