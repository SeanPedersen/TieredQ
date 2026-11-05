# Issue 26: unmodified main baseline at 8 bits/dimension

Main 5509f27, clean export, Python 3.12, seed 0, exact float32 ground truth, real TieredQ codec. OMP/OpenBLAS threads=8; affinity=8-15 on hartmut. Search times are preliminary: concurrent compilation and a pre-existing simulation were active. Recall is measured from codec output; no simulated recall is included. The auxiliary full cache matches full recall here; this does not isolate byte rounding from the different fast-target codebooks.


## glove  (n=100000, D=200, metric=ip, 8 bits/dim, 1000 queries)
| scorer | recall@10 | recall@100 | missed@10 | search s |
|---|---|---|---|---|
| full | 0.99180 | 0.99418 | 82 | 1.8 |
| full+cache | 0.99180 | 0.99418 | 82 | 0.1 |
| fast | 0.98410 | 0.98834 | 159 | 0.2 |

## gemini  (n=99000, D=768, metric=ip, 8 bits/dim, 1000 queries)
| scorer | recall@10 | recall@100 | missed@10 | search s |
|---|---|---|---|---|
| full | 0.99750 | 0.99757 | 25 | 6.2 |
| full+cache | 0.99750 | 0.99757 | 25 | 0.8 |
| fast | 0.99320 | 0.99359 | 68 | 0.7 |

## openai100k  (n=100000, D=1536, metric=ip, 8 bits/dim, 1000 queries)
| scorer | recall@10 | recall@100 | missed@10 | search s |
|---|---|---|---|---|
| full | 0.99790 | 0.99825 | 21 | 12.5 |
| full+cache | 0.99790 | 0.99825 | 21 | 0.8 |
| fast | 0.99620 | 0.99611 | 38 | 1.7 |
