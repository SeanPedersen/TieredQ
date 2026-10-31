Rerank tier: 6,144 B/vector FP32, reported separately (SSD or RAM); index RAM = serialized compressed IVF.

### 64 lists: least index RAM per recall@10 target (then fastest 1-thread)

| Target | Codec | Config (budget, nprobe, k') | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |
|---|---|---|---|---|---|---|---|
| 0.90 | TieredQ | 1416, nprobe 8, k' 20 | 1,706 | 0.939 | 3,857 | 30,450 | 20 |
| 0.90 | RaBitQ | 1, nprobe 8, k' 20 | 1,711 | 0.927 | 3,514 | 24,523 | 20 |
| 0.95 | TieredQ | 1416, nprobe 16, k' 20 | 1,706 | 0.971 | 2,349 | 18,583 | 20 |
| 0.95 | RaBitQ | 1, nprobe 16, k' 20 | 1,711 | 0.958 | 1,995 | 13,981 | 20 |
| 0.99 | TieredQ | 1416, nprobe 32, k' 50 | 1,706 | 0.993 | 1,362 | 9,870 | 50 |
| 0.99 | RaBitQ | 1, nprobe 32, k' 50 | 1,711 | 0.993 | 996 | 7,357 | 50 |

### 64 lists: fastest configuration per memory level and target (index RAM vs QPS)

| Target | Level | Codec | Config | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 0 | TieredQ | 1416, nprobe 8, k' 20 | 1,706 | 0.939 | 3,857 | 30,450 | 20 |
| 0.90 | 0 | RaBitQ | 1, nprobe 8, k' 20 | 1,711 | 0.927 | 3,514 | 24,523 | 20 |
| 0.90 | 1 | TieredQ | 2888 max4, nprobe 8, k' 10 | 3,318 | 0.903 | 1,839 | 15,070 | 10 |
| 0.90 | 1 | RaBitQ | 2, nprobe 8, k' 20 | 3,311 | 0.940 | 3,259 | 21,873 | 20 |
| 0.90 | 2 | TieredQ | 5820 z8-p3, nprobe 8, k' 10 | 6,359 | 0.932 | 919 | 6,624 | 10 |
| 0.90 | 2 | RaBitQ | 4, nprobe 8, k' 10 | 6,383 | 0.922 | 3,047 | 21,635 | 10 |
| 0.95 | 0 | TieredQ | 1416, nprobe 16, k' 20 | 1,706 | 0.971 | 2,349 | 18,583 | 20 |
| 0.95 | 0 | RaBitQ | 1, nprobe 16, k' 20 | 1,711 | 0.958 | 1,995 | 13,981 | 20 |
| 0.95 | 1 | TieredQ | 2888 max4, nprobe 16, k' 20 | 3,318 | 0.976 | 1,039 | 8,459 | 20 |
| 0.95 | 1 | RaBitQ | 2, nprobe 16, k' 20 | 3,311 | 0.975 | 1,832 | 12,531 | 20 |
| 0.95 | 2 | TieredQ | 5820 z8-p3, nprobe 16, k' 10 | 6,359 | 0.963 | 591 | 3,972 | 10 |
| 0.95 | 2 | RaBitQ | 4, nprobe 16, k' 10 | 6,383 | 0.952 | 1,812 | 12,760 | 10 |
| 0.99 | 0 | TieredQ | 1416, nprobe 32, k' 50 | 1,706 | 0.993 | 1,362 | 9,870 | 50 |
| 0.99 | 0 | RaBitQ | 1, nprobe 32, k' 50 | 1,711 | 0.993 | 996 | 7,357 | 50 |
| 0.99 | 1 | TieredQ | 2888 max4, nprobe 32, k' 20 | 3,318 | 0.993 | 566 | 4,572 | 20 |
| 0.99 | 1 | RaBitQ | 2, nprobe 32, k' 20 | 3,311 | 0.992 | 1,037 | 7,382 | 20 |
| 0.99 | 2 | TieredQ | 5820 z8-p3, nprobe 32, k' 20 | 6,359 | 0.993 | 325 | 2,416 | 20 |
| 0.99 | 2 | RaBitQ | 4, nprobe 32, k' 20 | 6,383 | 0.994 | 955 | 7,231 | 20 |

