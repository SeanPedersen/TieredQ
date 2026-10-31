Rerank tier: 6,144 B/vector FP32, reported separately (SSD or RAM); index RAM = serialized compressed IVF.

### 256 lists: least index RAM per recall@10 target (then fastest 1-thread)

| Target | Codec | Config (budget, nprobe, k') | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |
|---|---|---|---|---|---|---|---|
| 0.90 | TieredQ | 1558, nprobe 8, k' 20 | 1,681 | 0.901 | 1,117 | 7,539 | 20 |
| 0.90 | RaBitQ | 1, nprobe 8, k' 50 | 1,683 | 0.904 | 1,131 | 7,607 | 50 |
| 0.95 | TieredQ | 1558, nprobe 32, k' 20 | 1,681 | 0.970 | 370 | 2,511 | 20 |
| 0.95 | RaBitQ | 1, nprobe 32, k' 20 | 1,683 | 0.961 | 316 | 2,217 | 20 |
| 0.99 | TieredQ | 1558, nprobe 128, k' 20 | 1,681 | 0.994 | 106 | 696 | 20 |
| 0.99 | RaBitQ | 1, nprobe 128, k' 50 | 1,683 | 0.997 | 86 | 629 | 50 |

### 256 lists: fastest configuration per memory level and target (index RAM vs QPS)

| Target | Level | Codec | Config | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 0 | TieredQ | 1558, nprobe 8, k' 20 | 1,681 | 0.901 | 1,117 | 7,539 | 20 |
| 0.90 | 0 | RaBitQ | 1, nprobe 8, k' 50 | 1,683 | 0.904 | 1,131 | 7,607 | 50 |
| 0.90 | 1 | TieredQ | 3133 max4, nprobe 8, k' 20 | 3,276 | 0.903 | 658 | 4,080 | 20 |
| 0.90 | 1 | RaBitQ | 2, nprobe 8, k' 20 | 3,283 | 0.905 | 1,127 | 7,324 | 20 |
| 0.90 | 2 | TieredQ | 6173 z8-p3, nprobe 8, k' 50 | 6,336 | 0.902 | 329 | 2,420 | 50 |
| 0.90 | 2 | RaBitQ | 4, nprobe 8, k' 20 | 6,355 | 0.905 | 1,076 | 7,155 | 20 |
| 0.95 | 0 | TieredQ | 1558, nprobe 32, k' 20 | 1,681 | 0.970 | 370 | 2,511 | 20 |
| 0.95 | 0 | RaBitQ | 1, nprobe 32, k' 20 | 1,683 | 0.961 | 316 | 2,217 | 20 |
| 0.95 | 1 | TieredQ | 3133 z8-p3, nprobe 32, k' 20 | 3,275 | 0.971 | 193 | 1,463 | 20 |
| 0.95 | 1 | RaBitQ | 2, nprobe 32, k' 20 | 3,283 | 0.973 | 327 | 2,239 | 20 |
| 0.95 | 2 | TieredQ | 6173 z8-p3, nprobe 32, k' 20 | 6,336 | 0.969 | 126 | 910 | 20 |
| 0.95 | 2 | RaBitQ | 4, nprobe 32, k' 10 | 6,355 | 0.952 | 329 | 2,268 | 10 |
| 0.99 | 0 | TieredQ | 1558, nprobe 128, k' 20 | 1,681 | 0.994 | 106 | 696 | 20 |
| 0.99 | 0 | RaBitQ | 1, nprobe 128, k' 50 | 1,683 | 0.997 | 86 | 629 | 50 |
| 0.99 | 1 | TieredQ | 3133 z8-p3, nprobe 128, k' 20 | 3,275 | 0.996 | 56 | 417 | 20 |
| 0.99 | 1 | RaBitQ | 2, nprobe 128, k' 20 | 3,283 | 0.997 | 87 | 622 | 20 |
| 0.99 | 2 | TieredQ | 6173 z8-p3, nprobe 128, k' 20 | 6,336 | 0.994 | 36 | 266 | 20 |
| 0.99 | 2 | RaBitQ | 4, nprobe 128, k' 20 | 6,355 | 0.998 | 87 | 627 | 20 |

### 1024 lists: least index RAM per recall@10 target (then fastest 1-thread)

| Target | Codec | Config (budget, nprobe, k') | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |
|---|---|---|---|---|---|---|---|
| 0.90 | TieredQ | 1558, nprobe 32, k' 20 | 1,738 | 0.926 | 909 | 6,583 | 20 |
| 0.90 | RaBitQ | 1, nprobe 32, k' 20 | 1,740 | 0.917 | 1,068 | 6,962 | 20 |
| 0.95 | TieredQ | 1558, nprobe 64, k' 20 | 1,738 | 0.955 | 574 | 4,054 | 20 |
| 0.95 | RaBitQ | 1, nprobe 64, k' 50 | 1,740 | 0.958 | 612 | 4,201 | 50 |
| 0.99 | TieredQ | — | — | — | — | — | — |
| 0.99 | RaBitQ | — | — | — | — | — | — |

### 1024 lists: fastest configuration per memory level and target (index RAM vs QPS)

| Target | Level | Codec | Config | Index bits/vec | Recall | QPS 1t | QPS 8t×32 | Reads/q |
|---|---|---|---|---|---|---|---|---|
| 0.90 | 0 | TieredQ | 1558, nprobe 32, k' 20 | 1,738 | 0.926 | 909 | 6,583 | 20 |
| 0.90 | 0 | RaBitQ | 1, nprobe 32, k' 20 | 1,740 | 0.917 | 1,068 | 6,962 | 20 |
| 0.90 | 1 | TieredQ | 3102 z8-p3, nprobe 32, k' 20 | 3,318 | 0.928 | 547 | 4,205 | 20 |
| 0.90 | 1 | RaBitQ | 2, nprobe 32, k' 20 | 3,340 | 0.926 | 1,059 | 6,978 | 20 |
| 0.90 | 2 | TieredQ | 6112 z8-p3, nprobe 32, k' 10 | 6,382 | 0.905 | 380 | 3,078 | 10 |
| 0.90 | 2 | RaBitQ | 4, nprobe 32, k' 10 | 6,412 | 0.912 | 1,044 | 6,847 | 10 |
| 0.95 | 0 | TieredQ | 1558, nprobe 64, k' 20 | 1,738 | 0.955 | 574 | 4,054 | 20 |
| 0.95 | 0 | RaBitQ | 1, nprobe 64, k' 50 | 1,740 | 0.958 | 612 | 4,201 | 50 |
| 0.95 | 1 | TieredQ | 3102 z8-p3, nprobe 64, k' 20 | 3,318 | 0.957 | 353 | 2,639 | 20 |
| 0.95 | 1 | RaBitQ | 2, nprobe 64, k' 20 | 3,340 | 0.958 | 607 | 4,183 | 20 |
| 0.95 | 2 | TieredQ | 6112 z8-p3, nprobe 64, k' 20 | 6,382 | 0.955 | 229 | 1,748 | 20 |
| 0.95 | 2 | RaBitQ | 4, nprobe 64, k' 20 | 6,412 | 0.958 | 613 | 4,108 | 20 |
| 0.99 | 0 | TieredQ | — | — | — | — | — | — |
| 0.99 | 0 | RaBitQ | — | — | — | — | — | — |
| 0.99 | 1 | TieredQ | — | — | — | — | — | — |
| 0.99 | 1 | RaBitQ | — | — | — | — | — | — |
| 0.99 | 2 | TieredQ | — | — | — | — | — | — |
| 0.99 | 2 | RaBitQ | — | — | — | — | — | — |

