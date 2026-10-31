# OpenAI-1M, 256 coarse lists: sub-1-bit TieredQ prefixes + INT8 / refinement rerank vs RaBitQ 1 bit + INT8 / its own codes

Fastest timed cell (1t QPS) reaching each target per first stage and rerank tier; Hartmut, CPUs 0-7, alone. INT8 and
RaBitQ rows: `openai1m-rescore-time/` (its refinement rows ran on the default 1416 prefix and are void); refinement
rows: `openai1m-rescore-time2/` (scorer of this commit, the IVF's query weights shared with the rescorer). Untimed
recall per tier: `openai1m-rescore-recall-tables.md`.

#### R@10 ≥ 0.95

| First stage | Index bits/vec | Tier | Tier B/vec | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| TieredQ 1024 | 1,138 | int8 | 1,536 | 32, 20 | 0.954 | 525 / 3,500 |
| TieredQ 1024 | 1,138 | tieredq-refinement:3 | 462 | 32, 50 | 0.953 | 460 / 3,239 |
| TieredQ 1024 | 1,138 | tieredq-refinement:4 | 647 | 32, 50 | 0.960 | 453 / 3,295 |
| TieredQ 512 | 627 | int8 | 1,536 | 32, 50 | 0.955 | 914 / 6,343 |
| TieredQ 512 | 627 | tieredq-refinement:3 | 517 | 64, 50 | 0.951 | 471 / 3,490 |
| TieredQ 512 | 627 | tieredq-refinement:4 | 702 | 32, 100 | 0.954 | 719 / 5,525 |
| TieredQ 768 | 886 | int8 | 1,536 | 32, 50 | 0.967 | 663 / 4,682 |
| TieredQ 768 | 886 | tieredq-refinement:3 | 475 | 64, 50 | 0.961 | 329 / 2,523 |
| TieredQ 768 | 886 | tieredq-refinement:4 | 672 | 32, 50 | 0.958 | 589 / 4,338 |
| RaBitQ 1 | 1,683 | int8 | 1,536 | 32, 20 | 0.960 | 332 / 2,309 |
| RaBitQ 1 | 1,683 | rabitq:5 | 980 | 32, 20 | 0.952 | 298 / 2,088 |
| RaBitQ 1 | 1,683 | rabitq:6 | 1,172 | 32, 20 | 0.956 | 298 / 2,063 |

#### R@10 ≥ 0.97

| First stage | Index bits/vec | Tier | Tier B/vec | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| TieredQ 1024 | 1,138 | int8 | 1,536 | 32, 100 | 0.970 | 502 / 3,477 |
| TieredQ 1024 | 1,138 | tieredq-refinement:3 | 462 | 128, 50 | 0.974 | 136 / 968 |
| TieredQ 1024 | 1,138 | tieredq-refinement:4 | 647 | 64, 50 | 0.975 | 251 / 1,802 |
| TieredQ 512 | 627 | int8 | 1,536 | 32, 500 | 0.970 | 685 / 4,793 |
| TieredQ 512 | 627 | tieredq-refinement:4 | 702 | 64, 200 | 0.970 | 405 / 3,199 |
| TieredQ 768 | 886 | int8 | 1,536 | 32, 200 | 0.970 | 616 / 4,275 |
| TieredQ 768 | 886 | tieredq-refinement:4 | 672 | 64, 50 | 0.973 | 339 / 2,492 |
| RaBitQ 1 | 1,683 | int8 | 1,536 | 32, 50 | 0.971 | 323 / 2,261 |
| RaBitQ 1 | 1,683 | rabitq:5 | 980 | 64, 50 | 0.977 | 162 / 1,123 |
| RaBitQ 1 | 1,683 | rabitq:6 | 1,172 | 64, 20 | 0.971 | 160 / 1,137 |

#### R@10 ≥ 0.98

| First stage | Index bits/vec | Tier | Tier B/vec | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| TieredQ 1024 | 1,138 | int8 | 1,536 | 64, 50 | 0.986 | 273 / 1,828 |
| TieredQ 1024 | 1,138 | tieredq-refinement:4 | 647 | 128, 50 | 0.983 | 138 / 970 |
| TieredQ 512 | 627 | int8 | 1,536 | 64, 100 | 0.982 | 520 / 3,690 |
| TieredQ 768 | 886 | int8 | 1,536 | 64, 50 | 0.983 | 359 / 2,606 |
| TieredQ 768 | 886 | tieredq-refinement:4 | 672 | 128, 50 | 0.980 | 187 / 1,371 |
| RaBitQ 1 | 1,683 | int8 | 1,536 | 64, 50 | 0.986 | 168 / 1,179 |
| RaBitQ 1 | 1,683 | rabitq:5 | 980 | 128, 50 | 0.984 | 83 / 605 |
| RaBitQ 1 | 1,683 | rabitq:6 | 1,172 | 64, 50 | 0.982 | 157 / 1,137 |

#### R@10 ≥ 0.99

| First stage | Index bits/vec | Tier | Tier B/vec | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| TieredQ 1024 | 1,138 | int8 | 1,536 | 128, 50 | 0.994 | 142 / 978 |
| TieredQ 512 | 627 | int8 | 1,536 | 128, 100 | 0.990 | 271 / 1,958 |
| TieredQ 768 | 886 | int8 | 1,536 | 128, 50 | 0.991 | 192 / 1,377 |
| RaBitQ 1 | 1,683 | int8 | 1,536 | 128, 50 | 0.994 | 87 / 611 |
