# MS MARCO-1M real queries (dev questions), 256 lists: fastest timed cell per first stage and tier

Total bits = index bits + 8 x tier bytes per vector (RaBitQ native: its whole B-bit code is the index).
Hartmut, CPUs 0-7, alone; QPS includes rescoring. Raw rows: `real-time/`.

#### R@10 ≥ 0.9

| First stage | Index bits/vec | Tier | Tier B/vec | Total bits | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|---|
| RaBitQ 1 | 1,165 | int8 | 1,024 | 9,357 | 32, 50 | 0.928 | 530 / 3,799 |
| RaBitQ 1 | 1,165 | rabitq:5 | 660 | 6,445 | 32, 50 | 0.919 | 469 / 3,459 |
| RaBitQ 1 | 1,165 | rabitq:6 | 788 | 7,469 | 32, 50 | 0.926 | 467 / 3,380 |
| RaBitQ native 5-bit | 5,325 | — (in index) | 0 | 5,325 | 32, — | 0.921 | 519 / 3,638 |
| RaBitQ native 6-bit | 6,349 | — (in index) | 0 | 6,349 | 32, — | 0.928 | 523 / 3,775 |
| RaBitQ native 7-bit | 7,373 | — (in index) | 0 | 7,373 | 32, — | 0.933 | 518 / 3,689 |
| TieredQ 1024 | 1,133 | int8 | 1,024 | 9,325 | 32, 20 | 0.921 | 619 / 4,331 |
| TieredQ 1024 | 1,133 | tieredq-refinement:3 | 265 | 3,253 | 32, 20 | 0.909 | 591 / 4,265 |
| TieredQ 1024 | 1,133 | tieredq-refinement:4 | 393 | 4,277 | 32, 20 | 0.916 | 578 / 4,218 |
| TieredQ 512 | 608 | int8 | 1,024 | 8,800 | 32, 50 | 0.904 | 1,082 / 7,563 |
| TieredQ 512 | 608 | tieredq-refinement:3 | 327 | 3,224 | 32, 100 | 0.901 | 902 / 6,828 |
| TieredQ 512 | 608 | tieredq-refinement:4 | 455 | 4,248 | 32, 100 | 0.911 | 873 / 6,725 |
| TieredQ 768 | 867 | int8 | 1,024 | 9,059 | 32, 50 | 0.926 | 785 / 5,588 |
| TieredQ 768 | 867 | tieredq-refinement:3 | 281 | 3,115 | 32, 50 | 0.908 | 704 / 5,257 |
| TieredQ 768 | 867 | tieredq-refinement:4 | 409 | 4,139 | 32, 50 | 0.918 | 685 / 5,184 |

#### R@10 ≥ 0.95

| First stage | Index bits/vec | Tier | Tier B/vec | Total bits | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|---|
| RaBitQ 1 | 1,165 | int8 | 1,024 | 9,357 | 64, 50 | 0.957 | 263 / 1,935 |
| RaBitQ 1 | 1,165 | rabitq:5 | 660 | 6,445 | 128, 50 | 0.966 | 128 / 956 |
| RaBitQ 1 | 1,165 | rabitq:6 | 788 | 7,469 | 64, 50 | 0.954 | 254 / 1,845 |
| RaBitQ native 5-bit | 5,325 | — (in index) | 0 | 5,325 | 128, — | 0.969 | 134 / 960 |
| RaBitQ native 6-bit | 6,349 | — (in index) | 0 | 6,349 | 64, — | 0.958 | 266 / 1,932 |
| RaBitQ native 7-bit | 7,373 | — (in index) | 0 | 7,373 | 64, — | 0.963 | 259 / 1,910 |
| TieredQ 1024 | 1,133 | int8 | 1,024 | 9,325 | 64, 20 | 0.952 | 324 / 2,293 |
| TieredQ 1024 | 1,133 | tieredq-refinement:3 | 265 | 3,253 | 128, 20 | 0.955 | 161 / 1,129 |
| TieredQ 1024 | 1,133 | tieredq-refinement:4 | 393 | 4,277 | 64, 50 | 0.955 | 315 / 2,256 |
| TieredQ 512 | 608 | int8 | 1,024 | 8,800 | 64, 200 | 0.957 | 526 / 3,919 |
| TieredQ 512 | 608 | tieredq-refinement:3 | 327 | 3,224 | 128, 200 | 0.956 | 272 / 2,029 |
| TieredQ 512 | 608 | tieredq-refinement:4 | 455 | 4,248 | 64, 500 | 0.952 | 439 / 3,262 |
| TieredQ 768 | 867 | int8 | 1,024 | 9,059 | 64, 50 | 0.958 | 395 / 2,867 |
| TieredQ 768 | 867 | tieredq-refinement:3 | 281 | 3,115 | 128, 50 | 0.955 | 196 / 1,450 |
| TieredQ 768 | 867 | tieredq-refinement:4 | 409 | 4,139 | 64, 100 | 0.952 | 361 / 2,768 |

#### R@10 ≥ 0.97

| First stage | Index bits/vec | Tier | Tier B/vec | Total bits | nprobe, k′ | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|---|
| RaBitQ 1 | 1,165 | int8 | 1,024 | 9,357 | 128, 50 | 0.978 | 132 / 995 |
| RaBitQ 1 | 1,165 | rabitq:6 | 788 | 7,469 | 128, 50 | 0.974 | 131 / 956 |
| RaBitQ native 6-bit | 6,349 | — (in index) | 0 | 6,349 | 128, — | 0.978 | 134 / 984 |
| RaBitQ native 7-bit | 7,373 | — (in index) | 0 | 7,373 | 128, — | 0.984 | 132 / 974 |
| TieredQ 1024 | 1,133 | int8 | 1,024 | 9,325 | 128, 20 | 0.973 | 164 / 1,138 |
| TieredQ 1024 | 1,133 | tieredq-refinement:4 | 393 | 4,277 | 128, 50 | 0.975 | 160 / 1,132 |
| TieredQ 512 | 608 | int8 | 1,024 | 8,800 | 128, 200 | 0.977 | 284 / 2,099 |
| TieredQ 512 | 608 | tieredq-refinement:4 | 455 | 4,248 | 128, 500 | 0.972 | 243 / 1,825 |
| TieredQ 768 | 867 | int8 | 1,024 | 9,059 | 128, 50 | 0.979 | 201 / 1,440 |
| TieredQ 768 | 867 | tieredq-refinement:4 | 409 | 4,139 | 128, 100 | 0.971 | 189 / 1,405 |
