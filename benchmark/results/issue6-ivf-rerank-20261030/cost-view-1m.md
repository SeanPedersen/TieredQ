# Per-resource view at 1M vectors: RAM (in-memory index), disk (rerank tier read by id), QPS

Per configuration family, the fastest timed cell (1t QPS) reaching each R@10 target. RAM is the serialized IVF
(codes, lists, ids) per 1M vectors; disk is the rerank tier per 1M vectors, read as k′ random rows per query. QPS was
measured with the tier memory-resident (page cache): on SSD, a configuration needs QPS x k′ reads/s. RaBitQ native keeps
its whole B-bit code in RAM and reads nothing. Sources: `openai1m-rescore-time/` (INT8, RaBitQ tiers),
`openai1m-rescore-time2/` (refinement), `native-time/`, `../issue22-msmarco1m-20261031/real-time/`.

### OpenAI-1M (DBpedia, 1,536-d, held-out document queries), per 1M vectors


**R@10 ≥ 0.95**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 210 MB (1,683 bits/vec) | 1,536 MB | 20 | 32 | 0.960 | 332 / 2,309 |
| RaBitQ 1-bit + own 1→5 codes | 210 MB (1,683 bits/vec) | 980 MB | 20 | 32 | 0.952 | 298 / 2,088 |
| RaBitQ 1-bit + own 1→6 codes | 210 MB (1,683 bits/vec) | 1,172 MB | 20 | 32 | 0.956 | 298 / 2,063 |
| RaBitQ native 5-bit | 986 MB (7,891 bits/vec) | — | — | 32 | 0.963 | 331 / 2,283 |
| RaBitQ native 6-bit | 1,178 MB (9,427 bits/vec) | — | — | 32 | 0.967 | 337 / 2,312 |
| RaBitQ native 7-bit | 1,370 MB (10,963 bits/vec) | — | — | 32 | 0.971 | 328 / 2,307 |
| TieredQ 512 + INT8 | 78 MB (627 bits/vec) | 1,536 MB | 50 | 32 | 0.955 | 914 / 6,343 |
| TieredQ 512 + refinement:3 | 78 MB (627 bits/vec) | 517 MB | 50 | 64 | 0.951 | 471 / 3,490 |
| TieredQ 512 + refinement:4 | 78 MB (627 bits/vec) | 702 MB | 100 | 32 | 0.954 | 719 / 5,525 |
| TieredQ 768 + INT8 | 111 MB (886 bits/vec) | 1,536 MB | 50 | 32 | 0.967 | 663 / 4,682 |
| TieredQ 768 + refinement:3 | 111 MB (886 bits/vec) | 475 MB | 50 | 64 | 0.961 | 329 / 2,523 |
| TieredQ 768 + refinement:4 | 111 MB (886 bits/vec) | 672 MB | 50 | 32 | 0.958 | 589 / 4,338 |
| TieredQ 1024 + INT8 | 142 MB (1,138 bits/vec) | 1,536 MB | 20 | 32 | 0.954 | 525 / 3,500 |
| TieredQ 1024 + refinement:3 | 142 MB (1,138 bits/vec) | 462 MB | 50 | 32 | 0.953 | 460 / 3,239 |
| TieredQ 1024 + refinement:4 | 142 MB (1,138 bits/vec) | 647 MB | 50 | 32 | 0.960 | 453 / 3,295 |

**R@10 ≥ 0.97**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 210 MB (1,683 bits/vec) | 1,536 MB | 50 | 32 | 0.971 | 323 / 2,261 |
| RaBitQ 1-bit + own 1→5 codes | 210 MB (1,683 bits/vec) | 980 MB | 50 | 64 | 0.977 | 162 / 1,123 |
| RaBitQ 1-bit + own 1→6 codes | 210 MB (1,683 bits/vec) | 1,172 MB | 20 | 64 | 0.971 | 160 / 1,137 |
| RaBitQ native 5-bit | 986 MB (7,891 bits/vec) | — | — | 64 | 0.977 | 171 / 1,196 |
| RaBitQ native 6-bit | 1,178 MB (9,427 bits/vec) | — | — | 64 | 0.982 | 174 / 1,216 |
| RaBitQ native 7-bit | 1,370 MB (10,963 bits/vec) | — | — | 32 | 0.971 | 328 / 2,307 |
| TieredQ 512 + INT8 | 78 MB (627 bits/vec) | 1,536 MB | 500 | 32 | 0.970 | 685 / 4,793 |
| TieredQ 512 + refinement:4 | 78 MB (627 bits/vec) | 702 MB | 200 | 64 | 0.970 | 405 / 3,199 |
| TieredQ 768 + INT8 | 111 MB (886 bits/vec) | 1,536 MB | 200 | 32 | 0.970 | 616 / 4,275 |
| TieredQ 768 + refinement:4 | 111 MB (886 bits/vec) | 672 MB | 50 | 64 | 0.973 | 339 / 2,492 |
| TieredQ 1024 + INT8 | 142 MB (1,138 bits/vec) | 1,536 MB | 100 | 32 | 0.970 | 502 / 3,477 |
| TieredQ 1024 + refinement:3 | 142 MB (1,138 bits/vec) | 462 MB | 50 | 128 | 0.974 | 136 / 968 |
| TieredQ 1024 + refinement:4 | 142 MB (1,138 bits/vec) | 647 MB | 50 | 64 | 0.975 | 251 / 1,802 |

**R@10 ≥ 0.98**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 210 MB (1,683 bits/vec) | 1,536 MB | 50 | 64 | 0.986 | 168 / 1,179 |
| RaBitQ 1-bit + own 1→5 codes | 210 MB (1,683 bits/vec) | 980 MB | 50 | 128 | 0.984 | 83 / 605 |
| RaBitQ 1-bit + own 1→6 codes | 210 MB (1,683 bits/vec) | 1,172 MB | 50 | 64 | 0.982 | 157 / 1,137 |
| RaBitQ native 5-bit | 986 MB (7,891 bits/vec) | — | — | 128 | 0.984 | 89 / 623 |
| RaBitQ native 6-bit | 1,178 MB (9,427 bits/vec) | — | — | 64 | 0.982 | 174 / 1,216 |
| RaBitQ native 7-bit | 1,370 MB (10,963 bits/vec) | — | — | 64 | 0.986 | 168 / 1,218 |
| TieredQ 512 + INT8 | 78 MB (627 bits/vec) | 1,536 MB | 100 | 64 | 0.982 | 520 / 3,690 |
| TieredQ 768 + INT8 | 111 MB (886 bits/vec) | 1,536 MB | 50 | 64 | 0.983 | 359 / 2,606 |
| TieredQ 768 + refinement:4 | 111 MB (886 bits/vec) | 672 MB | 50 | 128 | 0.980 | 187 / 1,371 |
| TieredQ 1024 + INT8 | 142 MB (1,138 bits/vec) | 1,536 MB | 50 | 64 | 0.986 | 273 / 1,828 |
| TieredQ 1024 + refinement:4 | 142 MB (1,138 bits/vec) | 647 MB | 50 | 128 | 0.983 | 138 / 970 |

**R@10 ≥ 0.99**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 210 MB (1,683 bits/vec) | 1,536 MB | 50 | 128 | 0.994 | 87 / 611 |
| RaBitQ native 7-bit | 1,370 MB (10,963 bits/vec) | — | — | 128 | 0.994 | 85 / 621 |
| TieredQ 512 + INT8 | 78 MB (627 bits/vec) | 1,536 MB | 100 | 128 | 0.990 | 271 / 1,958 |
| TieredQ 768 + INT8 | 111 MB (886 bits/vec) | 1,536 MB | 50 | 128 | 0.991 | 192 / 1,377 |
| TieredQ 1024 + INT8 | 142 MB (1,138 bits/vec) | 1,536 MB | 50 | 128 | 0.994 | 142 / 978 |

### MS MARCO-1M (Cohere v3, 1,024-d, real dev questions), per 1M vectors


**R@10 ≥ 0.9**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 146 MB (1,165 bits/vec) | 1,024 MB | 50 | 32 | 0.928 | 530 / 3,799 |
| RaBitQ 1-bit + own 1→5 codes | 146 MB (1,165 bits/vec) | 660 MB | 50 | 32 | 0.919 | 469 / 3,459 |
| RaBitQ 1-bit + own 1→6 codes | 146 MB (1,165 bits/vec) | 788 MB | 50 | 32 | 0.926 | 467 / 3,380 |
| RaBitQ native 5-bit | 666 MB (5,325 bits/vec) | — | — | 32 | 0.921 | 519 / 3,638 |
| RaBitQ native 6-bit | 794 MB (6,349 bits/vec) | — | — | 32 | 0.928 | 523 / 3,775 |
| RaBitQ native 7-bit | 922 MB (7,373 bits/vec) | — | — | 32 | 0.933 | 518 / 3,689 |
| TieredQ 512 + INT8 | 76 MB (608 bits/vec) | 1,024 MB | 50 | 32 | 0.904 | 1,082 / 7,563 |
| TieredQ 512 + refinement:3 | 76 MB (608 bits/vec) | 327 MB | 100 | 32 | 0.901 | 902 / 6,828 |
| TieredQ 512 + refinement:4 | 76 MB (608 bits/vec) | 455 MB | 100 | 32 | 0.911 | 873 / 6,725 |
| TieredQ 768 + INT8 | 108 MB (867 bits/vec) | 1,024 MB | 50 | 32 | 0.926 | 785 / 5,588 |
| TieredQ 768 + refinement:3 | 108 MB (867 bits/vec) | 281 MB | 50 | 32 | 0.908 | 704 / 5,257 |
| TieredQ 768 + refinement:4 | 108 MB (867 bits/vec) | 409 MB | 50 | 32 | 0.918 | 685 / 5,184 |
| TieredQ 1024 + INT8 | 142 MB (1,133 bits/vec) | 1,024 MB | 20 | 32 | 0.921 | 619 / 4,331 |
| TieredQ 1024 + refinement:3 | 142 MB (1,133 bits/vec) | 265 MB | 20 | 32 | 0.909 | 591 / 4,265 |
| TieredQ 1024 + refinement:4 | 142 MB (1,133 bits/vec) | 393 MB | 20 | 32 | 0.916 | 578 / 4,218 |

**R@10 ≥ 0.95**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 146 MB (1,165 bits/vec) | 1,024 MB | 50 | 64 | 0.957 | 263 / 1,935 |
| RaBitQ 1-bit + own 1→5 codes | 146 MB (1,165 bits/vec) | 660 MB | 50 | 128 | 0.966 | 128 / 956 |
| RaBitQ 1-bit + own 1→6 codes | 146 MB (1,165 bits/vec) | 788 MB | 50 | 64 | 0.954 | 254 / 1,845 |
| RaBitQ native 5-bit | 666 MB (5,325 bits/vec) | — | — | 128 | 0.969 | 134 / 960 |
| RaBitQ native 6-bit | 794 MB (6,349 bits/vec) | — | — | 64 | 0.958 | 266 / 1,932 |
| RaBitQ native 7-bit | 922 MB (7,373 bits/vec) | — | — | 64 | 0.963 | 259 / 1,910 |
| TieredQ 512 + INT8 | 76 MB (608 bits/vec) | 1,024 MB | 200 | 64 | 0.957 | 526 / 3,919 |
| TieredQ 512 + refinement:3 | 76 MB (608 bits/vec) | 327 MB | 200 | 128 | 0.956 | 272 / 2,029 |
| TieredQ 512 + refinement:4 | 76 MB (608 bits/vec) | 455 MB | 500 | 64 | 0.952 | 439 / 3,262 |
| TieredQ 768 + INT8 | 108 MB (867 bits/vec) | 1,024 MB | 50 | 64 | 0.958 | 395 / 2,867 |
| TieredQ 768 + refinement:3 | 108 MB (867 bits/vec) | 281 MB | 50 | 128 | 0.955 | 196 / 1,450 |
| TieredQ 768 + refinement:4 | 108 MB (867 bits/vec) | 409 MB | 100 | 64 | 0.952 | 361 / 2,768 |
| TieredQ 1024 + INT8 | 142 MB (1,133 bits/vec) | 1,024 MB | 20 | 64 | 0.952 | 324 / 2,293 |
| TieredQ 1024 + refinement:3 | 142 MB (1,133 bits/vec) | 265 MB | 20 | 128 | 0.955 | 161 / 1,129 |
| TieredQ 1024 + refinement:4 | 142 MB (1,133 bits/vec) | 393 MB | 50 | 64 | 0.955 | 315 / 2,256 |

**R@10 ≥ 0.97**

| Configuration | RAM (index) | Disk (rerank tier) | Disk reads / query | nprobe | Recall | QPS 1t / 8t |
|---|---|---|---|---|---|---|
| RaBitQ 1-bit + INT8 | 146 MB (1,165 bits/vec) | 1,024 MB | 50 | 128 | 0.978 | 132 / 995 |
| RaBitQ 1-bit + own 1→6 codes | 146 MB (1,165 bits/vec) | 788 MB | 50 | 128 | 0.974 | 131 / 956 |
| RaBitQ native 6-bit | 794 MB (6,349 bits/vec) | — | — | 128 | 0.978 | 134 / 984 |
| RaBitQ native 7-bit | 922 MB (7,373 bits/vec) | — | — | 128 | 0.984 | 132 / 974 |
| TieredQ 512 + INT8 | 76 MB (608 bits/vec) | 1,024 MB | 200 | 128 | 0.977 | 284 / 2,099 |
| TieredQ 512 + refinement:4 | 76 MB (608 bits/vec) | 455 MB | 500 | 128 | 0.972 | 243 / 1,825 |
| TieredQ 768 + INT8 | 108 MB (867 bits/vec) | 1,024 MB | 50 | 128 | 0.979 | 201 / 1,440 |
| TieredQ 768 + refinement:4 | 108 MB (867 bits/vec) | 409 MB | 100 | 128 | 0.971 | 189 / 1,405 |
| TieredQ 1024 + INT8 | 142 MB (1,133 bits/vec) | 1,024 MB | 20 | 128 | 0.973 | 164 / 1,138 |
| TieredQ 1024 + refinement:4 | 142 MB (1,133 bits/vec) | 393 MB | 50 | 128 | 0.975 | 160 / 1,132 |
