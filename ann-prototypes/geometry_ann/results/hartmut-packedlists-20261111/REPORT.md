# Isolated cores (results/)

```
arxiv: lists 2048, storage 233.8 -> 233.1 MiB (-0.28%), file 238.0 -> 232.7 MiB, build 25 / 24 s
  arxiv-serve-1A: ids and scores identical to arxiv-serve-1A
  arxiv-serve-2B: ids and scores identical to arxiv-serve-1A
  arxiv-serve-3B: ids and scores identical to arxiv-serve-1A
  arxiv-serve-4A: ids and scores identical to arxiv-serve-1A
  k  10 nprobe    8: single 3023 -> 2986 (-1.2%, A spread 1.5%), batch 45276 -> 47026 (+3.9%, A spread 10.6%)
  k  10 nprobe   16: single 2257 -> 2254 (-0.1%, A spread 0.4%), batch 30999 -> 30128 (-2.8%, A spread 2.5%)
  k  10 nprobe   32: single 1519 -> 1485 (-2.3%, A spread 0.2%), batch 17393 -> 17775 (+2.2%, A spread 2.0%)
  k  10 nprobe   48: single 1137 -> 1122 (-1.3%, A spread 1.5%), batch 12078 -> 12390 (+2.6%, A spread 4.4%)
  k 100 nprobe    8: single 2721 -> 2723 (+0.1%, A spread 0.3%), batch 46558 -> 46026 (-1.1%, A spread 1.3%)
  k 100 nprobe   16: single 2065 -> 2051 (-0.7%, A spread 1.8%), batch 30637 -> 26970 (-12.0%, A spread 4.7%)
  k 100 nprobe   32: single 1413 -> 1415 (+0.2%, A spread 0.5%), batch 17273 -> 16214 (-6.1%, A spread 7.3%)
  k 100 nprobe   48: single 1088 -> 1061 (-2.4%, A spread 0.0%), batch 11872 -> 11988 (+1.0%, A spread 6.8%)
qwen: lists 65536, storage 1572.9 -> 1469.8 MiB (-6.55%), file 1553.8 -> 1421.2 MiB, build 228 / 209 s
  qwen-serve-1A: ids and scores identical to qwen-serve-1A
  qwen-serve-2B: ids and scores identical to qwen-serve-1A
  qwen-serve-3B: ids and scores identical to qwen-serve-1A
  qwen-serve-4A: ids and scores identical to qwen-serve-1A
  k  10 nprobe   64: single 962 -> 958 (-0.5%, A spread 4.5%), batch 12501 -> 12218 (-2.3%, A spread 1.0%)
  k  10 nprobe  128: single 655 -> 633 (-3.2%, A spread 2.5%), batch 7931 -> 7385 (-6.9%, A spread 0.9%)
  k  10 nprobe  256: single 397 -> 392 (-1.4%, A spread 0.6%), batch 4622 -> 4403 (-4.7%, A spread 0.2%)
  k  10 nprobe 1024: single 129 -> 128 (-0.8%, A spread 0.4%), batch 1355 -> 1302 (-3.9%, A spread 2.7%)
  k 100 nprobe   64: single 919 -> 914 (-0.6%, A spread 3.2%), batch 12327 -> 11966 (-2.9%, A spread 3.1%)
  k 100 nprobe  128: single 631 -> 612 (-2.9%, A spread 0.9%), batch 7402 -> 7586 (+2.5%, A spread 6.3%)
  k 100 nprobe  256: single 388 -> 383 (-1.3%, A spread 2.0%), batch 4419 -> 4389 (-0.7%, A spread 2.5%)
  k 100 nprobe 1024: single 128 -> 126 (-1.6%, A spread 1.5%), batch 1339 -> 1276 (-4.7%, A spread 1.0%)
```

# Loaded host (results-loaded-host/; the builds are in results/)

```
arxiv: lists 2048, storage 233.8 -> 233.1 MiB (-0.28%), file 238.0 -> 232.7 MiB, build 25 / 24 s
  arxiv-serve-1A: ids and scores identical to arxiv-serve-1A
  arxiv-serve-2B: ids and scores identical to arxiv-serve-1A
  arxiv-serve-3B: ids and scores identical to arxiv-serve-1A
  arxiv-serve-4A: ids and scores identical to arxiv-serve-1A
  k  10 nprobe    8: single 2966 -> 3001 (+1.2%, A spread 2.2%), batch 51530 -> 50592 (-1.8%, A spread 5.0%)
  k  10 nprobe   16: single 2251 -> 2240 (-0.5%, A spread 0.6%), batch 30839 -> 30529 (-1.0%, A spread 1.5%)
  k  10 nprobe   32: single 1510 -> 1487 (-1.5%, A spread 0.1%), batch 18233 -> 18059 (-1.0%, A spread 0.2%)
  k  10 nprobe   48: single 1120 -> 1112 (-0.8%, A spread 1.9%), batch 12037 -> 12209 (+1.4%, A spread 7.7%)
  k 100 nprobe    8: single 2679 -> 2702 (+0.9%, A spread 3.8%), batch 43942 -> 44293 (+0.8%, A spread 15.8%)
  k 100 nprobe   16: single 2063 -> 2042 (-1.0%, A spread 0.5%), batch 29809 -> 25607 (-14.1%, A spread 1.8%)
  k 100 nprobe   32: single 1410 -> 1402 (-0.5%, A spread 2.1%), batch 16452 -> 15914 (-3.3%, A spread 14.3%)
  k 100 nprobe   48: single 1080 -> 1064 (-1.6%, A spread 0.9%), batch 12381 -> 12248 (-1.1%, A spread 2.7%)
qwen: lists 65536, storage 1572.9 -> 1469.8 MiB (-6.55%), file 1553.8 -> 1421.2 MiB, build 228 / 209 s
  qwen-serve-1A: ids and scores identical to qwen-serve-1A
  qwen-serve-2B: ids and scores identical to qwen-serve-1A
  qwen-serve-3B: ids and scores identical to qwen-serve-1A
  qwen-serve-4A: ids and scores identical to qwen-serve-1A
  k  10 nprobe   64: single 959 -> 953 (-0.6%, A spread 1.7%), batch 11904 -> 12277 (+3.1%, A spread 6.3%)
  k  10 nprobe  128: single 644 -> 644 (+0.0%, A spread 1.7%), batch 7638 -> 7685 (+0.6%, A spread 5.6%)
  k  10 nprobe  256: single 402 -> 393 (-2.2%, A spread 2.8%), batch 4627 -> 4461 (-3.6%, A spread 0.7%)
  k  10 nprobe 1024: single 130 -> 126 (-2.5%, A spread 0.3%), batch 1364 -> 1318 (-3.4%, A spread 0.8%)
  k 100 nprobe   64: single 921 -> 905 (-1.8%, A spread 1.0%), batch 11744 -> 11609 (-1.1%, A spread 8.9%)
  k 100 nprobe  128: single 631 -> 618 (-2.1%, A spread 2.3%), batch 7579 -> 7543 (-0.5%, A spread 0.9%)
  k 100 nprobe  256: single 389 -> 379 (-2.6%, A spread 2.7%), batch 4418 -> 4415 (-0.1%, A spread 2.7%)
  k 100 nprobe 1024: single 128 -> 125 (-2.1%, A spread 0.3%), batch 1352 -> 1318 (-2.5%, A spread 0.9%)
arxiv 10 8 2998 2996 3005 2933 | 52784 49079 52105 50277
arxiv 10 16 2243 2237 2243 2258 | 30601 31350 29707 31076
arxiv 10 32 1510 1466 1509 1509 | 18219 18103 18015 18247
arxiv 10 48 1131 1111 1113 1110 | 11557 12278 12140 12516
arxiv 100 8 2729 2690 2714 2628 | 40166 42039 46547 47718
arxiv 100 16 2069 2022 2063 2058 | 30078 25474 25740 29540
arxiv 100 32 1424 1395 1410 1395 | 17553 14944 16884 15351
arxiv 100 48 1075 1065 1062 1085 | 12215 12114 12383 12547
qwen 10 64 951 950 956 967 | 11520 12346 12208 12288
qwen 10 128 649 654 634 638 | 7845 7687 7683 7430
qwen 10 256 396 390 396 407 | 4610 4447 4475 4643
qwen 10 1024 129 127 126 130 | 1370 1323 1313 1359
qwen 100 64 926 889 920 917 | 11200 10999 12219 12288
qwen 100 128 639 626 611 624 | 7546 7496 7589 7613
qwen 100 256 384 377 381 394 | 4476 4408 4423 4360
qwen 100 1024 128 125 126 128 | 1346 1324 1311 1358
```
