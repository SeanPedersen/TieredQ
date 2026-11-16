# #55 non-regression gate: aae87c7 (A) vs fba7ef5 (B)

## Kernel scans (whole-run block, ns per row, CPU 2)

| model | A 1 | B 2 | B 3 | A 4 | B / A | B tiles |
|---|---|---|---|---|---|---|
| 768-175 | 15.97 | 15.54 | 15.58 | 15.66 | -1.6% | 14.96 |
| 1024-100 | 13.14 | 13.04 | 12.88 | 13.00 | -0.8% | 11.68 |

## arxiv flat

Saved files identical across builds: True

| build | build s | tier s | storage MiB | peak RSS GiB |
|---|---|---|---|---|
| 1A | 21.9 | 0.0 | 233.1 | 4.68 |
| 2B | 21.8 | 0.0 | 233.1 | 4.68 |
| 3B | 22.3 | 0.0 | 233.1 | 4.68 |
| 4A | 22.0 | 0.0 | 233.1 | 4.68 |

Ids and scores identical to serve 1A: 1A True, 2B True, 3B True, 4A True

| nprobe | single QPS A | B | B / A | A-A spread | p50 us A | B | p95 us A | B |
|---|---|---|---|---|---|---|---|---|
| 8 | 2738 | 2764 | +1.0% | 1.1% | 363 | 360 | 403 | 397 |
| 16 | 2074 | 2079 | +0.2% | 0.4% | 482 | 481 | 530 | 530 |
| 32 | 1426 | 1420 | -0.4% | 0.3% | 701 | 703 | 772 | 776 |
| 48 | 1085 | 1087 | +0.2% | 0.4% | 921 | 918 | 1010 | 1008 |

Batch QPS (32 threads, nprobe 16): A 30789, B 30591 (-0.6%, A-A spread 4.8%). Serving RssAnon: A 0.273 GiB, B 0.272 GiB.

## arxiv nested

Saved files identical across builds: True

| build | build s | tier s | storage MiB | peak RSS GiB |
|---|---|---|---|---|
| A | 22.5 | 11.5 | 150.4 | 4.51 |
| B | 22.4 | 11.5 | 150.4 | 4.51 |

Ids and scores identical to serve 1A: 1A True, 2B True, 3B True, 4A True

| nprobe | single QPS A | B | B / A | A-A spread | p50 us A | B | p95 us A | B |
|---|---|---|---|---|---|---|---|---|
| 32 | 506 | 505 | -0.1% | 0.2% | 1976 | 1972 | 2095 | 2095 |
| 64 | 441 | 440 | -0.3% | 0.5% | 2263 | 2268 | 2408 | 2407 |
| 128 | 356 | 351 | -1.5% | 0.2% | 2806 | 2849 | 2996 | 3035 |

Batch QPS (32 threads, nprobe 64): A 2580, B 2583 (+0.1%, A-A spread 0.6%). Serving RssAnon: A 0.222 GiB, B 0.221 GiB.

## qwen flat

Saved files identical across builds: True

| build | build s | tier s | storage MiB | peak RSS GiB |
|---|---|---|---|---|
| 1A | 220.0 | 0.0 | 1469.8 | 17.56 |
| 2B | 206.0 | 0.0 | 1469.8 | 19.35 |
| 3B | 205.6 | 0.0 | 1469.8 | 19.36 |
| 4A | 204.8 | 0.0 | 1469.8 | 19.35 |

Ids and scores identical to serve 1A: 1A True, 2B True, 3B True, 4A True

| nprobe | single QPS A | B | B / A | A-A spread | p50 us A | B | p95 us A | B |
|---|---|---|---|---|---|---|---|---|
| 64 | 969 | 985 | +1.7% | 0.8% | 1024 | 1002 | 1104 | 1082 |
| 128 | 656 | 662 | +1.0% | 1.1% | 1509 | 1502 | 1647 | 1599 |
| 256 | 400 | 399 | -0.4% | 0.2% | 2472 | 2486 | 2696 | 2705 |
| 1024 | 131 | 129 | -1.4% | 0.6% | 7554 | 7660 | 8495 | 8639 |

Batch QPS (32 threads, nprobe 256): A 4530, B 4497 (-0.7%, A-A spread 0.0%). Serving RssAnon: A 1.504 GiB, B 1.504 GiB.

## qwen nested

Saved files identical across builds: True

| build | build s | tier s | storage MiB | peak RSS GiB |
|---|---|---|---|---|
| A | 206.7 | 98.0 | 1514.0 | 19.49 |
| B | 209.3 | 98.6 | 1514.0 | 19.53 |

Ids and scores identical to serve 1A: 1A True, 2B True, 3B True, 4A True

| nprobe | single QPS A | B | B / A | A-A spread | p50 us A | B | p95 us A | B |
|---|---|---|---|---|---|---|---|---|
| 128 | 386 | 381 | -1.3% | 4.9% | 2571 | 2605 | 2792 | 2803 |
| 256 | 276 | 272 | -1.5% | 4.4% | 3593 | 3649 | 3916 | 3929 |
| 512 | 183 | 178 | -2.6% | 3.3% | 5419 | 5565 | 6019 | 6100 |
| 1024 | 112 | 109 | -3.1% | 2.4% | 8778 | 9086 | 9790 | 10012 |

Batch QPS (32 threads, nprobe 256): A 2707, B 2686 (-0.8%, A-A spread 1.4%). Serving RssAnon: A 1.555 GiB, B 1.557 GiB.
