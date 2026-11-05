# Issue #28: quick build-time check

One baseline run followed by one patched run on Hartmut (Intel i9-7980XE), OpenAI 100K x 1536,
cosine, 4 bits/dimension, 32,768 sampled training rows, seed 0. Eight OpenMP threads pinned to CPUs 0-7;
OpenBLAS uses one thread. Dataset loading and compilation are excluded. Native flat fast index; default fine tiers.

| Stage | Baseline | Fix | Change |
|---|---:|---:|---:|
| Training | 2.09570 s | 2.10289 s | +0.34% |
| Add/encode | 3.80649 s | 3.81676 s | +0.27% |
| Total build | 5.90219 s | 5.91966 s | +0.30% |

Both runs produced model ID 254528849 and 77,663,024 storage bytes. No notable regression in this quick check;
one pair cannot establish statistical significance or measure fallback cost on other data.

The baseline is the existing Hartmut grid-fix build at
`/home/sean/experiments/TieredQ-issue26-gridfix-20261105/build/libdv_tieredq.a`.
The fixed archive replaces only its training object, compiled with the empirical-statistics fix and otherwise the
same source and compiler flags. Scratch directory: `sean@hartmut:/home/sean/experiments/TieredQ-issue28-quick.CXxKTg`.
Input: `/home/sean/experiments/TieredQ-issue20-ship-20261029/data/openai/base.f32`.

`build_time.cpp` is the driver; the JSON files contain raw wall times and the profile files contain stage timings.
Both runs used `OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 DV_TIEREDQ_PROFILE=1 taskset -c 0-7`.
