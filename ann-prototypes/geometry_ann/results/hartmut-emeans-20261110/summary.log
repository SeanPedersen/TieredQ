# Flat partitions: Faiss Lloyd vs E-means through our index (Qwen 8.84M x 1024, INT8)

Compare with ../hartmut-assignment-20261110/REPORT.md (hierarchical + reassignment, Faiss IVF-SQ8 search).
Held-out: 750 queries; settings frozen on 250 calibration queries. Builds: 32 threads; search: 1 thread pinned.
Clusters = mean leaves scanned (ours) or nprobe (Faiss). Vectors = mean vectors scanned / distance evaluations.

| Model | Target | Status | Setting | Recall | Clusters | Vectors | p50 ms | Routing ms |
|---|---:|---|---|---:|---:|---:|---:|---:|
| emeans-32k-s40-e1 | 0.95 | SELECTED | graph m8192 cap35000 | 0.9359 | 98.4 | 35000 | 11.65 | 6.74 |
| emeans-32k-s40-e1 | 0.95 | SELECTED | flat m32896 cap35000 | 0.9433 | 98.8 | 35000 | 19.95 | 14.16 |
| emeans-32k-s40-e1 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9805 | 1491.8 | 600000 | 87.89 | 7.17 |
| emeans-32k-s40-e1 | 0.99 | SELECTED | flat m32896 cap350000 | 0.9853 | 899.7 | 350000 | 64.54 | 14.20 |
| emeans-32k-s40-e1 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9805 | 1491.8 | 600000 | 87.89 | 7.17 |
| emeans-32k-s40-e1 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m32896 cap600000 | 0.9895 | 1516.8 | 600000 | 98.79 | 14.22 |
| emeans-32k-s40-e2 | 0.95 | SELECTED | graph m8192 cap120000 | 0.9619 | 322.3 | 119999 | 22.77 | 6.83 |
| emeans-32k-s40-e2 | 0.95 | SELECTED | flat m32896 cap35000 | 0.9507 | 99.5 | 35000 | 22.20 | 14.25 |
| emeans-32k-s40-e2 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9757 | 1519.0 | 599999 | 85.61 | 7.16 |
| emeans-32k-s40-e2 | 0.99 | SELECTED | flat m32896 cap250000 | 0.9836 | 659.2 | 250000 | 51.37 | 14.28 |
| emeans-32k-s40-e2 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9757 | 1519.0 | 599999 | 85.61 | 7.16 |
| emeans-32k-s40-e2 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m32896 cap600000 | 0.9907 | 1541.8 | 600000 | 98.14 | 14.29 |
| emeans-32k-s40-e4 | 0.95 | SELECTED | graph m8192 cap50000 | 0.9437 | 137.6 | 49998 | 13.50 | 6.71 |
| emeans-32k-s40-e4 | 0.95 | SELECTED | flat m32896 cap35000 | 0.9496 | 98.1 | 34999 | 22.61 | 14.11 |
| emeans-32k-s40-e4 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9747 | 1523.9 | 599998 | 85.01 | 7.04 |
| emeans-32k-s40-e4 | 0.99 | SELECTED | flat m32896 cap250000 | 0.9831 | 659.4 | 249999 | 50.94 | 14.07 |
| emeans-32k-s40-e4 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9747 | 1523.9 | 599998 | 85.01 | 7.04 |
| emeans-32k-s40-e4 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m32896 cap600000 | 0.9912 | 1552.7 | 599999 | 97.85 | 14.15 |
| lloyd-32k | 0.95 | SELECTED | graph m8192 cap35000 | 0.9440 | 116.3 | 35000 | 11.32 | 6.35 |
| lloyd-32k | 0.95 | SELECTED | flat m32896 cap35000 | 0.9580 | 117.1 | 35000 | 21.59 | 14.66 |
| lloyd-32k | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9752 | 1972.4 | 600000 | 87.61 | 6.44 |
| lloyd-32k | 0.99 | SELECTED | flat m32896 cap250000 | 0.9852 | 844.7 | 250000 | 52.39 | 14.73 |
| lloyd-32k | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m8192 cap600000 | 0.9752 | 1972.4 | 600000 | 87.61 | 6.44 |
| lloyd-32k | 0.995 | SELECTED | flat m32896 cap600000 | 0.9909 | 2036.9 | 600000 | 100.22 | 14.68 |

| Model | Train s | Final assign s | Index build s | Total s | Train peak anon GB | Leaves | Empty | Assignment objective | Build peak anon GB | Serving anon GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| emeans-32k-s40-e1 | 78 | 378 | 190 | 646 | 6.74 | 32698 | 70 | 4.43e+06 | 9.74 | 9.66 |
| emeans-32k-s40-e2 | 148 | 374 | 172 | 695 | 6.66 | 32766 | 2 | 4.31e+06 | 9.70 | 9.63 |
| emeans-32k-s40-e4 | 278 | 379 | 172 | 830 | 6.69 | 32768 | 0 | 4.273e+06 | 9.70 | 9.63 |
| lloyd-32k | 223 | 378 | 166 | 767 | 6.77 | 32768 | 0 | 4.363e+06 | 9.71 | 9.64 |
