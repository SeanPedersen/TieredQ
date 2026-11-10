# Leaf assignment vs Faiss IVF-SQ8 (Qwen 8.84M x 1024, INT8)

Held-out: 750 queries; settings frozen on 250 calibration queries. Builds: 32 threads; search: 1 thread pinned.
Clusters = mean leaves scanned (ours) or nprobe (Faiss). Vectors = mean vectors scanned / distance evaluations.

| Model | Target | Status | Setting | Recall | Clusters | Vectors | p50 ms | Routing ms |
|---|---:|---|---|---:|---:|---:|---:|---:|
| faiss-32k | 0.95 | SELECTED | nprobe96 | 0.9557 | 96.0 | 31301 | 12.67 | — |
| faiss-32k | 0.99 | SELECTED | nprobe1024 | 0.9880 | 1024.0 | 315461 | 49.87 | — |
| faiss-32k | 0.995 | NOT_REACHED_DIAGNOSTIC | nprobe1536 | 0.9900 | 1536.0 | 470999 | 70.25 | — |
| faiss-8k | 0.95 | SELECTED | nprobe48 | 0.9451 | 48.0 | 62029 | 9.87 | — |
| faiss-8k | 0.99 | SELECTED | nprobe256 | 0.9833 | 256.0 | 315817 | 42.96 | — |
| faiss-8k | 0.995 | NOT_REACHED_DIAGNOSTIC | nprobe512 | 0.9897 | 512.0 | 625699 | 82.87 | — |
| ours-32k-p0 | 0.95 | SELECTED | graph m8192 cap120000 | 0.9540 | 317.6 | 120000 | 21.74 | 6.01 |
| ours-32k-p0 | 0.95 | SELECTED | flat m32896 cap70000 | 0.9495 | 189.3 | 70000 | 25.24 | 14.31 |
| ours-32k-p0 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m16384 cap600000 | 0.9805 | 1606.9 | 600000 | 91.40 | 12.97 |
| ours-32k-p0 | 0.99 | SELECTED | flat m32896 cap442041 | 0.9857 | 1190.2 | 442041 | 74.66 | 14.31 |
| ours-32k-p0 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m16384 cap600000 | 0.9805 | 1606.9 | 600000 | 91.40 | 12.97 |
| ours-32k-p0 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m32896 cap600000 | 0.9881 | 1626.6 | 600000 | 95.51 | 14.33 |
| ours-32k-p8 | 0.95 | SELECTED | graph m8192 cap120000 | 0.9508 | 398.3 | 120000 | 23.07 | 6.76 |
| ours-32k-p8 | 0.95 | SELECTED | flat m32896 cap35000 | 0.9520 | 117.5 | 35000 | 20.70 | 14.11 |
| ours-32k-p8 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m16384 cap600000 | 0.9752 | 2021.5 | 600000 | 95.48 | 14.61 |
| ours-32k-p8 | 0.99 | SELECTED | flat m32896 cap350000 | 0.9883 | 1185.1 | 350000 | 63.48 | 14.22 |
| ours-32k-p8 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m16384 cap600000 | 0.9752 | 2021.5 | 600000 | 95.48 | 14.61 |
| ours-32k-p8 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m32896 cap600000 | 0.9907 | 2042.3 | 600000 | 98.40 | 14.33 |
| ours-8k-p0 | 0.95 | SELECTED | graph m4096 cap176816 | 0.9504 | 125.8 | 176816 | 25.49 | 2.86 |
| ours-8k-p0 | 0.95 | SELECTED | flat m8256 cap176816 | 0.9557 | 126.5 | 176816 | 27.04 | 3.55 |
| ours-8k-p0 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m4096 cap600000 | 0.9787 | 414.3 | 600000 | 79.45 | 3.07 |
| ours-8k-p0 | 0.99 | NOT_REACHED_DIAGNOSTIC | flat m8256 cap600000 | 0.9835 | 418.4 | 600000 | 81.09 | 3.57 |
| ours-8k-p0 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m4096 cap600000 | 0.9787 | 414.3 | 600000 | 79.45 | 3.07 |
| ours-8k-p0 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m8256 cap600000 | 0.9835 | 418.4 | 600000 | 81.09 | 3.57 |
| ours-8k-p4 | 0.95 | SELECTED | graph m4096 cap88408 | 0.9473 | 77.6 | 88408 | 14.23 | 2.66 |
| ours-8k-p4 | 0.95 | SELECTED | flat m8256 cap70000 | 0.9503 | 62.2 | 70000 | 13.61 | 3.64 |
| ours-8k-p4 | 0.99 | NOT_REACHED_DIAGNOSTIC | graph m4096 cap600000 | 0.9779 | 511.7 | 600000 | 80.45 | 2.92 |
| ours-8k-p4 | 0.99 | SELECTED | flat m8256 cap442041 | 0.9864 | 379.7 | 442041 | 61.87 | 3.71 |
| ours-8k-p4 | 0.995 | NOT_REACHED_DIAGNOSTIC | graph m4096 cap600000 | 0.9779 | 511.7 | 600000 | 80.45 | 2.92 |
| ours-8k-p4 | 0.995 | NOT_REACHED_DIAGNOSTIC | flat m8256 cap600000 | 0.9887 | 514.9 | 600000 | 82.23 | 3.70 |

| Model | Build s | Reassign s | Moved vectors | Leaves | Build peak anon GB | Build peak file GB | Build HWM GB | Serving anon GB | Serving RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| faiss-32k | 610 | 0 | — | 32768 | 14.34 | 9.09 | 23.43 | 14.31 | 14.34 |
| faiss-8k | 137 | 0 | — | 8192 | 14.17 | 9.09 | 23.26 | 14.15 | 14.18 |
| ours-32k-p0 | 230 | 0 | 0 | 32602 | 9.55 | 9.06 | 18.60 | 9.48 | 9.48 |
| ours-32k-p8 | 1540 | 1313 | 4520396 | 32602 | 9.56 | 9.06 | 18.62 | 9.49 | 9.49 |
| ours-8k-p0 | 77 | 0 | 0 | 8192 | 9.31 | 9.06 | 18.36 | 9.22 | 9.23 |
| ours-8k-p4 | 390 | 313 | 4773167 | 8192 | 9.32 | 9.06 | 18.38 | 9.23 | 9.24 |
