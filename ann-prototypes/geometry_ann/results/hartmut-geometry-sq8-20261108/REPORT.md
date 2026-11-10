# Geometry experiment results

Experiment status: **complete**.

Full-corpus Qwen runs use 250 calibration / 750 held-out queries, seed42, CPU2.
Rows below are frozen calibration selections. Targets are not held-out guarantees.
Latency is the median p50 of three query repeats; GB are decimal process RSS.
SIFT sanity results use 50 calibration queries only and are kept in summary.json.

| Model | Target | Route | Bounds | Weight | Recall | p50 ms | Build min | Serving GB | Peak build GB |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| qwen-k16 | 0.95 (SELECTED) | graph | none | 0 | 0.9504 | 25.829 | 22.39 | 9.235 | 18.378 |
| qwen-k16 | 0.95 (SELECTED) | flat | none | 0 | 0.9557 | 27.465 | 22.39 | 9.235 | 18.378 |
| qwen-k16 | 0.99 (NOT_REACHED_DIAGNOSTIC) | graph | none | 0 | 0.9721 | 59.957 | 22.39 | 9.235 | 18.378 |
| qwen-k16 | 0.99 (NOT_REACHED_DIAGNOSTIC) | flat | none | 0 | 0.9765 | 61.697 | 22.39 | 9.235 | 18.378 |
| qwen-k16 | 0.995 (NOT_REACHED_DIAGNOSTIC) | graph | none | 0 | 0.9721 | 59.957 | 22.39 | 9.235 | 18.378 |
| qwen-k16 | 0.995 (NOT_REACHED_DIAGNOSTIC) | flat | none | 0 | 0.9765 | 61.697 | 22.39 | 9.235 | 18.378 |
| qwen-k32 | 0.95 (SELECTED) | graph | none | 0 | 0.9504 | 25.822 | 28.14 | 9.240 | 18.383 |
| qwen-k32 | 0.95 (SELECTED) | flat | none | 0 | 0.9557 | 27.758 | 28.14 | 9.240 | 18.383 |
| qwen-k32 | 0.99 (NOT_REACHED_DIAGNOSTIC) | graph | shared-residual | 0.5 | 0.9705 | 61.998 | 28.14 | 9.240 | 18.383 |
| qwen-k32 | 0.99 (NOT_REACHED_DIAGNOSTIC) | flat | none | 0 | 0.9765 | 61.982 | 28.14 | 9.240 | 18.383 |
| qwen-k32 | 0.995 (NOT_REACHED_DIAGNOSTIC) | graph | shared-residual | 0.5 | 0.9705 | 61.998 | 28.14 | 9.240 | 18.383 |
| qwen-k32 | 0.995 (NOT_REACHED_DIAGNOSTIC) | flat | none | 0 | 0.9765 | 61.982 | 28.14 | 9.240 | 18.383 |
| qwen-k8 | 0.95 (SELECTED) | graph | none | 0 | 0.9504 | 25.780 | 19.66 | 9.233 | 18.376 |
| qwen-k8 | 0.95 (SELECTED) | flat | none | 0 | 0.9557 | 27.372 | 19.66 | 9.233 | 18.376 |
| qwen-k8 | 0.99 (NOT_REACHED_DIAGNOSTIC) | graph | none | 0 | 0.9721 | 60.036 | 19.66 | 9.233 | 18.376 |
| qwen-k8 | 0.99 (NOT_REACHED_DIAGNOSTIC) | flat | none | 0 | 0.9765 | 61.627 | 19.66 | 9.233 | 18.376 |
| qwen-k8 | 0.995 (NOT_REACHED_DIAGNOSTIC) | graph | none | 0 | 0.9721 | 60.036 | 19.66 | 9.233 | 18.376 |
| qwen-k8 | 0.995 (NOT_REACHED_DIAGNOSTIC) | flat | none | 0 | 0.9765 | 61.627 | 19.66 | 9.233 | 18.376 |

Calibration ranking diagnostics on first50 queries:

| Model | Bounds | Weight | Mean true-cluster rank | Bounds above reference |
|---|---|---:|---:|---:|
| qwen-k16 | none | 0.0 | 18.27 | 0.00 |
| qwen-k16 | radius | 0.0 | 18.27 | 51.10 |
| qwen-k16 | radius | 0.5 | 18.38 | 51.10 |
| qwen-k16 | radius | 1.0 | 35.80 | 51.10 |
| qwen-k16 | shared | 0.0 | 18.27 | 51.10 |
| qwen-k16 | shared | 0.5 | 18.40 | 51.10 |
| qwen-k16 | shared | 1.0 | 48.42 | 51.10 |
| qwen-k16 | shared-residual | 0.0 | 18.27 | 57.00 |
| qwen-k16 | shared-residual | 0.5 | 18.40 | 57.00 |
| qwen-k16 | shared-residual | 1.0 | 49.31 | 57.00 |
| qwen-k32 | none | 0.0 | 18.27 | 0.00 |
| qwen-k32 | radius | 0.0 | 18.27 | 51.10 |
| qwen-k32 | radius | 0.5 | 18.38 | 51.10 |
| qwen-k32 | radius | 1.0 | 35.80 | 51.10 |
| qwen-k32 | shared | 0.0 | 18.27 | 51.30 |
| qwen-k32 | shared | 0.5 | 18.33 | 51.30 |
| qwen-k32 | shared | 1.0 | 54.15 | 51.30 |
| qwen-k32 | shared-residual | 0.0 | 18.27 | 61.80 |
| qwen-k32 | shared-residual | 0.5 | 18.32 | 61.80 |
| qwen-k32 | shared-residual | 1.0 | 54.33 | 61.80 |
| qwen-k8 | none | 0.0 | 18.27 | 0.00 |
| qwen-k8 | radius | 0.0 | 18.27 | 51.10 |
| qwen-k8 | radius | 0.5 | 18.38 | 51.10 |
| qwen-k8 | radius | 1.0 | 35.80 | 51.10 |
| qwen-k8 | shared | 0.0 | 18.27 | 51.10 |
| qwen-k8 | shared | 0.5 | 18.43 | 51.10 |
| qwen-k8 | shared | 1.0 | 51.10 | 51.10 |
| qwen-k8 | shared-residual | 0.0 | 18.27 | 55.02 |
| qwen-k8 | shared-residual | 0.5 | 18.40 | 55.02 |
| qwen-k8 | shared-residual | 1.0 | 50.32 | 55.02 |
| sift-sanity-k8 | none | 0.0 | 22.36 | 0.00 |
| sift-sanity-k8 | radius | 0.0 | 22.36 | 3372.78 |
| sift-sanity-k8 | radius | 0.5 | 22.64 | 3372.78 |
| sift-sanity-k8 | radius | 1.0 | 56.09 | 3372.78 |
| sift-sanity-k8 | shared | 0.0 | 22.36 | 4338.56 |
| sift-sanity-k8 | shared | 0.5 | 22.53 | 4338.56 |
| sift-sanity-k8 | shared | 1.0 | 67.06 | 4338.56 |
| sift-sanity-k8 | shared-residual | 0.0 | 22.36 | 4407.78 |
| sift-sanity-k8 | shared-residual | 0.5 | 22.47 | 4407.78 |
| sift-sanity-k8 | shared-residual | 1.0 | 62.84 | 4407.78 |

See README.md for the bound construction, numerical contract, query protocol, and limitations.
summary.json retains calibration curves and query paging counters. A zero unsafe-member
count covers the diagnostic sample; exhaustive small tests and the mathematical bound
provide independent checks. No issue comment or external publication was made.
