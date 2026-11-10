# Edge supports and oracle coverage

Geometry and coverage diagnostics use the first 50 calibration queries.
Oracle costs cover the supplied FP32 top-10 IDs by scanning whole clusters.
They are optimistic coverage diagnostics, not guaranteed SQ8 search recall.
Ranked costs scan a complete cluster prefix without skipping oversized clusters.
Witness ordering is diagnostic only: its real-vector evaluations are not part
of the timed search configurations, and no witness latency win is claimed.

## qwen-support

| Score | Weight | Witness | Mean true-cluster rank | Mean oracle vectors for 10 | Mean ranked vectors for 10 | p95 ranked vectors for 10 |
|---|---:|---|---:|---:|---:|---:|
| none | 0.0 | False | 18.27 | 8640 | 130127 | 682907 |
| radius | 0.0 | False | 18.27 | 8640 | 130127 | 682907 |
| radius | 0.5 | False | 18.38 | 8640 | 132348 | 676040 |
| radius | 1.0 | False | 35.80 | 8640 | 365119 | 2364492 |
| support | 0.0 | False | 18.27 | 8640 | 130127 | 682907 |
| support | 0.5 | False | 17.87 | 8640 | 130371 | 634342 |
| support | 1.0 | False | 81.22 | 8640 | 768422 | 2687684 |
| support | 0.5 | True | 21.10 | 8640 | 133436 | 662722 |
| support | 1.0 | True | 54.89 | 8640 | 323790 | 1868511 |

Selections frozen independently for each bounds family. Recall differs; no exact matched-recall claim.

| Target | Bounds | Weight | Status | Recall | p50 ms | Build min | Serving GB | Peak build GB |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 0.95 | none | 0 | SELECTED | 0.9557 | 27.254 | 17.99 | 9.227 | 18.366 |
| 0.95 | radius | 0.5 | SELECTED | 0.9553 | 27.441 | 17.99 | 9.227 | 18.366 |
| 0.95 | support | 0 | SELECTED | 0.9557 | 27.762 | 17.99 | 9.227 | 18.366 |
| 0.99 | none | 0 | NOT_REACHED_DIAGNOSTIC | 0.9765 | 61.461 | 17.99 | 9.227 | 18.366 |
| 0.99 | radius | 0.5 | NOT_REACHED_DIAGNOSTIC | 0.9771 | 61.724 | 17.99 | 9.227 | 18.366 |
| 0.99 | support | 0.5 | NOT_REACHED_DIAGNOSTIC | 0.9756 | 62.108 | 17.99 | 9.227 | 18.366 |
| 0.995 | none | 0 | NOT_REACHED_DIAGNOSTIC | 0.9765 | 61.461 | 17.99 | 9.227 | 18.366 |
| 0.995 | radius | 0.5 | NOT_REACHED_DIAGNOSTIC | 0.9771 | 61.724 | 17.99 | 9.227 | 18.366 |
| 0.995 | support | 0.5 | NOT_REACHED_DIAGNOSTIC | 0.9756 | 62.108 | 17.99 | 9.227 | 18.366 |
## sift-sanity

| Score | Weight | Witness | Mean true-cluster rank | Mean oracle vectors for 10 | Mean ranked vectors for 10 | p95 ranked vectors for 10 |
|---|---:|---|---:|---:|---:|---:|
| none | 0.0 | False | 22.36 | 1475 | 14582 | 40901 |
| radius | 0.0 | False | 22.36 | 1475 | 14582 | 40901 |
| radius | 0.5 | False | 22.64 | 1475 | 14651 | 41366 |
| radius | 1.0 | False | 56.09 | 1475 | 34451 | 112036 |
| support | 0.0 | False | 22.36 | 1475 | 14582 | 40901 |
| support | 0.5 | False | 22.48 | 1475 | 15234 | 44451 |
| support | 1.0 | False | 73.08 | 1475 | 46751 | 107437 |
| support | 0.5 | True | 21.18 | 1475 | 13642 | 39712 |
| support | 1.0 | True | 43.13 | 1475 | 25317 | 76721 |
