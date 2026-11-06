203 configurations; 609 timed passes.

| R@10 target | zvec config | QPS | B/vector | TieredQ config | QPS | B/vector | Speed |
|---|---|---:|---:|---|---:|---:|---:|
| 0.80 | zvec 3b np4 | 4,832 | 340.2 | tieredq 1.16b np8 | 5,276 | 132.6 | 1.09× |
| 0.90 | zvec 5b np16 | 3,985 | 532.4 | tieredq 2.16b np16 | 2,776 | 232.0 | 0.70× |
| 0.95 | zvec 7b np32 | 3,312 | 724.4 | tieredq 4.16b np32 | 870 | 432.9 | 0.26× |
| 0.97 | zvec 7b np64 | 2,510 | 724.4 | nested-p1-c50 7b np64 | 715 | 696.0 | 0.28× |
| 0.98 | zvec 7b np64 | 2,510 | 724.4 | nested-p1-c50 7b np64 | 715 | 696.0 | 0.28× |
| 0.99 | zvec 7b np256 | 1,023 | 724.4 | nested-p1-c50 6b np128 | 615 | 608.0 | 0.60× |

TieredQ constrained to at most the fastest qualifying zvec configuration's index bytes:

| R@10 target | TieredQ config | QPS | B/vector | Speed vs zvec |
|---|---|---:|---:|---:|
| 0.80 | tieredq 1.16b np8 | 5,276 | 132.6 | 1.09× |
| 0.90 | tieredq 2.16b np16 | 2,776 | 232.0 | 0.70× |
| 0.95 | tieredq 4.16b np32 | 870 | 432.9 | 0.26× |
| 0.97 | nested-p1-c50 7b np64 | 715 | 696.0 | 0.28× |
| 0.98 | nested-p1-c50 7b np64 | 715 | 696.0 | 0.28× |
| 0.99 | nested-p1-c50 6b np128 | 615 | 608.0 | 0.60× |

Smallest qualifying index per engine (ties broken by QPS):

| R@10 target | zvec B/vector | QPS | TieredQ config | B/vector | QPS | Byte reduction |
|---|---:|---:|---|---:|---:|---:|
| 0.80 | 244.2 | 4,470 | tieredq 1.16b np8 | 132.6 | 5,276 | 45.7% |
| 0.90 | 340.2 | 3,371 | tieredq 2.16b np16 | 232.0 | 2,776 | 31.8% |
| 0.95 | 436.8 | 1,630 | nested-p2-c20 3b np64 | 309.4 | 724 | 29.2% |
| 0.97 | 532.4 | 1,550 | nested-p1-c50 4b np128 | 399.0 | 624 | 25.1% |
| 0.98 | 628.5 | 1,605 | nested-p2-c20 4b np256 | 407.4 | 346 | 35.2% |
| 0.99 | 724.4 | 1,023 | nested-p2-c20 5b np256 | 507.4 | 373 | 30.0% |
