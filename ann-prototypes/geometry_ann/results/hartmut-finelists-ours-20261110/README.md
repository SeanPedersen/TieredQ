# Experiment 3, stage B: our index at 32K/64K/128K lists

Status: complete (exit 0); see `REPORT.md`. Remote root: `/home/sean/experiments/TieredQ-issue44-finelists-ours-20261110`.
Compressed routing (2 bits/dim, 4-bit scan cap, lazy expansion) and degree-16 graph routing, both with FP32 routing
distances, patience windows and fixed caps, on the stage A Lloyd labels. The graph builder was parallelized during the run,
before the 128K graph build; graphs are identical for any thread count.

| Model | Held-out highlight | Routing ms |
|---|---|---:|
| 32K compressed, lazy | 0.9505 @ 20.8K vectors, 3.9 / 8.4 ms | 1.44 |
| 64K compressed, lazy | 0.9624 @ 19.6K vectors, 5.2 / 9.2 ms | 2.79 |
| 128K compressed, lazy | 0.9709 @ 18.6K vectors, 7.7 / 11.6 ms; 0.9836 @ 14.0 / 33.8 ms | 5.37 |
| 64K graph, budget 2048 | 0.9485 @ 19.9K vectors, 3.6 / 8.2 ms (plateaus at 0.967) | 1.13 |
| 128K graph, budget 4096 | 0.9569 @ 18.6K vectors, 5.0 / 8.7 ms | 2.54 |

One caveat: the 64K graph's calibration latencies overlapped a low-priority compile on hartmut. Its held-out replay ran
afterwards.
