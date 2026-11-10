# Router tuning: probed supers (#47)

Remote root: `/home/sean/experiments/TieredQ-issue44-routertune-20261110`. Loads the router study's scratch indexes and
reruns the router at expansions 64/128/256 (at least 32 supers), top-40 SQ8 re-rank, 750 held-out queries, one pinned
core.

| Lists | nprobe | Exact routing | Expansion 32 | **Expansion 64** | Expansion 128 | Expansion 256 |
|---|---:|---|---|---|---|---|
| 64K | 128 | 0.9591 @ 10.1 ms | 0.9468 @ 2.26 | **0.9544 @ 2.47** | 0.9584 @ 2.82 | 0.9587 @ 3.37 |
| 64K | 256 | 0.9704 @ 11.2 | 0.9643 @ 3.41 | **0.9697 @ 3.78** | 0.9701 @ 4.31 | 0.9705 @ 5.15 |
| 64K | 512 | 0.9799 @ 13.3 | 0.9787 @ 5.68 | **0.9788 @ 6.34** | 0.9792 @ 7.25 | 0.9792 @ 7.27 |
| 128K | 256 | 0.9709 @ 20.4 | 0.9596 @ 3.12 | **0.9663 @ 3.57** | 0.9692 @ 4.41 | 0.9695 @ 5.91 |
| 128K | 1024 | 0.9849 @ 25.4 | 0.9836 @ 9.25 | **0.9843 @ 10.7** | 0.9845 @ 13.4 | 0.9845 @ 13.4 |

Expansion 64 is the knee: higher expansions add 0.3–1.5 ms for at most 0.005 recall. It is now the library default
(with at least 32 probed supers). At 64K lists the router costs about 1.4 ms per query, against about 9 ms for exact
ranking.
