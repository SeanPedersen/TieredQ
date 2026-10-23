# Native zvec IVF: exact TieredQ, fast TieredQ, and RaBitQ

Intel i9-7980XE on hartmut; 1,183,514 normalized GloVe100 base vectors; the same 500 held-out queries and Recall@10 truth for both codecs.

Only two execution settings: single queries pinned to CPU 2 with one thread, and batches of 500 queries with 18 OpenMP workers (one per physical core). One complete warmup, three timed passes, and one untimed evaluation pass. One process per point. Cache construction, loading, and evaluation are outside timing. All runs are serialized.

| Configuration | Threads | Exact TieredQ QPS | Fast TieredQ QPS | IVF-RaBitQ QPS | Fast / exact | Fast / RaBitQ | Recall exact / fast / RaBitQ |
|---|---:|---:|---:|---:|---:|---:|---|
| big-full | 1 | 31.7 | 64.9 | 179.0 | 2.05x | 0.36x | 0.9840 / 0.9724 / 0.8732 |
| big-full | 18 | 401.9 | 649.2 | 2289.5 | 1.62x | 0.28x | 0.9840 / 0.9724 / 0.8732 |
| big-p64 | 1 | 113.7 | 229.0 | 560.5 | 2.02x | 0.41x | 0.9538 / 0.9432 / 0.8618 |
| big-p64 | 18 | 1369.5 | 2271.9 | 6894.4 | 1.66x | 0.33x | 0.9538 / 0.9432 / 0.8618 |
| big-p16 | 1 | 431.2 | 835.6 | 1657.9 | 1.94x | 0.50x | 0.8550 / 0.8490 / 0.8044 |
| big-p16 | 18 | 4905.8 | 8239.0 | 17392.2 | 1.68x | 0.47x | 0.8550 / 0.8490 / 0.8044 |
| small-full | 1 | 70.4 | 90.7 | 230.1 | 1.29x | 0.39x | 0.6674 / 0.6668 / 0.2982 |
| small-full | 18 | 841.2 | 1045.3 | 3004.3 | 1.24x | 0.35x | 0.6674 / 0.6668 / 0.2982 |

## Configurations

- big: existing TieredQ 664-bit / 256-list and RaBitQ 4-bit / 256-list indexes; probes 256 (full), 64, and 16.
- small: existing TieredQ 216-bit / 64-list and RaBitQ 1-bit / 64-list indexes; full probe.
- These are the existing matched index configurations, with different recall. They are not iso-recall comparisons.

## Implementation and reproducibility

TieredQ source is local commit 6bad55c. IVF-TieredQ integration was copied from /Users/Studio-MacMini/projects/open-source/zvec. The integration currently calls exact score_planes directly, so the experiment snapshot adds a bridge to TieredQ internal fast-cache construction and AVX-512 scoring. It does not change the scan kernels. Each IVF list owns a cache; query preparation is reused across all probed lists, with the same exact coarse probe ranking, filters, heap selection, and native streamer batch parallelism. Fast search scans only selected lists. An environment variable enables cache construction at open. This is an experiment adapter, not a shipped zvec fast-mode API.

The actual native IvfRabitqStreamer is the existing parallel RaBitQ benchmark executable in the parent experiment. Its query and truth files have identical SHA256 hashes to the TieredQ inputs. The existing persisted indexes are reused; nothing is retrained.

Run: `python3 -u scripts/run.py`, then `python3 scripts/report.py`. Build: `cmake -S scripts -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j6`.

Artifacts: results/all.json, results/summary.csv, results/validation.json, per-run JSON, ids, scores, scanned counts, call timings and logs. sources/ contains the exact modified source snapshot. The patch.py script records initial bridge installation; the final sources/ snapshot is authoritative.

## Memory

| Configuration | Exact RSS MiB | Fast RSS MiB | IVF-RaBitQ RSS MiB |
|---|---:|---:|---:|
| big-full | 116.9 | 239.9 | 112.7 |
| small-full | 52.5 | 101.6 | 49.0 |

## Validation and limits

All IDs, float-score payloads and per-query scan counts are byte-identical between one thread and 18 threads within each engine. Exact and fast TieredQ scan the same rows per query. Full-probe scans cover all 1,183,514 rows. All outputs contain ten finite, correctly ordered scores and valid keys per query; recall is evaluated outside timing.

This is one process per configuration, with three timed passes, on a shared host. Background services were left running. Small differences should be treated as noise; QPS is batch throughput for 18 threads, not single-query latency. Fast caches add resident memory; the original compressed payload remains resident. Reported index-size statistics omit the experiment cache, so RSS is used above.
