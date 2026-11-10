# Hartmut AVX-512 scan and metadata ablation (#44)

Intel i9-7980XE, Linux x86_64, GCC 16.2.1, CMake Release, one thread,
pinned to CPU 2. Three sequential runs per configuration, medians below.
AVX-512 is selected by runtime CPU detection; AVX-512 instructions were also
verified in the generated scan kernel. No frequency locking. See environment.txt.

Full SIFT1M, FP32 squared L2, 250 calibration / 1,000 held-out queries, seed 42.
The source SHA256, all exported-file hashes, and both query-ID splits exactly
match results/sift1m-graph-20261106/dataset-manifest.json. Dataset exports remain
on hartmut at /home/sean/experiments/TieredQ-issue44-simd-20261107/data.

## Results

| Configuration | Recall@10 | p50 ms | Build seconds | Persistent overhead MB |
| --- | ---: | ---: | ---: | ---: |
| Original double scan, full metadata, 2% cap | 0.9671 | 1.911 | 28.59 | 28.92 |
| AVX-512 filter, full metadata, 2% cap | 0.9671 | 1.200 | 28.55 | 28.92 |
| AVX-512 filter, unused metadata omitted, 2% cap | 0.9671 | 1.151 | 17.20 | 10.30 |
| Original double scan, full metadata, 5% cap | 0.9946 | 4.466 | 28.59 | 28.92 |
| AVX-512 filter, full metadata, 5% cap | 0.9946 | 2.855 | 28.55 | 28.92 |
| AVX-512 filter, unused metadata omitted, 5% cap | 0.9946 | 2.724 | 17.20 | 10.30 |
| Faiss IVFFlat, nprobe 128 | 0.9815 | 0.862 | 41.36 | 12.26* |
| Faiss IVFFlat, nprobe 256 | 0.9952 | 1.527 | 41.36 | 12.26* |

Rows above use centroid-only graph routing. Radius/hybrid routing is also in
the CSVs; it gives 0.9672 recall at 2%, 0.9946 at 5%, and no material latency win.
All prototype configurations retain identical recall and scanned work after
filtering/removing unused projections: about 19,996 / 49,999 vectors. IVF scans
about 18,608 / 35,993 vectors. These are nearby recall points, not exactly matched
recall; the lower-recall IVF row actually has higher recall than the prototype.

The scan filter cuts p50 by 36–37%. Removing unused projections cuts build time
by 40% and owned overhead by 64%. It does not eliminate radius metadata, which
is needed for the radius ablation. Prototype build RSS is about 1.08 GB (lean)
versus 1.41 GB for IVF. RSS includes inputs and temporary allocations, not just
the index. Raw vectors occupy 512 MB in all configurations. *IVF overhead is
serialized bytes minus raw, whereas prototype overhead counts owned capacities;
these are different proxies, not an allocator-exact RAM comparison. Different
clustering algorithms and sampling budgets also prevent an equal-work build claim.

**Conclusion:** faster builds and small index overhead are promising, but the
prototype still loses search latency to IVFFlat near 0.995 recall by about 1.8x.
No win over IVFFlat is established. More effective partitions and cheaper routing
remain necessary; richer maximum-extent geometry has not earned its cost.

## Reproduce and validate

Remote root: /home/sean/experiments/TieredQ-issue44-simd-20261107.
Build source/ with CMake Release; use GEOMETRY_FAST_SCAN=OFF for reference.
All requested settings and benchmark commands are in settings.csv and run.sh.
The reference/ and fast/ binaries used the earlier full-metadata harness,
archived as full-metadata-graph_benchmark.cpp (from repository commit 0128676).
To reproduce those binaries, substitute that file for graph_benchmark.cpp in
an isolated copy of the current prototype before building. The lean/ binary
uses the current harness, which builds only requested projection metadata.
Run its graph_bench with the same arguments/settings from run.sh. Do not run
benchmark processes concurrently. The three lean runs followed the reference,
fast, and Faiss sequence; their results are not randomized-order trials.

The actual command for Faiss replays frozen nprobe 128/256 on the held-out
queries; it does not recalibrate on that split. Logs use the script's generic
"calibration selections" label, which does not describe a new tuning step here.
Faiss query timings include Python API overhead. Only IVFFlat was rerun.

CTest passes for the AVX-512 and original scan builds. Additional tests cover
SIMD tails, near ties, underflow/overflow, complete-search equality, and smaller
metadata builds preserving graph search results while reducing owned storage.
These are isolated prototype changes; the production TieredQ library is untouched.
