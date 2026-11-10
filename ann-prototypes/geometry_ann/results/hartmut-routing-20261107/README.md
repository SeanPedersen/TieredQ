# Controlled routing and partition study (#44)

Completed on hartmut: i9-7980XE, GCC 16.2.1, CMake Release, one thread pinned
to CPU 2. SIFT1M: 1,000,000 x 128 FP32 base vectors, 250 calibration queries,
1,000 disjoint held-out queries. Dataset exports are the hash-verified split
from ../hartmut-simd-20261107/dataset-manifest.json. No normalization.

## Main findings

The graph provides a modest routing gain over a stronger flat control, while
fixed hierarchical k-means improves neighbor concentration over adaptive splits.
IVFFlat still wins search latency. No substantial serving-memory win is shown.

| At the >=0.99 calibration target | Held-out recall@10 | p50 ms | Mean vector scans | Build seconds |
| --- | ---: | ---: | ---: | ---: |
| Adaptive 4 iterations + graph | 0.9946 | 2.631 | 49,999 | 17.11 |
| Fixed k-means 4 iterations + graph | 0.9940 | 1.981 | 35,000 | 24.02 |
| Same fixed partitions + flat heap routing | 0.9946 | 2.176 | 35,000 | 23.68* |
| Fixed k-means 12 iterations + graph | 0.9936 | 1.974 | 35,000 | 40.65 |
| Faiss IVFFlat, nprobe 192 | 0.9915 | 1.197 | 27,410 | 40.49 |

Recall values are nearby operating points satisfying the same threshold, not
exactly equal recall. Full settings, calibration decisions, work counters,
p99s, and target failures are in summary.json and faiss-summary.json.

The graph evaluates 2,048 descriptors including 64 parents versus 8,192 leaf
descriptors for flat routing. It improves total p50 by about 9% over the flat
heap control, at a small observed recall difference. It is not a demonstrated
win over Faiss's optimized centroid quantizer.

Fixed 4-iteration training reduces the selected scan cap from 50K to 35K at
similar high recall, and p50 by about 25%, for a build increase from 17 to 24 s.
Twelve iterations do not justify their additional build cost on this split.
At the 0.995 calibration target, the selected fixed12 graph and flat settings
both miss the target on held-out queries (0.9940 and 0.9941). Do not retune
those settings using the test split. Fixed4's selected 50K graph setting does
reach 0.9969 held-out recall, with p50 2.632 ms.

## RAM

Decimal MB/GB; raw vectors always occupy 512 MB.

| Configuration | Owned persistent overhead | Raw + overhead | Process peak build RSS |
| --- | ---: | ---: | ---: |
| Adaptive + graph | 10.30 MB | 522.30 MB | 1.079 GB |
| Fixed 4 + graph | 10.27 MB | 522.27 MB | 1.073 GB |
| Fixed 12 + graph | 10.27 MB | 522.27 MB | 1.073 GB |
| Faiss IVFFlat | 12.26 MB serialized proxy | 524.26 MB serialized | 1.417 GB |

Graph edges/entries account for just 0.2624 MB, included in the overhead above.
Prototype overhead counts owned capacities and objects, excluding allocator
bookkeeping and query temporaries. Faiss uses serialized size minus raw bytes;
that is not allocator-exact resident memory. A roughly 2 MB difference against
a 512 MB corpus does not establish a meaningful serving-RAM advantage.

Build RSS is approximately 24% lower than Faiss in this harness, but includes
loaded inputs, transient buffers, packed index, runtime and allocator retention.
The prototype holds the 512 MB input and a separate packed vector copy during
these measurements. Dedicated steady-state serving RSS and peak query scratch
were not measured. Substantial serving-RAM savings require compressing the
vectors, not just reducing graph/geometry metadata. The production TieredQ
codecs are a candidate for a separate compressed-scan/reranking experiment.

## Protocol, controls and reproduction

Remote root: /home/sean/experiments/TieredQ-issue44-routing-20261107.
Data is a symlink to the preceding SIMD experiment's exports. The isolated
source is copied under source/. Build via CMake Release to build/.

Run study.py from that root. It evaluates 24 settings per model on calibration
queries: vector caps 10K/20K/35K/50K; graph descriptor budgets
512/1024/2048/4096; flat and hierarchical controls. The same immutable index,
partitions, graph, vector layout, and scan kernel serve each model's settings.
All prototype settings for targets 0.95/0.99/0.995 are frozen in decisions.json
before any held-out measurements. Selected settings are replayed three times,
with deterministic rebuilds and reversed model order on the middle repetition.
There is one seed; repetitions measure timing variability, not seed uncertainty.

Faiss calibrates nprobe 32/64/96/128/192/256 on the same 250 queries and replays
its selected settings for targets 0.95/0.99 on the held-out set. Calibration is
repeated per build. All three runs choose 96 and 192. Its nlist is 8192, training
sample 327680, and Lloyd iterations 4. Python API overhead is included in Faiss
query timing. Prototype sampling is 4096 per clustering call, so build workloads
are different. V0 uses 64 coarse groups x 128 children; V2 uses 8192 adaptive
leaves and target size 64. Graph degree 8, parent probes 4, refinements 2,
search seeds 4. Increasing --iterations changes both coarse and fine training.

The initial study binary used full sorting for flat routing. After identifying
that cost, the final implementation uses linear-time heap construction and pops
only candidates needed by the scan budget. The stronger flat control is replayed
for the promising fixed4 model using flat_control.py and flatheap/graph_bench.
It reuses the frozen settings, without retuning on held-out data. Three query
repeats share one immutable build (*23.68 seconds); all work counters and recall
are checked against the sorted control by summarize.py. Other models' flat rows
in summary.json remain explicitly labeled full_sort diagnostics. The graph gain
shrinks from roughly 24% against sorting to 9% against heap routing.

To reproduce the initial sorting baseline in a separate source copy, replace
the flat branch's make_heap/pop_heap loop with std::sort using
VisitGreater{}(b,a), followed by ordered scans. The final heap implementation
is the shipped prototype code. environment.txt describes the initial build;
flatheap-environment.txt captures final source hashes. Graph and flat phase
counters use different queue mechanics; flat scanning time includes heap pops.
Hierarchy interleaves routing/scanning and its phase split is not comparable.
Total p50 is the primary latency comparison.

CTest passes on hartmut including exact SIMD centroid equivalence, complete V0
graph coverage, bounds-mode validation, budgets, ties and SIMD tails. Local
AddressSanitizer/UndefinedBehaviorSanitizer tests also pass. Production TieredQ
is untouched. Compressed scans, boundary spill, better graph construction,
geometry pruning instrumentation and cross-dataset validation remain future work.
