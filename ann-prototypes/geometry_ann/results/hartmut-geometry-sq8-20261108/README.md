# Shared convex envelopes and residual geometry — complete (see REPORT.md)

Remote root: `/home/sean/experiments/TieredQ-issue44-geometry-sq8-20261108`.
New source/build snapshot; previous large-study source and results are untouched.
Use `status.json`, `job.exitcode`, and per-model `exit.json` for progress.
The Qwen input reuses the existing verified NVMe INT8 corpus; no additional
9 GB dataset copy is made. Experiments hold the shared benchmark lock and run
serially on CPU 2 of hartmut, using the previous Release compiler flags.

## Scope

Fixed hierarchical k-means, 4 iterations, 64 coarse groups × 128 fine clusters.
One immutable partitioning/topology per direction count, using decoded INT8
training values. Seed 42. Compare centroid ranking, enclosing radii, shared
projection intervals, and projection intervals with a residual shell. Both
pure-bound and 50/50 centroid/bound priorities are evaluated. The new metadata
never changes partition training or the graph. Geometry pruning encloses the
**decoded INT8** members, with full-population extrema and numerical padding.
No sampled hull is used for pruning.

Full Qwen trials use **8, 16, 32 directions**, all 8,840,823 vectors × 1,024D,
250 calibration queries, and 750 disjoint held-out queries. Matched vector caps:
35,000 / 176,816 / 442,041. Flat discovery evaluates every cluster, isolating
ranking from graph discovery. Graph controls use 2,048/4,096 metadata evaluations
and compare centroid routing with residual/hybrid routing. Geometry settings
use routing-score scan order, so they actually affect the scanned clusters.

Each build calibrates all settings, freezes the fastest settings reaching
95%/99%/99.5% calibration recall per routing policy, and only then loads held-out
queries. Unreached targets retain explicit diagnostic labels. Selected held-out
settings run three query repeats on the same index. Baseline and metadata
variants share one build, so its combined build cost is not a standalone cost
measurement for each ablation. Basis and full-population metadata construction
have separate build counters. These are three independently timed builds, not
three timing repeats of one build. No setting is chosen using held-out results.

The graph still discovers candidates before scanning them. Interleaved graph
navigation/scanning is deferred until the flat experiment demonstrates useful
shape information; it is not part of this first comparison.

## Geometry

Let the rows of U be shared, stored FP32 PCA directions learned on 4,096
training samples. Extrema over every cluster member define slabs containing its
convex hull. Query projections are reused for every cluster. Intervals and
centroid projections are stored as padded FP64 endpoints in this prototype.
Consequently this implementation uses more metadata than the earlier 2.1 MB
estimate for FP32 min/max intervals alone. Actual index capacities and process
RSS are measured, rather than using that estimate as a RAM result.

The residual extension does not assume perfect orthogonality of stored FP32
vectors. With a conservative upper bound L >= ||U||², define

    BᵀB = I - UᵀU/L
    residual(x) = ||B(x-c)||

where c is the cluster centroid. Store the full-population min/max residual
norm. The positive-semidefinite residual map need not be explicitly formed:

    residual(x)² = ||x-c||² - ||U(x-c)||²/L.

Numerical intervals expand both terms before subtraction/square roots. If g_i
is query-to-projection-interval separation and h is separation from the residual
norm interval, then

    sum(g_i²)/L + h² <= min_{x in cluster} ||q-x||².

This follows by decomposing squared distance into the projected and residual
seminorms and applying the reverse triangle inequality to residual norms.
Combine with the enclosing-ball bound by taking their maximum. The original
shared-only control additionally takes the strongest individual normalized slab
bound; with nonorthogonal directions its bound need not be dominated by the
new joint residual expression. No universal tightness improvement is assumed.

Graph pruning only rejects the vectors of a cluster. It never cuts traversal
to graph neighbors. The global PCA basis has full-dimensional dense directions;
query reuse makes evaluation cheap per cluster, but constructing full-member
extrema still requires substantial work. That cost is part of the experiment.

## Diagnostics and checks

`geometry-diagnostics.csv` uses the first 50 **calibration** queries. It reports
true-neighbor-weighted cluster ranks under each score, positive bound counts,
bounds exceeding the reference distance, and bounds divided by that distance.
Reference distance is the largest decoded-vector distance among the supplied
FP32 top-10 IDs; it is not claimed to be the exact SQ8 kth-neighbor distance.
`unsafe_members` tests whether a true member's cluster bound exceeds its actual
decoded distance and must stay zero. These sampled checks complement the
mathematical bound and small exhaustive tests; they are not exhaustive validation
of every query on the full dataset.

Local Release and ASAN/UBSAN tests pass, as do remote Release tests and an
end-to-end fixture. Checks cover a thin cluster where the new residual bound
adds separation missed by radius/slabs, nonorthogonal/duplicated directions,
very small/large query scales, native INT8 full-search agreement for flat/graph/
hierarchy policies, hard caps, settings parsing, and diagnostic output.

## Initial SIFT sanity observation

This is a small **50-calibration-query** sanity check, not the full Qwen result.
All methods use the same 35K vector cap and full centroid discovery, 8 directions.

| Score | Recall@10 | p50 ms |
|---|---:|---:|
| Centroid | 0.982 | 1.871 |
| Radius, 50/50 hybrid | 0.982 | 2.531 |
| Shared intervals, 50/50 hybrid | 0.982 | 2.309 |
| Shared + residual, 50/50 hybrid | 0.984 | 2.744 |
| Shared + residual, pure bound | 0.922 | 2.850 |

The hybrid residual gain is only one hit among 500; it is not persuasive evidence
of improvement. Pure residual-bound ranking worsens the weighted mean true
cluster rank from 22.36 (centroid) to 62.84. Hybrid rank is 22.47. Around 4,408
of 8,192 clusters can be excluded at the reference distance, but this does not
establish useful ordering among the remaining competitive clusters. Sampled
unsafe-member count is zero. Combined index build took 28.36 s, including
2.63 s for shared ranges/residual metadata. Full Qwen conclusions are pending.

## Reproduction / artifacts

Build `source/` with CMake Release and run `study.py` from the remote root.
`preflight.py source build` exercises the end-to-end fixture using uv.
`source-sha256.json` identifies the frozen source snapshot.
Run `summarize.py ROOT` after copying small `results/` artifacts; it validates
repeats, memory-row alignment, hard caps and sampled safety counts. Timing/RSS
are never inferred from index payload size. Large input data stays outside git.
