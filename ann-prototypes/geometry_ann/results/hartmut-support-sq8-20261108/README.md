# Local graph-edge supports, oracle coverage, and witness ranking

Status: complete (2026-11-08 09:19 UTC, exit 0). Results in `results/`, summary in `REPORT.md`.
Remote root: `/home/sean/experiments/TieredQ-issue44-support-sq8-20261108`.
`status.json` records queue/build/run/complete/failure; `job.exitcode` is written
on termination. The job acquires the existing global benchmark lock **before**
compilation or testing, so it does not overlap the preceding timed experiments.
Its source snapshot, input hashes, results, and logs are isolated from those runs.

## First-principles questions

For a cluster C, the desired score is f_C(q) = min_{x in C} ||q-x||². Centroid
distance is only a proxy. A safe convex envelope supplies a lower bound on f;
actual member representatives supply upper bounds. Either can be valid yet too
loose to rank competitive clusters well. A graph accelerates discovering proxy
scores; it does not by itself repair their mismatch with f.

The completed Qwen k8 diagnostic illustrates the mismatch: on its first50
calibration queries, shared/residual bounds average 0.0843 times the reference
distance and exclude only 55.02/8192 clusters at that distance. Pure bound
ranking moves true-neighbor-weighted mean cluster rank from 18.27 to 50.32.
Hybrid rank is 18.40. No unsafe-member comparisons were observed in that sample.
Reference is the largest decoded SQ8 distance of the supplied FP32 top-10 IDs,
not the exact SQ8 kth-neighbor distance. These observations motivate the next
experiments; they do not establish an impossibility result for geometric bounds.

This experiment separates:

1. **Partition cost:** an oracle knows which clusters contain the supplied true
   neighbors. An exact small dynamic program finds the minimum total whole-
   cluster population covering at least 9/10 and 10/10 IDs. These are optimistic
   *coverage* costs, not achievable SQ8 recall guarantees or equivalent to a
   population-level 95% recall target.
2. **Ranking waste:** compare oracle costs with the cluster-prefix populations
   required under centroid, support-bound, and witness scores. Prefix diagnostics
   scan whole clusters in order, without the timed search's skip-if-too-large
   behavior. Report means/medians/p95, not just mean cluster ranks.
3. **Local geometry:** use directions toward centroid-graph neighbors rather than
   globally shared PCA axes, with full-member support extrema.
4. **Actual representatives:** retain IDs of the points defining support extrema.
   Rank by their minimum true distance as a separate diagnostic. These are real
   points, so their nearest distance is an upper bound on a cluster's nearest
   member. It may still be a poor predictor. No representative query-time
   performance claim is made: those point evaluations occur outside timed
   search, and their count is explicitly recorded as `witness_evaluations`.

If oracle coverage itself is costly, finer partitions are the natural next
control. If coverage is cheap but rankings waste most scans, focus on ranking
and selective refinement. Even a good witness ranking must pay for its point
lookups before it can become a viable budgeted search configuration.

## Cheap local convex metadata

For edge i→j, set a=c_j-c_i. Scan **all decoded INT8 members** of cluster i,
storing the padded min/max of (x-c_i)·a and the member IDs attaining the padded
extrema. These slabs contain the cluster convex hull; they are not an exact
convex hull. At query time:

    (q-c_i)·a = (||q-c_i||² + ||c_j-c_i||² - ||q-c_j||²) / 2.

Thus full flat discovery reuses already-computed centroid distances. No new
1024D query projections or stored direction vectors are needed. Query-to-slab
gap squared / ||a||² is a lower bound on cluster distance. Take the maximum of
all edge bounds and the enclosing-ball bound. Numerical padding covers dot
products, norm/distance evaluation, and cancellation in the identity; coincident
centroids contribute zero. The conservative bound is the only new pruning rule.

Current implementation intentionally requires flat routing with every centroid
evaluated. Partial graph discovery cannot treat absent neighbor distances as
free; using supports during graph navigation would require a separate cached,
budgeted evaluation policy. It is not part of this first experiment.

Each edge stores three doubles plus two uint32 witness offsets: normally 32B.
At 8192 leaves × degree8 this is **~2.10 MB**, plus existing radii and graph.
This is additional metadata, not a total serving-RAM measurement. No vectors are
duplicated for witnesses; IDs reference the packed corpus. Their construction
is a separate `support_build_s` counter and is included in overall build time.

## Controlled experiment

Reuse verified NVMe INT8 files, original supplied ground truth, fixed4
hierarchical partitions (64×128), graph degree8, seed42, CPU2, Release flags.
No dataset download or copy is repeated. Locality-based metadata is built after
packing, scanning the owned INT8 corpus. Partitioning and topology remain
identical for all score variants within an index.

After remote build/CTest/end-to-end preflight, run:

- SIFT1M sanity: first50 calibration queries, 35K vector cap.
- Full Qwen 8,840,823×1024: 250 calibration / 750 disjoint held-out queries;
  caps35K / 176,816 / 442,041; exhaustive cluster discovery.
- Centroid-only control; radius hybrid/pure; support weights0/.25/.5/.75/1.
  Weight0 isolates pruning without changing centroid order.
- Select the fastest configurations reaching .95/.99/.995 calibration recall
  **independently for each bounds family**, then load held-out queries and run
  three repeats. Unreached targets retain explicit diagnostic labels. Do not
  choose weights/budgets using held-out results.
- First50 calibration queries additionally compare all support-extreme witnesses
  (deduplicated per cluster) with centroid/hybrid scores. This diagnostic can
  inspect up to16 actual members per leaf; it is not a claimed cheap search.

All variants share the combined build and metadata allocation; this does not
measure standalone per-variant build cost or RAM. Serving RSS, swap and query
major faults are recorded separately from peak build RSS. The report preserves
paging counters and does not infer runtime RAM from index payload size.

## Validation and outputs

Local Release, ASAN/UBSAN, and end-to-end fixture checks pass. Tests cover plane
separation, extreme query magnitudes, duplicate centroids, witness membership,
complete INT8 search agreement, vector caps, rejection of incomplete/graph
centroid caches, oracle coverage consistency, diagnostic output, bounds-family
selection, and the report generator. The remote job repeats correctness gates
before any full timing; remote checks for this new snapshot are still queued.

`study.py` runs the controlled sequence under the lock. `preflight.py source build`
exercises the native path. `summarize.py ROOT` validates CSVs, repeat completeness,
hard caps, memory-row alignment, sampled safety and oracle costs, then writes
`summary.json` and `REPORT.md` automatically. Source hashes are recorded after
remote build; input manifest is copied without including the large dataset.

Results for this experiment are in `REPORT.md` and `summary.json`. The prior
shared-geometry run completed independently; its artifacts are saved under
`../hartmut-geometry-sq8-20261108/results/`.
