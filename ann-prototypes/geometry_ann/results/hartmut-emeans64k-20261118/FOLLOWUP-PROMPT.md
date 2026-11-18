# Follow-up session prompt: e-means for TieredQ IVF at 8.84M / 64K lists (#45)

Work on #45 again. Make native e-means coarse training beat the automatic fast Lloyd recipe in `TieredQIVFIndex` /
`TieredQNestedIVFIndex` on Qwen/MS MARCO 8.84M x 1024 at 64K lists, the #52 shipping low-RAM build. Fix both
problems the last session found: **more peak build RAM** and a **slower build**. Evaluate on hartmut.

## Acceptance criteria (all on 2 partition seeds, 0 and 7)

Compare against the automatic fast Lloyd recipe in the same build path:
- **Index build time** must be ≤ Lloyd's. Lloyd took 202.9 s (`TieredQNestedIVFIndex.build`, 32 threads, tier write
  excluded).
- **Peak build RSS** must be ≤ Lloyd's 20.93 GB.
- **Held-out recall@10 at equal p50 latency** must be ≥ Lloyd's, at nprobe 96-1024 (750 test queries, one pinned
  core).
- **Serving RAM** must stay at Lloyd's level, about 1.58 GB anonymous RSS.

Partition seed alone moves recall@10 by up to 0.0045 at 8.84M, so read recall gains through both seeds. Do not
change any default until this is met. Lloyd stays an option either way.

## How to operate (user feedback from the last session)

- Validate first with small, fast, train-only checks before any long study. Never queue slow reference
  implementations (the Faiss Python e-means is 7x slower than ours). Give ETAs.
- All heavy runs go on hartmut (`ssh sean@hartmut`) under the flock `/home/sean/experiments/.tieredq-issue44-benchmark.lock`.
  Builds use 32 threads; serving is pinned to CPU 2 with 1 OpenMP thread. Only local builds of single test targets
  with `nice cmake --build -j 2`.
- systemd-oomd kills the whole session under memory pressure (62 GB host, about 44 GB available). Estimate peak RAM
  before every 8.84M run.

## State of the code (uncommitted working tree on main, on top of the uncommitted #56 SQ8/tiling work)

- `src/tieredq_train_emeans.h` is a port of Faiss contrib `NumpyEmeans` (commit fc1095a):
  - `emeans_recipe()` resolves the batch law, step law, cosine schedule, epoch layout and revival threshold. It
    matches Faiss exactly (unit test).
  - `EmeansTrainer` trains with exact batch assignment through `nearest_centroids()`, or restricted assignment
    through `SuperRestriction`.
  - `SuperRestriction` assigns each batch row to the nearest member of its probed supers. `regroup` runs k-means of
    the centroids. `follow` moves each super to its members' mean after every batch. `reassign` regroups every 5
    batches. Member centroids are gathered and scored with FP32 BLAS, using single-threaded sgemms per (super, block)
    work item.
  - `train_coarse(sample, rows, d, lists, seed, TieredQCoarseTraining)` dispatches to e-means when
    `emeans_epochs > 0` and to Lloyd otherwise. It moved here from `tieredq_ivf_assign.h`.
- `src/tieredq_ivf_assign.h`: `nearest_centroids()` (FP32 BLAS argmax) is factored out of `BlasAssigner`.
- `include/tieredq/tieredq.h`: `TieredQCoarseTraining::emeans_epochs` (0 = Lloyd) is the 5th field, validated in
  `src/tieredq_ivf.cpp`. `automatic()` is unchanged: fast Lloyd `{4, 256, 16, 40}` from 8,192 lists, exact Lloyd
  `{20, 0, 16, 64}` below.
- `TieredQIVFFlatIndex` has `trainer` (lloyd | emeans), `epochs`, `set_centroids`, `clear_centroids` and
  `centroids()`. Its defaults are still Lloyd.
- Python bindings:
  - `TieredQCoarseTraining(..., emeans_epochs=)`.
  - `TieredQIVFIndex(..., seed=)`, `set_coarse_training(..., emeans_epochs=)` and `.build_stats` (`train_s`,
    `assign_s`, `encode_s`, `router_s`).
  - `TieredQNestedIVFIndex.build(..., seed=)`.
  - `TieredQIVFFlatIndex(trainer=, epochs=)`.
- Tests:
  - `tests/test_tieredq_emeans.cpp`: recipe against Faiss values, quality on blobs, restricted e-means within 3% of
    exact on grouped blobs, IVF-Flat and `TieredQIVFIndex` with e-means.
  - `tests/python/test_ivf_flat.py`: given centroids and the emeans trainer.
  - The full C++ suite and the Python tests pass.
- Study scripts:
  - `ann-prototypes/knn_graph/`: `partition_study.py`, `summarize_partitions.py` (`--cost join_s|scanned`) and
    `ivf_lists_check.py`, a 1M `TieredQIVFIndex` check with native e-means through `set_coarse_training`.
  - `ann-prototypes/geometry_ann/results/hartmut-emeans64k-20261118/`: `build.py`, `serve.py`, `study.py`, the 8.84M
    nested bench.
  - Remote: `/home/sean/experiments/TieredQ-issue45-emeans64k-20261118/` (results/<model>-seed<s>/{build,serve}) and
    `/home/sean/experiments/TieredQ-issue45-emeans-ivfflat-20261118/` (the 1M studies, data/msmarco1m.npy,
    results/msmarco1m-queries/query_truth.npz).
  - Local mirrors: `ann-prototypes/knn_graph/results/hartmut-partitions-20261118/`.

## Results so far

**IVF-Flat (`TieredQIVFFlatIndex`), MS MARCO 1M x 1024, 1,024 lists, 2 seeds.** E-means 256 rows/list x 2.5 epochs
(exact assignment) trains in 1.66 s against 2.06 s for default Lloyd (64 x 10). It gains +0.015 / +0.008 / +0.004
knn_graph recall@15 at equal join time, and +0.018 to +0.003 query recall@10 at equal rows scanned. It matches Lloyd
256 x 10 (6.8 s). Its sample is 4x Lloyd's (1.07 vs 0.27 GB).

**`TieredQIVFIndex` 1M, 2 bits per dimension, router off, MS MARCO queries, 2 seeds**, Δ recall@10 at equal rows
scanned, from fewest to most:

| Lists | Training | Train | Build | Δ recall |
|---|---|---:|---:|---|
| 2,048 | exact Lloyd 64 x 20 | 12.5 s | 35.5 s | 0 |
| 2,048 | e-means 256 x 2.5, exact | 5.9 s | 28.7 s | +0.011 / +0.004 / +0.002 / +0.002 / -0.001 |
| 2,048 | e-means 64 x 2.5, exact | 1.6 s | 24.6 s | - / -0.008 / -0.005 / -0.003 / -0.004 |
| 8,192 | fast Lloyd 40 x 4, restricted 256 x 16 | 8.0 s | 31.5 s | 0 |
| 8,192 | e-means 256 x 2.5, restricted | 11.8 s | 35.1 s | +0.013 / +0.008 / +0.003 / +0.001 / -0.001 |
| 8,192 | e-means 64 x 2.5, restricted | 6.3 s | 29.4 s | +0.004 / +0.002 / -0.001 / -0.001 / -0.002 |

**8.84M nested, 64K lists, seed 0** (the seed 7 rows and e-means 96 x 2.5 may have landed in the remote results):

| | Lloyd fast | E-means 64 x 2.5, restricted |
|---|---|---|
| Index build | 202.9 s | 232.8 s (+15%) |
| Peak build RSS | 20.93 GB | 28.64 GB (+7.7 GB) |
| Recall@10 @ p50, nprobe 96 | 0.9381 @ 2.27 ms | 0.9416 @ 2.31 ms |
| nprobe 256 | 0.9620 @ 3.51 ms | 0.9671 @ 3.60 ms |
| nprobe 1024 | 0.9804 @ 8.74 ms | 0.9823 @ 9.10 ms |
| Serving RAM | 1.58 GB | 1.58 GB |

The user rejected this trade: more build RAM and a slower build for a modest recall gain.

## Diagnosis and leads

1. **RAM.** Both trainers hold an FP32 sample of max(32,768, sample_rows_per_list x lists) rows:
   - Lloyd at 40 rows/list x 64K lists: 2.6M rows, 10.7 GB.
   - E-means at 64/list: 4.2M rows, 17.2 GB. At 96/list: 25.8 GB. 256/list would be all 8.84M rows, 36 GB.

   E-means is mini-batch, so it never needs the sample in RAM. Stream batches instead:
   - keep only the sampled row ids, or stream the whole SQ8 store;
   - decode or lift each batch of B rows from the source (`TieredQIVFIndex` builds read rows through a loader,
     `rows(ids, count, out)` and `coarse_rows(...)` in `src/tieredq_ivf.cpp`; nested builds use an SQ8 RowLoader);
   - hold only the centroids, the closing-pass sums (each lists x d x 4 B, 0.27 GB at 64K x 1024), the batch, and
     the restriction state.

   Peak training RAM then drops below Lloyd's, and the sample can grow (more rows per list, or one pass over all
   rows) at no RAM cost. Check which build-path buffers dominate the remaining 10 GB of the 20.9 GB peak.
2. **Build time.** Measure first. Nested builds don't report the training phase, so expose or log `build_stats` at
   8.84M. Or time `EmeansTrainer` alone on a 64K-list sample: a train-only harness on hartmut that loads N rows from
   the SQ8 store, runs fast Lloyd vs e-means with the same restricted assignment, and reports train_s and objective.
   Minutes, not a full build. Candidates:
   - Per batch, `SuperRestriction::assign` gathers every probed super's member centroids into a fresh buffer and runs
     many small single-threaded FP32 sgemms. Fast Lloyd's `RestrictedAssigner` scores binary16 panels with the
     `half_matrix` kernel instead.
   - Remedies: keep the centroids physically grouped by super (permute at `reassign`) so there's no per-batch gather;
     reuse the binary16 panel kernel; check work-item balance across 32 threads.
   - `follow()` costs O(lists x d) per batch and `reassign()` O(lists x supers x d) every 5 batches.
   - The recipe at n=4.2M, k=64K, T=2.5 gives batch 262,144 and 40 steps. Check whether fewer, larger steps or
     another epoch budget (1-2 passes over more streamed rows) keeps the quality.
   - E-means 64 x 2.5 makes the same row passes as Lloyd 40 x 4 (160 per list), so equal FLOPs should mean equal
     time. The gap is implementation overhead.
3. **Quality.** At 1M, e-means only beat Lloyd clearly with 256 rows/list; 64/list tied. Streaming should let
   e-means use many more rows at equal RAM and time. That's the path to a real recall gain at 64K lists.

## Suggested order

1. Read this file, the code above, and the remote results of the 8.84M bench (seed 7, 96 x 2.5). Note each run's
   peak RSS.
2. Write a train-only timing and RAM harness (C++ or Python through `TieredQIVFIndex.build_stats`) for 64K lists on
   hartmut. Profile e-means against fast Lloyd.
3. Implement streaming e-means batches from the row source, and remove the restricted-assignment overhead. Unit
   tests first, locally on 2 threads.
4. Quick 1M check with `ivf_lists_check.py` (2,048 and 8,192 lists, 2 seeds, about 10 min), then the 8.84M nested
   bench (`study.py`, 2 seeds) against the acceptance criteria above.
5. If accepted: make e-means the automatic default (`TieredQCoarseTraining::automatic` and `TieredQIVFFlatIndex`),
   keep Lloyd as an option, update the docs/README tables and CLAUDE.md, and draft the #45 comment for the user to
   approve before posting.
