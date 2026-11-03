# Issue #24: the nested refinement's per-query suffix projection (2026-11-03)

OpenAI 100k, d = 1536, 64 lists, 768-bit prefix (476 ranks) + `refinement:4`, k' 50, nprobe 32, seed 0. The finer model
codes 1,530 ranks, so a query is projected onto 1,054 suffix-only ranks (runs of 59 x 5, 367 x 4, 347 x 3, 246 x 2,
35 x 1 bits, `refinement-runs.txt`). Its int8 panels are 1.6 MB, which does not fit this CPU's L2 (i9-7980XE, 1 MB
per core). Baseline (`f3f7db6`): 119 us per query rescoring vs INT8 23 us. perf puts about 64% of that in the
projection; most of the rest is the per-candidate kernel (about 0.75 us x 50).

## Codec routes: no-go

`benchmark/experiments/issue24_projection/tail_sim.py` approximates the tail code offline (k-means residuals, PCA,
same tail bits) and compares the tail score error <q, t - t_hat>^2 on each query's top 50 (`tail-sim.txt`):

| Tail scheme | Tail MSE vs current | Query cost |
|---|---:|---|
| current PCA runs (rotated per run, Lloyd per run) | 1.00x | d x 1,054 |
| Hadamard-like rotation of the exact tail subspace, uniform bits | 1.63x | about d x 476 (dense) |
| same rotation over all d coordinates (O(d log d), noise leaks into prefix directions) | 20x | d log d |
| dense >= 4-bit runs + leaky rotation for the rest | 18x | d x 426 + d log d |
| cap at 768 / 512 / 256 tail ranks (same bits re-allocated) | 5x / 32x / 154x | proportional |

- An exact tail basis has to annihilate the prefix span, so the query cost floor is about d x prefix ranks, not
  d log d. The O(d log d) rotation skips that step and its error is 20x.
- The tail's bit spread (5 down to 1 bits) means a uniform code loses the per-rank allocation. The spectrum is also
  flat enough that dropping ranks is expensive.

## Kernels

`kernels.cpp` (one core): at 1,054 ranks the per-query projection is L3-bandwidth-bound. Float 59 us, int16 madd 55,
int8 maddubs 54 (the 7-bit query would also cost precision). A batch of 4 float queries sharing each decoded panel:
26 us per query. `tile.cpp`: tiles of 8 queries x 2 panels with the block held in L2 give 27 us per query, bit-identical to
the per-query kernel. Without VNNI on this CPU, integer batches cost about as many ALU ops per MAC as FP32 FMA.

## Change

`TieredQRefinement::query(vectors, rows, prefix_queries)`: batched query terms, equal to per-row calls exactly. Tiles of
up to 8 queries x 2 panels; tasks of 4 panels x 8 queries run on OpenMP workers (68 tasks for a batch of 32).
`TieredQNestedIVFIndex::search_batch` uses it. The projected mean is summed in 16 fixed partial chains on every path.
The per-query kernel is unchanged.

## Results

`rescore-profile.tsv` (rescoring alone on fixed candidates, one thread, `RESCORE_BATCH=32`) and `nested.tsv` (engine
QPS, CPUs 0-7). Old = `f3f7db6`, new = this change, 3 interleaved rounds, medians. The host was shared (two services
and another session's test runs), so 1t single-query cells vary by about 15%.

| Cell | Old | New |
|---|---:|---:|
| INT8 rescoring, us/query | 23.2 | 23.1 |
| Refinement rescoring, per query, us | 118.6 | 114.8 |
| Refinement rescoring, batches of 32, us/query | - | 73.4 (3.2x INT8, was 5.1x) |
| Nested engine nprobe 16, QPS 8t b32 | 17,044 | 19,006 (+12%) |
| Nested engine nprobe 32, QPS 8t b32 | 12,754 | 13,526 (+6%) |

Recall is unchanged in every cell (0.9765 at nprobe 32, 0.9606 at 16).

At 8 threads the batched projection takes about 200 us of a batch of 32's 2.4 ms. The IVF scan takes about 1.8 ms.
The issue's target (within about 2x INT8 per query at unchanged recall) is not reached. Even with no projection at
all, scoring 50 candidates and the per-query overhead are about 45 us, already 1.9x INT8. Further gains have to come
from the candidate kernel or from a CPU whose L2 holds the panels (or VNNI / AMX for integer batches).
