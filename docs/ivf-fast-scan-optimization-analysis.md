# Closing the native IVF-RaBitQ speed gap

Analysis of TieredQ issue [#11](https://github.com/SeanPedersen/TieredQ/issues/11), the actual zvec IVF-RaBitQ scan, and profiles collected on hartmut on 2026-10-23. This is an optimization proposal, not a measured implementation speedup.

## Measured target

The existing 664-bit TieredQ index has 100 coded coordinates: 64 at seven bits and 36 at six bits. It therefore bypasses the compact byte-table cache, which requires every tier to have at most four bits. Its fast scan uses expanded signed-byte levels and AVX-512BW multiply-adds. The 216-bit model has 16 three-bit and 84 two-bit coordinates and does use the compact byte-table path.

| Larger index | Fast TieredQ, 1 core | IVF-RaBitQ, 1 core | Fast TieredQ, 18 threads | IVF-RaBitQ, 18 threads |
|---|---:|---:|---:|---:|
| Full probe | 64.9 | 179.0 | 649 | 2,289 |
| nprobe 64 | 229 | 561 | 2,272 | 6,894 |
| nprobe 16 | 836 | 1,658 | 8,239 | 17,392 |

Matching the existing RaBitQ configurations requires approximately 2.0–2.8x more single-core throughput and 2.1–3.5x more 18-thread throughput. Recall differs: full-probe fast TieredQ is 0.9724 versus RaBitQ 0.8732. The target must specify whether TieredQ must retain its current recall or may spend some of its recall advantage. Equal nprobe also does not mean identical candidate counts: at nprobe 64 TieredQ visits 341,368 rows/query versus 320,038 for RaBitQ.

The full benchmark and input controls are in [the benchmark report](ivf-glove100-fast-scan.md).

## What RaBitQ actually does

The native path in `zvec/src/core/algorithm/ivf_rabitq/ivf_rabitq_entity.cc`, `search_cluster_impl`, calls `split_batch_estdist` for 32 vectors. That calls the AVX-512 HACC accumulator in `rabitq/src/fastscan/fastscan_avx512.cpp`. It uses nibble extraction and byte shuffles over packed one-bit codes, retaining high-accuracy query LUT values. This measured path is not a popcount kernel.

For multibit codes the first pass produces an estimate and a lower-distance estimate. Only lanes whose lower-distance estimate beats the current heap threshold receive `split_distance_boosting`, which reads their extra bits. The unfiltered scan is separately instantiated with `HasFilter=false`. One-bit search rejects scores before reading keys or constructing heap entries.

By contrast, the TieredQ adapter computes the complete fast score for every vector in every selected list, materializes list scores, copies them to 256-score blocks, and then performs scalar candidate processing. It invokes a type-erased filter callback for every vector even when the context has no filter.

## Profile evidence

Single-core full-probe profiles were collected sequentially with `perf record -e cycles:u -F 199 -g`. Each process includes index loading/cache construction, one warmup, one timed pass and one evaluation pass over the same 500 queries. These are whole-process sampling proportions, not isolated search-stage wall times.

| Self cycles | TieredQ fast | RaBitQ |
|---|---:|---:|
| Expanded int8 scan | 54.90% | — |
| IVF search/result loop | 25.54% | 31.28% |
| Type-erased TieredQ filter callback | 13.57% | — |
| One-bit HACC accumulator | — | 33.61% |
| Batch estimator/correction | — | 25.50% |
| Extra-bit dot kernel | — | 4.26% |
| Fast cache construction | 2.25% | — |

The TieredQ result-loop bucket includes selection and other inlined operations; it must not be described as pure heap time. The extra-bit percentage does not establish an exact refinement rate. Instrument that rate before choosing a screening design.

At a 55% scan fraction, doubling only the kernel gives roughly 1.38x overall under Amdahl's law. Even eliminating that fraction entirely gives roughly 2.22x. These approximate ceilings explain why a faster dot instruction alone is insufficient for the full-probe target.

Raw profiles and text summaries are preserved in `~/experiments/issue6-parallel-rabitq-20261023/fast-mode/results/perf-{tieredq,rabitq}.{data,txt}` on hartmut.

## Recommended sequence

### 1. Remove avoidable integration work while preserving results

- Choose filtered/unfiltered execution once per query. Pass an empty predicate when `context.filter().is_valid()` is false; specialize the hot scan rather than calling `std::function` for every row.
- Compare a block of scores against the current top-k threshold using SIMD; enumerate only competitive lanes before loading keys and invoking real filters. Preserve the existing score/key tie ordering and refresh the threshold after insertion. A threshold snapshot may admit extra work, but must not exclude a valid candidate.
- Consume scores directly from scan blocks, avoiding the list-sized score buffer and subsequent copies. Keep scoring and selection as separate small components connected by a narrow block interface.
- Specialize the IVF epilogue: every vector in a list has the same centroid. Broadcast its two query terms once per list instead of executing two centroid gathers per 16-vector block.
- Give the wide fast path lightweight query preparation. It currently calls `codec.query`, which builds exact LUTs and grouped tables even though the int8 kernel needs only weights and centroid terms. Preserve exact coarse ranking if that is a requirement.

Benefits: preserves the current model and fast scoring, removes measured overhead, and makes subsequent kernel work easier to measure. Limit: cannot promise the entire 2–3.5x gap from these changes. Query preparation particularly matters at smaller nprobe, whereas the full scan dominates full probe.

### 2. Add progressive scoring with conservative rejection

Borrow RaBitQ's execution strategy rather than its quantizer. Use a cheap coarse representation of TieredQ's **reconstructed numerical levels**, followed by the existing int8 score or exact TieredQ score only for candidates that can still enter top-k.

A controlled first prototype is partial-coordinate scoring with a residual-norm bound. In the existing fast numerical domain, write the score as `b + a * dot(q, l)`, where `b` contains centroid/correction terms and `a` contains factor and scaling. After scoring coordinate subset A, the remaining contribution is bounded by `abs(a) * norm(q_B) * norm(l_B)`. Precompute document residual norms at a small number of stage boundaries. Reject only if the resulting upper score bound is strictly below the kth-best completed score; retain equality for key ties and handle rounding conservatively.

This starts with the already quantized numerical levels and full-precision quantized-query values, so it does not repeat the failed low-bit query experiment. A positive result can motivate a more compact one- or two-bit screening cache, storing per-document approximation error norms and keeping residual data cold until refinement.

Benefits: can avoid most full-dimensional scoring while preserving current fast results if bounds are valid and useful. Risks: worst-case norm bounds may be too loose; extra norm loads and branch/compaction costs may outweigh saved work. RaBitQ's probabilistic error estimates are not automatically valid bounds for TieredQ. Do not copy its formulas without deriving them for TieredQ.

Measure stage survival rate, bytes actually read, screening time and refinement time. Stop if most candidates survive. Measure preservation against the current fast top-k separately from recall against ground truth.

### 3. Optimize the surviving int8 kernel and batch data reuse

- Use multiple independent accumulators/document blocks to shorten the current 25-group accumulation dependency chain; compare generated assembly and throughput.
- For 18-thread batches, tile a small number of queries over resident list blocks so each level load serves several queries. Schedule tiles across the existing 18 workers, avoid nested OpenMP teams, cap scratch and register use, and preserve per-query probe membership. Cross-query reuse is an alternative to per-query pruning, so benchmark their combination rather than assuming additive gains.
- Treat bandwidth as a hypothesis until counters or a roofline measurement establish it. The expanded cache adds about 123 MiB resident memory for the larger index; it exceeds Hartmut's 24.8 MiB LLC, but that alone does not prove bandwidth saturation.

Hartmut is Skylake-X: it has AVX-512F/BW, but lacks AVX-512 VNNI, VBMI and VPOPCNTDQ. `vpdpbusd`, `vpermb`, or vector-popcount-only designs cannot close the gap on this host. Retain the current `vpmaddubsw`/`vpmaddwd` baseline; add VNNI as optional dispatch for other CPUs if measurements justify it. ARM dot-product availability likewise needs runtime/platform handling.

### 4. Explore an explicitly different recall target

A model capped at four bits per coordinate can use the compact byte-table cache instead of expanding 6/7-bit levels. A cheap first pass followed by larger-candidate exact TieredQ reranking is another option. Both can reduce recall and/or add rerank work; neither is an established iso-recall optimization. Training a new model is necessary to assess the first option.

If the actual target is RaBitQ's 0.8732 recall rather than TieredQ fast's 0.9724, tune nprobe and candidate count against that recall target before optimizing machine instructions. The existing p16 point is only 0.8490, so it cannot be claimed as an equal-recall win. Keep single-core and 18-thread measurements only.

## What issue #11 rules out

The issue reports 1/2-bit query popcount recall of 0.61/0.72 versus 0.936 exact, and only 0.878 with four query bits. It also records that raw Lloyd-Max code indices and uniform-level reinterpretation failed badly. Its useful pivot is int8 numerical levels and int8 query weights, which the current wide fast path already implements with Hartmut-compatible multiply-adds.

The next major gain should come from **doing fewer full scores and less candidate-processing work**, not repeating the original low-bit query popcount proposal. No unmeasured speedup above is a promise that TieredQ can match RaBitQ at TieredQ's higher recall.
