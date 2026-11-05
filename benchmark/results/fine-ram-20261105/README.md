# Fine-tier serving RAM: shared prefix-view model and one-shot fine-row load (2026-11-05)

The #4 precheck ([issue4-precheck-20261105](../issue4-precheck-20261105/README.md)) found that fine-tier and exact indexes
with fine ranks used 6-16% more steady RAM after reload than the same budget with fine tiers off, at equal disk bytes.
At equal steady RAM, that made fine tiers a tie on OpenAI at 4 bits/dim.

## Causes (100K, after reload)

| cause | OpenAI-1536 | Gemini-768 | scales with |
|---|---|---|---|
| `TieredQ::prefix_view` deep-copied the serving model (half-precision basis + centroids) through a serialize/deserialize round trip | 6.3 MB | 2.0 MB | constant |
| glibc kept that round trip's freed temporaries resident (`malloc_trim(0)` released about 50 B/vector) | ~5 MB | ~2 MB | mostly constant |
| `read_fast_payload` read the fine rows at exact size and then appended the read slack: a realloc copy of every fine row, at 2× capacity | load peak +116 B/vector at 8 bits/dim | +71 B | n |
| `storage_bytes` did not count the view, and counted the 2× capacity | accounting only | | |

## Change

- **Shared model arrays:** `TieredQ` holds its basis, half-precision basis and centroids as
  `tieredq_detail::SharedArray` (`include/tieredq/detail/tieredq_shared_array.h`). These are immutable arrays shared by
  copies of a model. `train_model` builds them as plain local vectors and sets them once.
- **Prefix view:** `prefix_view` copies the model (so it shares those arrays), keeps only the persisted config, and
  rebuilds derived state with `validate_model()` and `prepare_serving()`, as a load does.
- **Fine-row load:** reserves the rows plus slack before reading. This applies to the flat and IVF loaders alike.
- **Accounting:** `storage_bytes()` counts the view's own state through `TieredQ::memory_bytes_beyond(codec_)`.
- **Regression test** (`test_tieredq_fast_fine.cpp`): a reloaded fine-tier index allocates no more than the built
  one, and an empty fine-tier index (whose storage is the view alone) holds less than one basis. The test fails if the
  old deep copy is restored.

## A/B on hartmut

- **Builds:** baseline `fef7b75` vs candidate `fef7b75` + `candidate.patch`. Data, hosts and harness are those of the
  precheck.
- **Tests on the candidate** (GCC 16, AVX-512): the C++ suite passes (`cand-ctest.log`) and so do the benchmark tests
  (`cand-pytest.log`). The Python bindings suite (134 passed, 3 skipped) and the C++ suite (Clang, NEON) also passed on
  the Mac.
- **Memory** (`memory-{base,cand}/memory.csv`): measure_memory quick cells for default, off and exact at 1/2/4/8
  bits/dim, seed 0, on CPUs 8-31.
- **Timing** (`cells/`, `cells-quick/`), seed 0, alone on CPUs 0-7:
  - OpenAI 4 bits/dim: all three variants, full ABBA, k=10 and k=100, at 8t·b32, 8t·b1000, 1t·b1 and 8t·b1.
  - The other changed paths: default at OpenAI 8-bit and Gemini 4/8-bit, and exact at 8-bit on both. One pair each,
    k=10, at 8t·b32, 1t·b1 and 8t·b1. This shortened pass replaced the rest of the ABBA, to get results faster.
- Full tables: [summary.md](summary.md).

## Result

Steady RSS after reload, bytes/vector, baseline → candidate:

| | default 4-bit | default 8-bit | exact 8-bit | off 4 / 8-bit (unchanged) |
|---|---|---|---|---|
| OpenAI 100K | 983.6 → 857.0 (−12.9%) | 1,755.8 → 1,628.7 (−7.2%) | 1,737.9 → 1,617.5 (−6.9%) | 852 / 1,622 |
| Gemini 99K | 464.8 → 424.7 (−8.6%) | 869.7 → 811.3 (−6.7%) | 856.5 → 806.0 (−5.9%) | 422 / 809 |

- **Steady RAM:** the default and exact now match off.
- **Load peak at 8 bits/dim:** OpenAI default 1,880 → 1,636 (−13%), exact 1,870 → 1,621, Gemini default 945 → 817.
- **Accounted RAM:** now tracks RSS (OpenAI default 8-bit: 1,613 accounted vs 1,629 RSS).
- **Results:** every memory cell returns the same hit ids on both trees, and every timed R@10 is identical.
- **Unchanged:** 1-2 bits/dim, off, and exact at 4 bits/dim, which have no fine ranks.
- **Timing:** every changed path is within ±4% of baseline on QPS and p50, mostly 0.98-1.02×. The unchanged off path
  reads ±5% on the same host, so these differences are noise, not regressions.
- **Expected at 1M (not measured):** steady RAM −~11 B/vector, since the constant costs spread over 10× the rows. Load
  peak at 8 bits/dim −~125 B/vector, since the fine-row realloc grows with n.
