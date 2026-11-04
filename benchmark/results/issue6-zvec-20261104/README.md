# Issue #6: TieredQ's own IVF against zvec's IVF_RABITQ, both as shipped (2026-11-04)

Produced by [ivf_vs_zvec.py](../../experiments/issue6_zvec/ivf_vs_zvec.py) ([job.sh](../../experiments/issue6_zvec/job.sh), [job_nested.sh](../../experiments/issue6_zvec/job_nested.sh))
on Hartmut (i9-7980XE, AVX-512, GCC 16.2.1), TieredQ at `SOURCE_REVISION` (27babf3) through its Python bindings, zvec
0.8.0 from PyPI. Summary: [`summary.py`](../../experiments/issue6_zvec/summary.py) on `zvec.tsv tieredq.tsv nested.tsv` prints `tables.md`.

- **Data:** DBpedia Gemini-768 (99,000 base, 1,000 held-out document queries, inner-product top-10 truth).
- **zvec:** `IvfRabitqIndexParam(metric=IP, nlist=256, total_bits=B)` collection, insert + `optimize()`, queried with
  `IvfRabitqQueryParam(nprobe)` (no refiner). Its own k-means and its shipped encoding (`faster_config`).
- **TieredQ:** `TieredQIVFIndex(code_bits, lists=256)` (exact pruning; `tieredq-z8` = `TieredQPruning(confidence=8)`), and
  `TieredQNestedIVFIndex.build(prefix_bits, total_bits, lists=256, tier_rows="suffix")` with the tier in RAM
  (`nested-pP-cC` = P bits/dim prefix, C candidates rescored). The tier is in RAM here, so nested is charged
  prefix + tier as RAM. Its intended deployment keeps only the prefix IVF in RAM and the tier on SSD (#22); that is
  not measured here (TODO below).
- **Index RAM:** zvec = its RaBitQ IVF file (`*.qindex.*`); TieredQ = `storage_bytes` (+ tier for nested). zvec also keeps
  a full FP32 copy (`embedding.index.1.proxima`, 307.5 MB = 24,850 bits/vector) that these queries never read; it is
  not counted.
- **Timing:** one pinned core (CPU 17, SMT sibling idle), one query per Python call, 100 warmup queries, then one pass
  over all 1,000. Both pay Python call overhead; zvec also pays its collection layer (~260-300 µs/query floor at nprobe 4).

## At equal recall: fastest config, and least-RAM config (bits/vector, QPS 1 thread × 1 query)

| R@10 | zvec fastest | TieredQ fastest | Speed | zvec least RAM | TieredQ least RAM | RAM | Speed |
|---|---|---|---:|---|---|---:|---:|
| 0.80 | 3b np4: 2,721 / 3,676 | IVF 1.16b np8: 1,084 / 5,810 | 1.58x | 2b np8: 1,953 / 3,522 | IVF 1.16b np8: 1,084 / 5,810 | 0.56x | 1.65x |
| 0.90 | 5b np16: 4,259 / 2,938 | IVF 3.16b np16: 2,684 / 2,709 | 0.92x | 3b np32: 2,721 / 2,477 | IVF 2.16b np32: 1,879 / 2,405 | 0.69x | 0.97x |
| 0.95 | 5b np32: 4,259 / 2,549 | nested p1 c20 7b np32: 5,591 / 2,094 | 0.82x | 4b np128: 3,490 / 1,143 | nested p2 c20 3b np64: 2,498 / 1,219 | 0.72x | 1.07x |
| 0.97 | 6b np64: 5,027 / 1,958 | nested p1 c50 5b np64: 4,007 / 1,565 | 0.80x | 5b np256: 4,259 / 854 | nested p1 c50 4b np128: 3,215 / 1,144 | 0.75x | 1.34x |
| 0.98 | 7b np64: 5,794 / 1,816 | nested p1 c50 7b np64: 5,591 / 1,531 | 0.84x | 6b np128: 5,027 / 1,330 | nested p2 c20 4b np256: 3,282 / 513 | 0.65x | 0.39x |
| 0.99 | 7b np256: 5,794 / 751 | nested p1 c50 7b np128: 5,591 / 1,109 | 1.48x | 7b np256: 5,794 / 751 | nested p2 c20 5b np256: 4,082 / 520 | 0.70x | 0.69x |

## Findings
- **RAM and recall: TieredQ wins everywhere.** At every recall target TieredQ needs 28-44% less index RAM. At equal RAM
  it has equal or higher recall at every nprobe. The gap is large at low bits: full probe at 1 bit is 0.860 vs 0.741,
  at 2 bits 0.941 vs 0.862, at 3 bits 0.969 vs 0.924. zvec's 1-2 bit RaBitQ tops out below 0.87.
- **Speed at equal recall: split.** TieredQ is faster at R@10 ≤ 0.85 (1.6x, flat IVF at 1-2 bits) and at 0.99 (1.5x,
  nested). For 0.90-0.98, zvec's fastest config is 1.1-1.25x faster than TieredQ's, but it uses 4,259-5,794 bits/vector,
  more RAM than TieredQ needs for the same recall.
- At R@10 0.98, nested p1 c50 at 5 bits (4,007 bits/vector, nprobe 128) runs 1,168 QPS. zvec's least-RAM config
  for that target (6 bits, 5,027 bits/vector) runs 1,330 QPS: 0.80x the RAM at 0.88x the speed.
- **Speed at equal RAM: TieredQ ahead up to ~4 bits/dim; zvec slightly ahead at 6-7.** At ~3,200-3,500 bits,
  nested (c50, nprobe 64) reaches 0.967 at 1,572 QPS. zvec 4-bit reaches only 0.955 even at full probe (660 QPS).
  At ~5,600-5,800 bits and R@10 0.982, zvec runs 1,816 QPS vs nested's 1,531.
- **Flat `TieredQIVFIndex` above 3 bits/dim is the wrong engine.** It scans every bit of every probed row, so QPS falls
  17-25x from nprobe 4 to 256 (zvec: ~4x). Its 7-bit full probe runs 104 QPS vs zvec's 751. The nested prefix + refinement
  index fixes this; it is the RaBitQ-like two-stage design. Above ~3 bits/dim use nested. `confidence=8` pruning
  helps the flat index only at high bits and high nprobe (1.3-1.5x).

## Caveats
- One seed, one host, 99k rows, single-thread single-query only (no batched or multi-thread runs). Each cell was
  timed in one pass, not repeated: treat differences under ~10% as noise.
- Lists differ: each engine trains its own 256 k-means lists, as shipped.
- zvec as shipped uses RaBitQ-Library's `faster_config` encoding. Accurate encoding (RaBitQ's own IVF, measured in
  `issue6-ivf-20261030`) has higher recall at a given bit width.

## TODO: nested with the tier on SSD
Re-run the nested cells with `write_tier` + `attach_tier(path, reads="direct", queue_depth=32)`. Report prefix RAM
(`storage_bytes`) and tier SSD bytes as separate columns, with QPS at 1 thread. Expected: RAM drops to the prefix IVF
(roughly 1,000-1,700 bits/vector, not logged separately here) against zvec's 4,259-5,794 for R@10 0.95-0.99. QPS
falls by the cost of 20-50 direct reads per query.
