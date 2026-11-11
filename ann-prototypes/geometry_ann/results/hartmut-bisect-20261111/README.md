# Bisect of the Python end-to-end recall gap (#52)

Remote root: `/home/sean/experiments/TieredQ-issue52-bisect-20261111`. The driver is `defaults_qwen.cpp` from
`../hartmut-defaults-20261111/`, extended with `seed=`, `builder=` (`TieredQNestedIVFIndex::build`), `check_batch=`,
`round_trip=`, `tier_file=` and `internal_lists=`. 2-bit prefix, fast list training plus router, 8-bit fine tier, 20
candidates, 750 held-out queries alone on pinned core 2.

The first Python end-to-end builds (`../hartmut-pythondefaults-20261111/`) were 0.002-0.005 recall@10 below the C++
study drivers at every nprobe, for 1- and 2-bit prefixes alike.

## 1M rows, 8K lists (truth: exact L2 over the subset)

| Build | nprobe 128 / 256 / 512 |
|---|---|
| explicit, seed 42 | 0.9575 / 0.9736 / 0.9836 |
| explicit, seed 0 | 0.9531 / 0.9725 / 0.9843 |
| builder, seed 42 | 0.9552 / 0.9721 / 0.9832 |
| builder, seed 0 | 0.9580 / 0.9727 / 0.9840 |
| builder 0 + save/load round trip | 0.9580 / 0.9727 / 0.9840 |
| builder 0 + `write_tier()` file, direct reads | 0.9580 / 0.9727 / 0.9840 (all rows equal to `encode_tier()`) |

One-row `search_batch` (the Python path) returned `search()`'s ids in every sweep (0 mismatches). Seeds move recall by
up to 0.0045 at nprobe 128. None of these explains a systematic gap.

## 8.84M rows, 64K lists (dataset truth)

| Build | nprobe 96 / 128 / 256 / 1024 |
|---|---|
| explicit (`train_lists` on its own sample), seed 42 | 0.9436 / 0.9513 / 0.9685 / 0.9843 (= hartmut-ssdtier) |
| explicit, seed 0 | 0.9397 / 0.9525 / 0.9691 / 0.9848 |
| explicit, seed 7 | 0.9403 / 0.9496 / 0.9647 / 0.9816 |
| builder, seed 0 | 0.9380 / 0.9463 / 0.9636 / 0.9821 |
| explicit codec, lists trained inside `build()`, seed 0 | 0.9381 / 0.9464 / 0.9636 / 0.9819 |
| builder after the fix, seed 0 | 0.9401 / 0.9495 / 0.9647 / 0.9825 |
| builder after the fix, seed 42 | 0.9420 / 0.9513 / 0.9664 / 0.9837 |
| builder after the fix, seed 7 | 0.9413 / 0.9492 / 0.9649 / 0.9831 |

Cause: `TieredQIVFIndex::build` drew its list-training sample with `TrainingRng(seed)` and k-means then drew its
initial centroids from a fresh `TrainingRng(seed)`, so both consumed the same engine outputs. The nested builder's
codec sample had the same pattern with the codec's own seeded training. Every earlier study trained its lists through
`train_lists()` on a separately seeded sample, which is why none of them saw it. Fix: both samples draw from their own
salted seed streams (and the list sample reads rows in id order).

The fix moved the builder back into the seed spread. Before it, the builder at nprobe 128 (0.9463) sat below all six
other samples (lowest: 0.9492). After it, three builder seeds and three explicit seeds overlap: nprobe 128
0.9492-0.9513 against 0.9496-0.9525, 256 0.9647-0.9664 against 0.9647-0.9691, 1024 0.9825-0.9837 against
0.9816-0.9848. The partition seed alone moves recall@10 by up to 0.0044. That is wider than a +-0.002 match against
one reference partition can resolve: the shipping reference (seed 42) sits at the top of the spread.
