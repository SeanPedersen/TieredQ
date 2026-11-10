# Leaf assignment study: fewer scanned clusters at the same recall

Status: complete (exit 0); see `REPORT.md`. Reassignment cuts vectors at 95% from 177K to 70K at 8K leaves. 32K + P=8 reaches 95.2% at 35K vectors. Faiss IVF-SQ8 at the same 32K is as good or better (95.6% at 31K) and builds in 610 s vs 1,540 s. Our serving anon RSS is 9.5 GB vs Faiss 14.3 GB. P=all was stopped (projected ~75 min reassignment with the exact double kernel).
Remote root: `/home/sean/experiments/TieredQ-issue44-assignment-20261110` (`status.json`, `job.exitcode`).

## Motivation

At 8,192 lists on Qwen 8.84M×1024 (same SQ8 codes), Faiss IVF-SQ8 reaches 96.0% held-out recall with
64 clusters / 81.7K vectors and 99.0% with 316K. Our fixed hierarchical k-means needs 126.5 clusters /
176.8K vectors for 95.6% and never reaches 99% within 442K. Time per scanned vector is equal (0.15 µs).
The gap is therefore partition quality, not the scan kernel.

Hypothesis: each vector is assigned to the nearest fine centroid **inside its coarse parent**, but search
ranks all leaf centroids globally. Vectors near coarse boundaries sit in leaves whose centroid is not their
nearest, so queries near them probe the wrong leaves.

## Treatment

`--reassign-parents P`: after hierarchical training, reassign each vector to the nearest leaf centroid
among the leaves of its P nearest coarse parents (P = all gives global assignment), then recompute leaf
means. Two iterations. Parent envelopes are recomputed so their bounds still enclose their leaves. Assignment
is parallel and accumulation serial, so results are identical for any thread count. CTest checks complete search
against brute force after reassignment. A SIFT1M gate compares 1 and 32 build threads before Qwen runs.

| Model | Leaves | Reassignment |
|---|---:|---|
| ours-8k-p0 / p4 | 64×128 | none / 4 parents (P=all stopped: ~75 min reassignment with the exact double kernel) |
| ours-32k-p0 / p8 | 128×256 | none / 8 parents |
| faiss-8k / faiss-32k | 8,192 / 32,768 | flat k-means (4 iterations, 40 samples per list) |

All runs: same INT8 codes, 4 Lloyd iterations, 32 build threads, single-core pinned search. Calibration
on 250 queries selects the fastest setting reaching 95/99/99.5%; 750 held-out queries, one repeat (scan
counts are deterministic). Ours: flat routing (all centroids) and graph routing, 10 vector caps.

## Memory accounting

The earlier "peak build RSS 18.4 GB" mixed the memory-mapped input (file-backed, reclaimable page cache)
with allocated memory. Both harnesses now sample peak **anonymous** and **file-backed** RSS during the build
(`memory.json`), plus serving RSS after the input is released.

Output: `REPORT.md`, `summary.json` (regenerated after each model).
