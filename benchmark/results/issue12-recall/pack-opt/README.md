# Issue #12: packed code storage and serving memory

Archived bytes/vector values use the earlier memory counters; see the
[final-review accounting correction](../../../../docs/high-bit-recall.md#final-review-accounting-correction).
Use the [refreshed current storage table](../storage-refresh/README.md) for the eleven reported configurations.

Historical threshold-7 results below. The approved [threshold-5 follow-up](../code5/README.md) supersedes the five affected budget points; raw curves are preserved.

Follow-up to the [scan optimization](../scan-opt/README.md). Fast caches for models with 7/8-bit tiers store codes at tier width instead of one byte per coordinate. Models no longer keep encoder-only state resident until the first append. Recall is identical in all 36 primary rows and fast scores stay bit-identical to the scalar codec. [Design and analysis](../../../../docs/high-bit-recall.md#packed-code-storage-and-serving-memory).

## Primary results

Full datasets, 1,000 held-out queries, top-10, seeds 0/1/2 for recall, seed 0 timing, **8 threads, batch 32**, `taskset -c 0-7`. "Baseline" is the delivered scan optimization, rerun in the same session from a separate workspace.

| Point | Recall@10 mean | Bytes/vector baseline → packed | QPS baseline → packed | RaBitQ bytes / QPS (recall) |
| --- | ---: | ---: | ---: | ---: |
| SIFT 4-bit | 0.93383 | 137.833 → **73.342** | 1,717 → 1,666 | 84.001 / 1,723 (0.84503) |
| GloVe 4-bit | 0.90153 | 121.580 → 112.877 | 11,067 → 10,946 | 148.012 / 12,607 (0.88183) |
| OpenAI 4-bit | 0.98387 | 1798.694 → **887.724** | 1,052 → 1,098 | 788.070 / 4,439 (0.96737) |
| SIFT 8-bit | 0.98853 | 141.859 → 137.346 | 1,537 → 1,415 | 148.001 / 1,644 (0.98820) |
| GloVe 8-bit | 0.99257 | 221.486 → 212.783 | 5,261 → 7,391 | 276.012 / 10,822 (0.98850) |
| OpenAI 8-bit | 0.99827 | 1798.814 → 1655.844 | 908 → 898 | 1556.070 / 3,997 (0.99670) |

- **What changed bytes.** 4-bit SIFT/OpenAI gain mainly from packing; the other points only from the serving-model trim. The 8-bit packed layout is byte-identical to the old one.
- **Noise.** RaBitQ moved by up to 8% between the two runs, and the GloVe 8-bit baseline is a low outlier. Batch-32 differences are within that noise.

## Single query (batch 1, seed 0, QPS at 1 / 8 threads)

| Point | Baseline | Packed | RaBitQ |
| --- | ---: | ---: | ---: |
| SIFT 4-bit | 60.1 / 390 | 43.6 / 307 | 232 / 230 |
| OpenAI 4-bit | 53.8 / 252 | 63.5 / 286 | 590 / 591 |
| GloVe 4-bit | 1,596 / 6,022 | 1,600 / 5,983 | 1,698 / 1,687 |
| SIFT 8-bit | 25.1 / 184 | 24.6 / 180 | 222 / 220 |
| GloVe 8-bit | 170 / 950 | 163 / 911 | 1,604 / 1,579 |
| OpenAI 8-bit | 21.3 / 112.7 | 21.0 / 111.2 | 543 / 544 |

SIFT 4-bit learns eight short tiers (8:2, 7:12, 6:16, 5:20, 4:22, 3:28, 2:18, 1:8). Its odd-width segments never fill a 32-rank compile-time period, so single queries pay variable-shift unpacking. OpenAI's halved bytes outweigh the unpacking.

## Budget curve (1–8 bits, 8 threads, batch 32)

`curve/` runs every budget for both codecs; `curve/summary.json` holds the equal-memory comparisons. Packed bytes and recall:

| Bits | SIFT | GloVe | OpenAI |
| ---: | ---: | ---: | ---: |
| 1 | 27.3 / 0.498 | 38.7 / 0.404 | 287.5 / 0.895 |
| 2 | 101.3 / 0.760 | 63.8 / 0.667 | 1,489.8 / 0.938 |
| 3 | 57.3 / 0.877 | 112.9 / 0.817 | 1,651.5 / 0.959 |
| 4 | 73.3 / 0.934 | 112.9 / 0.902 | 887.7 / 0.984 |
| 5 | 89.3 / 0.965 | 212.8 / 0.945 | 1,079.7 / 0.992 |
| 6 | 105.3 / 0.981 | 212.8 / 0.965 | 1,271.7 / 0.996 |
| 7 | 121.3 / 0.986 | 188.8 / 0.986 | 1,463.7 / 0.997 |
| 8 | 137.3 / 0.989 | 212.8 / 0.993 | 1,655.8 / 0.998 |

Points whose widest tier has 5–6 bits (SIFT 2, GloVe 5–6, OpenAI 2–3) still use x86 int8 numerical levels at one byte per coordinate, so the curve is not monotonic there. GloVe 3-bit uses the narrow lookup cache, whose padded layout is the same size as 4-bit. Learned tiers and the resident/serving model split for every point are in `microbench/tiers.json` (GloVe/OpenAI) and `microbench/tiers-sift.json`. The SIFT rows of `tiers.json` used the legacy lift and are superseded.

## Evidence

- **Runs.**
  - `primary/`, `single4/`, `single8/` and `curve/` come from the packed implementation; `primary/preprocessing.csv` includes stage counters (`profile.py --stages`).
  - `baseline-rerun/`, `baseline-single4/` and `baseline-single8/` are the same-session baseline.
  - `microbench/run.sh`/`run.log` drove the runs. The baseline profile step failed on a CMake cache copied into the baseline workspace and was rerun separately; `run3.sh`/`run3.log` resumed from there.
- **Synthetic A/B.** `microbench/ab1.log` (`ebench.cpp`, `build.sh`, `ab.sh`): result hashes match the baseline library for all five synthetic shapes. A 26:7/148:6/26:5 model drops from 312.0 to 240.4 B/vector at unchanged QPS.
- **Kernels.** `microbench/kbench.log` (`kbench.cpp`, `kbench.sh`) has interleaved single-core kernel costs on an SMT-idle core, including the two rejected runtime-offset variants (`new2`, `new3`; their headers are in the source archive). `kbench-smt-contended.log` was measured beside the benchmark on a shared physical core and is not used.
- **Sources.** `as-run-sources.tar.gz` holds the compiler inputs and harness (sha256 `a9200f3d10a9b6340422b55010817b6e63aaccc48aee1250dc27a38cd31d2780`).

## Validation

- **C++ (ARM, x86 Release, x86 ASan).** All three pass. New checks:
  - Hand-built tiers of every width 8→1 at odd bit offsets: every register-table period and phase, straddling codes and a partial block, against a reference reading the original rows (portable, AVX2, AVX-512 tile and single, NEON).
  - Packed bytes per vector.
  - A legacy byte-per-coordinate payload loads with identical search results, resaves byte-identical to a fresh index, and rejects a code above its tier width.
  - Lazy encoder: serving-only loaded models, identical encodes, shared state across copies.
- **Python.** 42 tests on ARM and x86. The corruption test now sets a bit past the packed code budget.
- **Benchmark validation.** 16 tests in both workspaces (`microbench/run.log`).

## Reproduction

```sh
cd benchmark
./validate.sh
taskset -c 0-7 ./run.sh --bits 4 8 --threads 8 --batches 32 --seeds 0 1 2 --output output-pack
taskset -c 0-7 uv run --locked --python 3.12 python profile.py output-pack --stages
taskset -c 0-7 ./run.sh --bits 8 --threads 1 8 --batches 1 --seeds 0 --output output-pack-single
taskset -c 0-7 ./run.sh --bits 4 --threads 1 8 --batches 1 --seeds 0 --output output-pack-single4
taskset -c 0-7 ./run.sh --bits 1 2 3 4 5 6 7 8 --threads 8 --batches 32 --seeds 0 1 2 --output output-pack-curve
```

Remote workspace: `sean@hartmut:/home/sean/experiments/TieredQ-issue12-recall` (`pack-opt/` holds the baseline workspace, microbenchmarks and drivers).
