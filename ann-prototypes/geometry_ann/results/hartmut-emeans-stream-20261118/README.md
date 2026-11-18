# #45: streamed e-means replaces Lloyd as the automatic list training (hartmut, 2026-11-18)

The previous session's e-means (64 rows per list x 2.5 epochs at 8.84M / 64K lists,
[hartmut-emeans64k-20261118](../hartmut-emeans64k-20261118/)) gained recall but built 15% slower at 7.7 GB more peak
RSS. Two changes fixed both:

- **Streaming.** `EmeansTrainer` reads every batch through a `TrainingRowSource` (the build's row loader, here the
  memory-mapped SQ8 store) in 32K-row chunks and accumulates per-list sums. Training holds the centroids, the batch
  and closing-pass sums and one chunk (1.6 GB at 64K x 1024), never a sample, so it can train on every row.
- **Grouped restriction.** `SuperRestriction::prepare` copies the centroids into super-member order once per batch.
  Before, every 256-row work item re-gathered its super's 1 MB of members (about 64 copies per super per batch).
  `emeans_probes` lets batch rows probe fewer supers than the final assignment (`assignment_probes`).

New defaults (`TieredQCoarseTraining::automatic`; the former Lloyd recipe is `TieredQCoarseTraining::lloyd`):
from 8,192 lists e-means 2 epochs over 256 rows per list (all rows at 8.84M / 64K), 256 supers, 4 probes per batch
row and 16 in the final assignment; below that, exact e-means 2.5 epochs over 256 rows per list.
`TieredQIVFFlatIndex` defaults to `trainer="emeans"`, 256 rows per list.

## Train-only harness: 8.84M x 1024, 64K lists, seed 0

`examples/coarse_training_benchmark.cpp` (`train64k.sh`, 32 threads): anonymous RSS while training (store pages
excluded), and the objective (mean squared distance) and imbalance (k * sum n^2 / N^2) of exact assignment of 131,072
held-out rows. The objective ranked the earlier full builds in the same order as their recall@10.

| Training | Rows | train_s | Anon RSS | Objective | Imbalance |
|---|---:|---:|---:|---:|---:|
| Lloyd fast (4 it x 40/list, 256 x 16) | 2.6M | 83.1 | 11.6 GB | 0.4568 | 1.943 |
| e-means 64/list x 2.5, 16 probes | 4.2M | 86.9 | 1.57 GB | 0.4422 | 1.850 |
| e-means 96/list x 2.5, 16 probes | 6.3M | 129.4 | 1.58 GB | 0.4373 | 1.808 |
| e-means all x 1.0, 16 / 8 / 4 probes | 8.84M | 73.3 / 44.0 / 29.0 | 1.59 GB | 0.4443 / 0.4452 / 0.4472 | 1.796 / 1.788 / 1.778 |
| e-means all x 1.25, 16 probes | 8.84M | 91.7 | 1.59 GB | 0.4397 | 1.800 |
| e-means all x 1.5, 8 / 4 probes | 8.84M | 65.0 / 42.4 | 1.59 GB | 0.4383 / 0.4401 | 1.777 / 1.770 |
| **e-means all x 2.0, 4 probes** | 8.84M | **55.7** | 1.58 GB | **0.4378** | **1.763** |

Assignment is about 90% of e-means' time (`train64k.profile`), so probes trade time almost linearly; more epochs at 4
probes beat more probes at fewer epochs.

## Full build: TieredQNestedIVFIndex defaults, 8.84M x 1024, 64K lists

`bench.sh` (`build.py`, `serve.py`): build on 32 threads from the SQ8 memmap, tier on NVMe, 750 held-out queries on
pinned CPU 2. Lloyd seed 0 is from the previous session's window; seed 7 ran in the same window as e-means.

| | Lloyd s0 | e-means s0 | Lloyd s7 | e-means s7 |
|---|---:|---:|---:|---:|
| Build (tier write excluded) | 202.9 s | **178.6 s** | 195.5 s | **174.5 s** |
| Training / final assignment | - | 62.5 / 54.8 s | 82.7 / 52.0 s | 62.2 / 51.6 s |
| Peak build RSS | 20.93 GB | **15.35 GB** | 21.11 GB | **15.39 GB** |
| Serving anon RSS | 1.582 GB | 1.582 GB | 1.588 GB | 1.589 GB |

Recall@10 @ p50:

| nprobe | Lloyd s0 | e-means s0 | Lloyd s7 | e-means s7 |
|---:|---|---|---|---|
| 96 | 0.9381 @ 2.27 ms | **0.9472** @ 2.29 | 0.9412 @ 2.28 | **0.9448** @ 2.33 |
| 128 | 0.9480 @ 2.55 | **0.9556** @ 2.54 | 0.9488 @ 2.55 | **0.9535** @ 2.58 |
| 192 | 0.9560 @ 3.03 | **0.9636** @ 3.06 | 0.9592 @ 3.06 | **0.9619** @ 3.10 |
| 256 | 0.9620 @ 3.51 | **0.9699** @ 3.56 | 0.9660 @ 3.58 | **0.9675** @ 3.61 |
| 512 | 0.9741 @ 5.33 | **0.9791** @ 5.40 | 0.9769 @ 5.39 | **0.9781** @ 5.55 |
| 1024 | 0.9804 @ 8.74 | **0.9836** @ 8.93 | 0.9836 @ 8.82 | **0.9852** @ 9.30 |

E-means is 0-5% slower per query at equal nprobe; interpolated to equal p50 it stays ahead at every nprobe on both
seeds (seed 7: 0.9448 vs 0.9426 at 2.33 ms, 0.9843 vs 0.9836 at 8.82 ms). All four acceptance criteria hold.

## 1M check: TieredQIVFIndex, MS MARCO 1M x 1024, 2 bits per dimension, router off, 8 threads

`run-1m.sh` (`ann-prototypes/knn_graph/ivf_lists_check.py`, `lloyd-auto` = `TieredQCoarseTraining.lloyd`, `auto` = the
new default), seeds 0 and 1 in one window. Lloyd seed 0 at 8,192 lists trained in 32.8 s against 8.2 s on seed 1 (a
host outlier); its recall is unaffected.

| Lists | Training | Build | Train | Recall@10 by nprobe, seed 0 / seed 1 |
|---|---|---:|---:|---|
| 8,192 | Lloyd | 31.7 s (s1) | 8.2 s | 16-256: 0.815/0.853/0.882/0.903/0.914 ; 0.823/0.857/0.886/0.903/0.916 |
| 8,192 | e-means | 29.2-29.3 s | 6.1 s | 0.823/0.860/0.887/0.904/0.915 ; 0.829/0.860/0.886/0.904/0.915 |
| 2,048 | Lloyd | 35.5-35.8 s | 12.5-12.7 s | 8-128: 0.803/0.852/0.883/0.903/0.916 ; 0.802/0.855/0.886/0.906/0.918 |
| 2,048 | e-means | 29.3-29.5 s | 6.7 s | 0.820/0.859/0.890/0.908/0.918 ; 0.809/0.857/0.886/0.907/0.917 |

E-means builds 8-18% faster with no training sample in RAM, at equal or higher recall per nprobe. Its probed lists
hold 3-11% more rows, so at equal rows scanned the highest nprobes are 0.001-0.002 lower, within the 0.002 seed spread.
