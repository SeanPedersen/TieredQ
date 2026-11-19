# #45: SuperEMeans vs TieredQ C++ e-means (hartmut, 2026-11-18)

SuperEMeans (https://www.lkuffo.com/superemeans/) = e-means + SuperKMeans dimension pruning, from
`cwida/SuperKMeans` at `bb51014` (the commit the post used), run in `lkuffo/reproducibility-superemeans` at `394a37b`.
`tq_emeans.cpp` puts TieredQ's `EmeansTrainer` (source at `dd4999d`) into the same harness: the same rows, the same exact
fp32 final assignment (`SuperKMeans::Assign`), WCSS and IVF recall evaluator. `harness.patch` adds the dataset, `THREADS`
and peak RSS (VmHWM reset before training).

Data: the first 2M Qwen/MS MARCO rows (fp32 1024-d from the source HDF5), 1,000 queries (750 test + 250 calibration),
exact top-100 (`prepare.py`). k = 8,192 (244 rows per list), 2 epochs, 32 threads, seed 42, one seed.
All implementations resolve the same e-means recipe: batch 65,536 rows, initial step 0.3798.

| | Train | Peak RSS (train overhead over the 7.6 GB of data) | WCSS | R@100 @ 24K vectors | R@10 @ 24K |
|---|---:|---:|---:|---:|---:|
| e-means (SuperKMeans, BLAS only) | 69.0 s | 15.8 GB (+8.2) | 1.0570e6 | 0.9226 | 0.9567 |
| **SuperEMeans** (pruning) | 43.8 s | 15.8 GB (+8.2) | 1.0571e6 | 0.9226 | 0.9566 |
| TieredQ e-means, exact | 48.7 s | 8.05 GB (+0.41) | 1.0567e6 | 0.9224 | 0.9598 |
| **TieredQ e-means, 256 supers x 4 probes** (default from 8,192 lists) | **6.4 s** | 8.05 GB (+0.41) | 1.0620e6 | 0.9200 | 0.9580 |

Recall at matched vectors scanned (`matched.py logs/*.log`, log-interpolated):

| vectors scanned | 5K | 12K | 24K | 60K |
|---|---:|---:|---:|---:|
| SuperEMeans R@100 | 0.8161 | 0.8833 | 0.9226 | 0.9597 |
| TieredQ exact R@100 | 0.8145 | 0.8829 | 0.9224 | 0.9590 |
| TieredQ 256x4 R@100 | 0.8125 | 0.8810 | 0.9200 | 0.9575 |

- The post's claim holds here: pruning is 1.58x faster than the same library's e-means (post: 1.59-1.78x at 2 epochs).
- TieredQ's exact e-means is within 11% of SuperEMeans without pruning, and needs 20x less extra RAM: SuperKMeans keeps
  a second (rotated) copy of the dataset while training; ours streams 32K-row chunks.
- TieredQ's default restricted assignment trains 6.8x faster than SuperEMeans; R@100 is 0.002-0.004 lower at matched
  scan and R@10 is equal or higher (one seed; the partition seed spread at 8.84M is up to 0.0045).
- Porting the pruning would not pay: it needs rotated rows (a full copy, or a d x d rotation of every streamed row per
  pass: 2 d^2 = 2.1 MFLOP per row, about 3x the restricted assignment's 2 d (256 supers + 128 candidates) = 0.8 MFLOP), and with 4/256 supers each row scores only about
  128 candidate centroids, too few for dimension pruning to skip much.
