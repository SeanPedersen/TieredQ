# SAQ IVF reference on the #55 arxiv partition (faiss#5940)

Answer to faiss#5940 ("how is it different from SAQ, arxiv:2509.12086?"). Remote root
`/home/sean/experiments/TieredQ-issue55-saq-20261119` on hartmut, 2026-11-19, under the shared lock. SAQ upstream
2163ebc (`SAQ_COMMIT`, with the 2026-10-27 study's benchmark patches) driven by `saq_ivf.cpp`; `study.py` runs it.
TieredQ was not rerun: its numbers come from `../hartmut-faiss-plugin-20261116` (same partition, seed 1).

## Setup

- **Same partition.** The #55 acceptance binary (3623bea) rebuilt arxiv-nomic-768 (1.34M, normalized) partition seed 1:
  Faiss k-means, 2,048 lists. `export_assignment` wrote TieredQ's row assignment; SAQ gets those centroids and that
  assignment.
- **SAQ as upstream runs it.** Full-corpus PCA (fp64 covariance, all 768 dims, as upstream `pca.py`), then
  `IVF::construct` at 1/2/3/4 average bits (default `QuantizeConfig`: segmentation on, CAQ), 32 threads. Search is
  upstream `IVF::search` (variance-pruned scan, default `SearcherConfig`, L2 = IP ranking on normalized data), with the
  query's PCA included in the timing. Coarse ranking is SAQ's fp32 flat scan (TieredQ / plugin use fp16; about 90 us
  per query on this data, see #55).
- **Serving** as #55: fresh process, RAM = RssAnon growth across load after `malloc_trim`, one thread on CPU 2, QPS
  best of 3 passes, recall@100 id overlap over 1,000 queries, nprobe 8-256.

## Results (seed 1, recall@100)

| index | RAM GiB | nprobe 16 recall / QPS | nprobe 64 | nprobe 256 (ceiling) |
|---|---|---|---|---|
| SAQ 1 bit | 0.204 | 0.797 / 1,419 | 0.842 / 837 | 0.850 / 381 |
| SAQ 2 bits | 0.336 | 0.868 / 1,339 | 0.937 / 766 | 0.951 / 341 |
| SAQ 3 bits | 0.448 | 0.879 / 1,355 | 0.960 / 773 | 0.976 / 345 |
| SAQ 4 bits | 0.610 | 0.882 / 1,256 | 0.967 / 709 | 0.986 / 311 |
| TieredQ plugin 1.25 bits (bisect) | 0.169 | | | ≥ 0.908 |
| TieredQ plugin 1.75 bits | 0.230 | 0.862 / 2,186 | 0.930 / 944 | 0.943 / 290 |
| TieredQ plugin 2 bits | 0.261 | 0.867 / 2,088 | 0.938 / 862 | 0.951 / 262 |
| TieredQ plugin 2.5 bits | 0.320 | 0.876 / 1,878 | 0.954 / 763 | 0.969 / 224 |
| TieredQ plugin 3 bits | 0.381 | 0.879 / 1,721 | 0.959 / 662 | 0.976 / 194 |
| TieredQ plugin 4 bits | 0.503 | 0.882 / 1,468 | 0.965 / 540 | 0.983 / 154 |

SAQ's bit plans (PCA dims → bits): 1 bit `64×5, 192×2, 512×0`; 2 bits `64×6, 128×4, 256×2, 320×0`; 3 bits
`64×7, 192×5, 256×3, 256×0`; 4 bits `64×9, 192×6, 192×4, 192×2, 128×0`.

- **Same idea.** SAQ also splits a bit budget unevenly over PCA dimensions (to 0 bits on the tail).
- **SAQ's nominal bits are not its stored bits.** Its code bits per dimension are 0.92 / 1.83 / 2.83 / 3.75 at
  nominal 1 / 2 / 3 / 4, but it stores 160 / 264 / 352 / 481 bytes per row (index file / rows): about 85 bytes of ids
  and per-row factors beside the codes. The plugin stores about 208 bytes per row at 2 bits (192 B codes). At equal
  nominal bits the recall curves coincide within 0.003 (4 bits: SAQ 0.986, TieredQ 0.983), while SAQ uses 21-26% more
  RAM (0.336 / 0.448 / 0.610 against 0.261 / 0.381 / 0.503 GiB).
- **At equal RAM TieredQ has higher recall.** SAQ 1 bit (0.204 GiB) reaches 0.850; TieredQ 1.25 bits (0.169 GiB)
  reaches ≥ 0.908. SAQ 2 bits (0.336 GiB) 0.951; TieredQ 2.5 bits (0.320 GiB) 0.969. SAQ 3 bits (0.448 GiB) 0.976;
  TieredQ between 3 bits (0.381, 0.976) and 4 bits (0.503, 0.983). The gap closes toward 4 bits; above recall 0.983
  TieredQ is unmeasured.
- **Speed depends on nprobe.** At equal bits TieredQ is 1.2-1.6× faster at nprobe 16; at nprobe 64 it is 1.13×
  faster at 2 bits but 0.86× / 0.76× at 3 / 4 bits; SAQ's variance pruning makes it 1.3-2× faster at nprobe 256. At recall 0.95: SAQ 3 bits (0.448 GiB) about 890 QPS (nprobe 48), plugin 2.5
  bits (0.320 GiB) 763 QPS (nprobe 64). At 0.97: SAQ 3 bits 536 QPS (nprobe 128, 0.448 GiB), plugin 3 bits 368 (0.381
  GiB). SAQ is faster at high recall, TieredQ is smaller.
- **TieredQ builds 1.7-2.3x faster** (32 threads, shared k-means of about 3 s excluded from both):

  | bits (SAQ / TieredQ) | SAQ construct | SAQ + PCA | TieredQ codec train + build |
  |---|---|---|---|
  | 1 / 1.75 | 33.5 s | 54.5 s | 0.9 + 12.2 = 13.1 s |
  | 2 / 2 | 30.2 s | 51.6 s | 0.9 + 12.8 = 13.8 s |
  | 3 / 3 | 30.1 s | 50.9 s | 1.0 + 16.4 = 17.4 s |
  | 4 / 4 | 35.6 s | 56.4 s | 1.0 + 20.3 = 21.3 s |

  SAQ's construct takes the row assignment as input (upstream computes it beforehand in `python/ivf.py` with Faiss
  k-means over all rows); TieredQ's build includes assigning every row. SAQ's PCA is this driver's (Eigen, fp64
  covariance over all rows; upstream uses Faiss `PCAMatrix` in Python); TieredQ trains its codec, rotation included,
  on 32,768 rows. SAQ's construct splits into about 9 s of setup and bit planning and 21-26 s of quantization.
  TieredQ times are from the #55 builds of the same partition.
- Single seed, one dataset (arxiv); Qwen 8.84M at 65K lists is not run (SAQ's IVF has no HNSW coarse quantizer
  above 20,000 lists).
