# All-rows k-NN graphs for dimensionality reduction (#56)

Can one of our IVF variants build the k-NN graph that UMAP, t-SNE and NCVis need (every row's k nearest other rows)
faster than HNSW or NN-descent? Each time below is the full cost: index build plus searching every row.

## Answer

**Update (PCA + NN-descent + exact rerank, `methods.pca_nndescent`):** project the rows to 256 PCA dimensions (a
32K-row sample), run pynndescent there for C = 30-45 neighbours, then rerank those candidates in full dimension. On
MS MARCO 1M x 1024 this beats tuned pynndescent at matched recall@15, on both seeds:

| recall@15 | PCA-256 + NN-descent + rerank | pynndescent (tuned) | speedup |
|---|---|---|---|
| ~0.953 | **29.2 / 29.4 s** (C = 30: 0.953 / 0.953) | 44.1 / 43.6 s (t32 i30 c60: 0.954) | 1.5x |
| ~0.975 | **40.1 s** (C = 40: 0.975, seed 0) | ~51 s (interpolated between 0.954 and 0.979) | 1.27x |
| ~0.98 | **47.0 / 48.0 s** (C = 45: 0.981 / 0.981) | 52.2 / 52.3 s (n30: 0.979) | 1.1x |

At 1M, PCA takes 1.7 s and the rerank 2.2-3.3 s; the rest is NN-descent in 256-d. 192 dimensions lose too many true
neighbours (0.911 at C = 30), and 128 lose more. On DBpedia 100K x 512 it only ties pynndescent
(PCA-256, C = 30: 2.30 s at 0.968; C = 45: 3.74 s at 0.990, against 3.81 s at 0.989), since distances there cost less
of NN-descent's time. Seeding pynndescent from an IVF-Flat graph (`ivf_init_nndescent`, `tree_init=False`) did not
help: 1.70 s at 0.921 at 100K, and the descent broke down from a one-probe start (0.50 at 1M). PCA candidates from
IVF-Flat instead of NN-descent level off at 0.85 (100K) and 0.77 (1M).

The original question, about IVF on its own:

- **Against HNSW: yes.** The library's unquantized `TieredQIVFFlatIndex.knn_graph` (list-major self-join) is 2.5-3x
  faster than hnswlib at equal recall on DBpedia 100K.
- **Against pynndescent (UMAP's default): no.** It loses 1.15-1.7x at 100K and about 7x at 1M. IVF scores every
  row of every probed list. Reaching recall@15 0.93 takes about 6% of all rows per row at 100K, and that share does
  not fall fast enough as N grows. NN-descent computes far fewer distances. A faster GEMM kernel cannot close a 7x gap:
  the join already runs at a third of the CPU's AVX-512 peak at 100K.
- **The TieredQ codec is the wrong tool for this job:** training alone took 3.8 s at 100K, more than pynndescent's
  whole run.

## Setup

hartmut (i9-7980XE, AVX-512), 8 threads pinned to cores 0-7, k = 15, a fresh process per run.
- `dbpedia100k`: OpenAI text-embedding-3-small 512-d, 100K rows, L2-normalized (the projection-benchmark set).
  Recall is exact over all rows.
- `msmarco1m`: a seeded 1M-row subset of Qwen/MS MARCO 1024-d, decoded from the SQ8 store (`prepare_msmarco1m.py`).
  Recall is measured on 10,000 seeded rows, each ranked against all 1M rows.

`compare.py` is the acceptance driver; `study.py` is the earlier exploration (quantized TieredQ IVF/nested, Faiss
IVF-Flat, list-level cluster join, NN-descent refinement).

## Results: DBpedia 100K (seeds 0 / 1)

| recall@15 | IVF-Flat knn_graph | pynndescent | hnswlib |
|---|---|---|---|
| ~0.93 | 1.97 / 1.96 s (256 lists, p16: 0.925 / 0.923) | **1.48 / 1.47 s** (t16 i20 c30: 0.932) | |
| ~0.94 | 2.40 / 2.35 s (512 lists, p32: 0.940) | **2.04 / 2.07 s** (t32 i30 c60: 0.946) | 5.94 s (M8 ef64: 0.942) |
| ~0.98 | 4.39 / 4.32 s (128 lists, p32: 0.982 / 0.981) | **2.52 / 2.62 s** (n30: 0.979) | 12.8 s (M16 ef32: 0.980) |
| ~0.99 | 5.69 / 5.49 s (512 lists, p128: 0.989) | **3.81 / 3.71 s** (n30 t32 i30 c60: 0.989) | |

The two seeds agree within 0.008 recall and 5% time. IVF-Flat build (k-means plus assignment) takes 0.2-0.7 s; the
rest is the join. Before the C++ version, the numpy prototype (`methods.ivf_join`) reached 2.09 s at the 0.927 point.

## Results: MS MARCO 1M (seed 0, stopped early)

| | time | recall@15 |
|---|---|---|
| pynndescent (default) | **26.0 s** | 0.918 |
| pynndescent t16 i20 c30 | 30.0 s | 0.940 |
| pynndescent t32 i30 c60 | 44.1 s | 0.954 |
| pynndescent n30 | 52.2 s | 0.979 |
| pynndescent n30 t32 i30 c60 | 75.1 s | 0.988 |
| IVF-Flat 512 lists, p8 | 172.2 s (join 169.7 s) | 0.868 |

The run was stopped after this point. Larger probe settings cost proportionally more and could not catch up.

## SQ8 row storage, list tiling, and Rust k-NN graph libraries (2026-11-18)

`TieredQIVFFlatIndex(storage="sq8")` keeps rows as int8 codes spanning each dimension's [min, max] and decodes each
probed list into an FP32 tile before the sgemm. The same pass capped the per-sgemm list tile at 512 KiB (128 rows at
1,024-d) on both storages; before, each probed list was one sgemm. Compared with rinnd 0.4.4 (lmcinnes, Rust
NN-descent) and manifolds-rs 0.7.0 (NN-descent and IVF via ann-search-rs), each at its defaults and one or two
higher-recall settings, Rayon pinned to the same 8 threads. Results: `results/hartmut-ivfflat-sq8-20261118/`
(`*-libs.jsonl` is the final pass, `*-tile.jsonl` the tile A/B, the plain files the untiled first pass).

**Tiling.** At 1M x 1024 and 512 lists / 8 probes the FP32 join fell from 170 s to 101 s (same recall); SQ8 from
211 s (1,024-row tiles) to 83 s. At 100K the lists are small enough that tiling changes nothing.

**SQ8 vs FP32** (same window, SQ8 directly after its FP32 twin; time is build plus join):

| set, lists / probes | FP32 | SQ8 | rows FP32 / SQ8 |
|---|---|---|---|
| DBpedia 100K, 128 / 32 (seeds 0 / 1) | 4.53 / 4.32 s, 0.982 / 0.981 | 4.36 / 4.32 s, 0.975 / 0.975 | 197 / 50 MB |
| DBpedia 100K, 256 / 16 | 1.91 / 1.98 s, 0.925 / 0.923 | 1.76 / 1.88 s, 0.921 / 0.919 | 197 / 50 MB |
| DBpedia 100K, 512 / 32 | 2.33 / 2.35 s, 0.940 / 0.940 | 2.30 / 2.28 s, 0.936 / 0.935 | 198 / 51 MB |
| MS MARCO 1M, 512 / 8 | 103.5 s, 0.868 | **86.0 s**, 0.866 | 3,917 / 987 MB |
| MS MARCO 1M, 1024 / 16 | 105.3 s, 0.908 | **93.4 s**, 0.905 | 3,920 / 990 MB |
| MS MARCO 1M, 1024 / 32 | 176.6 s, 0.946 | **162.2 s**, 0.942 | 3,920 / 990 MB |

SQ8 is a quarter of the row bytes and as fast at 100K, 8-17% faster at 1M (a 4 GB FP32 base streams from DRAM; SQ8
reads a quarter of it), for 0.002-0.007 recall@15. The loss is the full-range min/max codec re-encoding rows; an SQ8
store built with `build(codes=...)` keeps its codes and loses nothing beyond the store's own quantization.

**Against the Rust libraries** (wall seconds, recall@15):

| recall@15 | TieredQ IVF-Flat | pynndescent | rinnd | manifolds-rs |
|---|---|---|---|---|
| 100K ~0.91 | | **1.21 / 1.24 s** (0.906) | **1.22 / 1.22 s** (0.907) | |
| 100K ~0.92 | SQ8 256/16: 1.76 / 1.88 s (0.921) | | | NN-descent 2.85 s (0.929), IVF 3.1 s (0.928) |
| 100K ~0.96 | | | | IVF 2x probes 4.2 s (0.964) |
| 100K ~0.98 | FP32 128/32: 4.4 s (0.982), SQ8 4.3 s (0.975) | **2.5 s** (n30: 0.979) | 2.6 s (n30: 0.979) | IVF 4x probes 6.0 s (0.985) |
| 1M ~0.92 | SQ8 1024/16: 93.4 s (0.905) | **25.6 s** (0.918) | 27.9-29.8 s (0.921-0.923) | |
| 1M ~0.95 | SQ8 1024/32: 162 s (0.942) | 30.5 s (t16: 0.940) | | NN-descent 85 s (0.955), IVF 169 s (0.953) |
| 1M ~0.98 | | | **54.8 s** (n30: 0.979) | |

- rinnd matches pynndescent within about 10% at both sizes (no speedup over it here, 8 threads, AVX-512).
- TieredQ IVF-Flat beats manifolds-rs IVF 1.35-1.7x at 100K, but at 1M they are level in time and manifolds-rs is
  about 0.01 ahead in recall (its e-means partition at 1,000 lists against our 10-iteration k-means at 1,024).
- NN-descent (pynndescent, rinnd) still wins: 1.7x at 100K / 0.98 and 3.6-5x at 1M. IVF-Flat scores every row of
  the probed lists; NN-descent only scores neighbours of neighbours. PCA-256 + NN-descent + rerank (above) remains
  the configuration that beats pynndescent.

## Earlier dead ends (study.py, 100K)

- Quantized TieredQ IVF (2/4-bit) and nested IVF: 8.0-8.5 s at recall 0.94, of which 3.8-5.3 s is build. Sorting
  queries by list made no difference.
- Faiss IVF-Flat (per-query search): 8.7 s at 0.915.
- List-level probing (every row of a list uses the list's probe set): fast, but recall levels off at 0.86.
- One or two NN-descent rounds (numba) on top of an IVF graph: 5-8 s for +0.03-0.06 recall.
