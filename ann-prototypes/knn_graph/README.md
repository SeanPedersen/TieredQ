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

## Earlier dead ends (study.py, 100K)

- Quantized TieredQ IVF (2/4-bit) and nested IVF: 8.0-8.5 s at recall 0.94, of which 3.8-5.3 s is build. Sorting
  queries by list made no difference.
- Faiss IVF-Flat (per-query search): 8.7 s at 0.915.
- List-level probing (every row of a list uses the list's probe set): fast, but recall levels off at 0.86.
- One or two NN-descent rounds (numba) on top of an IVF graph: 5-8 s for +0.03-0.06 recall.
