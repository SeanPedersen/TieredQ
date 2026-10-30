# TieredQ: Variance-Aware Bit Allocation

TieredQ is a new vector quantization algorithm that dynamically allocates bits per dimension, providing higher recall compression ratio (compared to RaBitQ or TurboQuant). TieredQ centers and rotates vectors into a PCA basis, then greedily allocates 0–8 bits per dimension to reduce quantization error. Existing ANN indexes (HNSW / IVF) can use the codec directly as drop-in for RaBitQ or TurboQuant.

TieredQ core algorithm:
1. PCA
2. Variance-aware greedy bit allocation per dimension (principal component)
3. Scalar quantization per dimension with allocated bit budget

## Install

Requires C++20, CMake, LAPACK/BLAS and OpenMP. On macOS, install `libomp` with
Homebrew; Accelerate provides LAPACK. On Ubuntu, install `liblapack-dev`,
`libblas-dev` and OpenMP support. Repository access is required; there is no
PyPI release.

```sh
uv add "tieredq @ git+ssh://git@github.com/SeanPedersen/TieredQ.git"
```

## Python

```python
import gensim.downloader as api
from tieredq import TieredQIndex

vectors = api.load("glove-wiki-gigaword-100").vectors  # (400000, 100), float32
index = TieredQIndex(dimensions=100, code_bits=200) # 2 bits per dim
index.add(vectors)  # first add trains; later adds only encode
scores, ids = index.search(vectors[:2], k=10)  # (2, 10) float32, uint32

index.save("index.tq")
index = TieredQIndex.load("index.tq")
```

- `metric="ip"` (inner product) is the default; `metric="cosine"` normalizes rows and
  queries, and `metric="l2"` serves Euclidean search with scores of `-|q - x|^2`.
  L2 lifts vectors into one extra dimension with a lift whitened for the query
  distribution, stored in the model, so saved indexes stay self-contained.
- Bit allocation follows the metric: inner-product and cosine models allocate
  on PCA variance^1.5 (recall-optimal when queries are distributed like the data,
  +0.5 to +0.8 R@10 points at equal disk on DBpedia OpenAI/Gemini, GloVe neutral to
  +1.3, issue #9) in both modes; L2 models use variance (MSE), which is optimal for
  whitened queries.
  `allocation_power=` overrides it.
- `mode="fast"` is the default and stores the native fast search representation.
  Use `mode="full"` for compact packed rows and FP32 scoring of quantized vectors,
  with slower search. Both modes scan every vector; saving and loading preserve
  the selected representation. Read the selection from `index.mode`.
  The former `fast_mode` boolean argument/property has been removed; use `mode`.
- In C++, use `dv::TieredQIndex(codec)` for fast mode or
  `dv::TieredQIndex(codec, dv::TieredQIndexMode::full)` for full mode.
  Inspect the enum with `index.mode()`; the boolean constructor is removed.
- The first `add()` needs a representative batch of at least two vectors.
  Inputs must be contiguous float32 NumPy arrays.
- Queries accept one vector or a matrix. Scores estimate inner products.
- Fast storage size depends on the model and CPU. Inspect `bytes_per_vector`
  for the representation stride and `storage_bytes` for allocated vector data.
  Fast scans use AVX-512 or AVX2+F16C on x86, and NEON on ARM.
  On AVX-512, single-vector queries with compatible packed-code models use a
  range checkpoint after four tiers to skip uncompetitive suffixes while preserving
  fast-scan IDs and score bits. Bounds are prepared per query and add no stored bits
  per vector. Other layouts, backends, and multi-vector queries use the full scanner.
- Matrix queries use OpenMP workers. Set `OMP_NUM_THREADS` before starting Python
  to control parallelism.
- `mode` also steers training. Fast models whose tiers all fit 4 bits use an additive
  3-bit codebook, so the x86 nibble lookup stores 3-bit tiers at exactly 3 bits.
  This adds +1.2 to +3.6 R@10 points on GloVe-200 at 2.25–3.9 bits/dim, at equal
  storage and speed. It is neutral on SIFT, and DBpedia OpenAI/Gemini at 2–4 bits/dim
  are unchanged. Full mode keeps the Lloyd-Max 3-bit codebook, and either model
  serves in either mode. See [issue #20 results](benchmark/results/issue20-additive-20261029/README.md).

See [fast-mode measurements](docs/glove200-fast-scan.md) for accuracy and speed.
Models with tiers wider than 4 bits now retain original codebook precision in fast mode.
This improves recall at a throughput cost; existing numerical-level fast indexes
need rebuilding to recover that precision. Their fast storage keeps codes at tier
width; files saved with the earlier one-byte-per-coordinate layout still load and are
repacked. See the [high-bit investigation](docs/high-bit-recall.md).

## C++

```sh
cmake -S . -B build
cmake --build build -j
OMP_NUM_THREADS=2 ctest --test-dir build --output-on-failure
./build/tieredq_example
cmake --install build --prefix /desired/prefix
```

Include `<tieredq/tieredq.h>`, use `find_package(TieredQ CONFIG REQUIRED)`, and
link `TieredQ::TieredQ`. FetchContent also works. See [integration.md](integration.md)
for layouts, threading, ownership and dependency setup.

### IVF (C++)

`dv::TieredQIVFIndex` is a fast-mode inverted-file index. With `lists = 0` the codec's k-means centroids
(at most 256) are the lists, at no extra storage. Any other count trains a separate coarse quantizer, whose
centroids are stored.

```cpp
dv::TieredQIVFIndex index(codec, /*lists=*/1024);
index.build(rows, count);
auto hits = index.search(query, /*k=*/10, /*nprobe=*/16);
// Opt-in approximate pruning for models with more than four tiers (about 2+ bits/dim):
auto fast = index.search(query, 10, 16, dv::TieredQPruning{/*confidence=*/8, /*prefix_tiers=*/4});
```

- **Pruning.** By default the scan prunes exactly: hits equal an unpruned scan. A positive `confidence`
  bounds the remaining tiers statistically instead, which is faster and approximate. On OpenAI 100k, confidence 8
  kept recall unchanged (±0.0001) at 1.7× the QPS at 4 bits/dim. With `prefix_tiers = 3` it gave 1.6–2.4× the QPS
  for 0.001–0.003 recall at 2 and 4 bits/dim. Confidence 4 or less loses clearly more recall.
- **Bit cap.** At about 2 bits/dim or less, training with `TieredQConfig::max_bits = 4` keeps every tier on the
  nibble lookup kernel. That gave 1.7× RaBitQ's IVF QPS at equal recall and memory (OpenAI 100k), for about
  0.006 recall at full probe. Don't use it at 4 bits/dim, where it loses recall.
- **Results** against RaBitQ's IVF at equal measured memory are in
  [issue #19](https://github.com/SeanPedersen/TieredQ/issues/19) and the
  [issue #6 eval](benchmark/experiments/issue6_ivf/ivf_eval.cpp).

## Codec comparison benchmark

See [benchmark/README.md](benchmark/README.md) for the one-command CPU comparison
against official RaBitQ on SIFT1M, GloVe-200 and OpenAI3-1536, including measured
storage overhead, recall, throughput and latency.

Disclaimer: everything but the core idea and the README was created using both Opus 5.5 and GPT 6.1 Sol
