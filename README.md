# TieredQ: Variance-Aware Bit Allocation

TieredQ is a new vector quantization algorithm that dynamically allocates bits per dimension, providing a very high recall compression ratio (compared to RaBitQ or TurboQuant). TieredQ centers and rotates vectors into a PCA basis, then greedily allocates 0–8 bits per dimension to reduce quantization error. Other ANN indexes can use the codec directly.

TieredQ core steps:
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
index = TieredQIndex(dimensions=100, code_bits=200, fast_mode=True)
index.add(vectors)  # first add trains; later adds only encode
scores, ids = index.search(vectors[:2], k=10)  # (2, 10) float32, uint32

index.save("index.tq")
index = TieredQIndex.load("index.tq")
```

- `fast_mode` is optional and defaults to `False` (FP32 scoring of quantized
  vectors). `True` trades some score/ranking accuracy for speed and stores only
  the fast representation. Both modes scan every vector; saving preserves the mode.
- The first `add()` needs a representative batch of at least two vectors.
  Inputs must be contiguous float32 NumPy arrays.
- Queries accept one vector or a matrix. Scores estimate inner products;
  larger is better. Normalize inputs for cosine similarity.
- Fast storage size depends on the model and CPU. Inspect `bytes_per_vector`
  for the representation stride and `storage_bytes` for allocated vector data.
  Fast scans use AVX-512 or AVX2 on x86, and NEON on ARM. Files loaded on
  another CPU may use a slower scalar backend and emit a warning.
- Matrix queries use OpenMP workers. Set `OMP_NUM_THREADS` before starting Python
  to control parallelism.

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

## Codec comparison benchmark

See [benchmark/README.md](benchmark/README.md) for the one-command CPU comparison
against official RaBitQ on SIFT1M, GloVe-200 and OpenAI3-1536, including measured
storage overhead, recall, throughput and latency.

Disclaimer: everything but the core idea and the README was created using both Opus 5.5 and GPT 6.1 Sol
