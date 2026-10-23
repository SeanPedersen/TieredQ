# TieredQ: Variance-Aware Bit Allocation

TieredQ is a new vector quantization algorithm that dynamically allocates bits per dimension instead of uniformly like RaBitQ or TurboQuant. TieredQ centers and rotates vectors into a PCA basis, then greedily allocates 0–8 bits per dimension to reduce quantization error. Other ANN indexes can use the
codec directly.

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
  Files loaded on another CPU may use a slower scalar backend and emit a warning.
- Matrix queries use OpenMP workers. Set `OMP_NUM_THREADS` before starting Python
  to control parallelism.

See [fast-mode measurements](docs/glove200-fast-scan.md) for accuracy and speed.

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
