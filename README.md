# TieredQ — Variance-Aware Bit Allocation

Efficient vector quantization for C++ and Python, with a standalone exhaustive
in-memory index. Other indexes can use the codec directly.

TieredQ subtracts a learned centroid from each vector and rotates the residual
into a PCA basis. It normally allocates 0–8 bits per dimension greedily, giving
each extra bit to the coordinate where it reduces expected quantization error
most. The preserved 512-dimension, 512-bit configuration instead uses a fixed
layout of 64×4, 64×2 and 128×1 bits. Query lookup tables estimate inner products
directly from the encoded vectors; larger scores are better. Normalize inputs
for cosine similarity.

## Install

Requires C++20, CMake, LAPACK/BLAS and OpenMP. On macOS: brew install libomp
(Accelerate provides LAPACK). On Ubuntu: apt install liblapack-dev libblas-dev.
Repository access is required; there is no PyPI release.

Add Python bindings to your project using uv:

    uv add "tieredq @ git+ssh://git@github.com/SeanPedersen/TieredQ.git"

```python
import gensim.downloader as api
from tieredq import TieredQIndex

vectors = api.load("glove-wiki-gigaword-100").vectors   # shape (400000, 100), float32

index = TieredQIndex(dimensions=100, code_bits=200)   # 2 bits per dimension
index.add(vectors)   # first add trains the codec; later adds only encode

scores, ids = index.search(vectors[:2], k=10)   # (2, 10) float32, uint32
index.save("index.tq")
index = TieredQIndex.load("index.tq")
```

The first `add()` trains the codec on that batch and stores it, so pass a
representative batch first; it needs at least 2 rows. The trainer uses
`min(256, rows // 2)` centroids, where `rows` is that first batch's vector
count. Later `add()` calls only encode. Queries may be one vector or a
`(queries, dimensions)` matrix. Scores are inner products, so normalize inputs
for cosine similarity.

Inputs must be contiguous float32 NumPy arrays; use np.ascontiguousarray when
needed. Outputs own their memory. The Python package uses NumPy for arrays;
the C++ library has no Python or NumPy dependency.

## C++

    cmake -S . -B build
    cmake --build build -j
    OMP_NUM_THREADS=2 ctest --test-dir build --output-on-failure
    ./build/tieredq_example
    cmake --install build --prefix /desired/prefix

Include <tieredq/tieredq.h>, use find_package(TieredQ CONFIG REQUIRED), and link
TieredQ::TieredQ. FetchContent works too. The example trains, searches, persists
an index and reports timing. See [integration.md](integration.md) for layouts,
threading, ownership and dependency setup.
