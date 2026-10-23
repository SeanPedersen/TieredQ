# TieredQ — Variance-Aware Bit Allocation

Efficient vector quantization for C++ and Python, with a standalone exhaustive
in-memory index. Other indexes can use the codec directly.

TieredQ subtracts a learned centroid from each vector and rotates the residual
into a PCA basis. It allocates 0–8 bits per dimension greedily, giving each
extra bit to the coordinate where it reduces expected quantization error most.
This greedy allocation applies at every dimensionality. Query lookup tables
estimate inner products directly from the encoded vectors; larger scores are
better. Normalize inputs for cosine similarity.

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

## Optional fast search

`search()` uses the exact TieredQ FP32 scorer by default. To use the approximate
full-scan kernel, build its auxiliary cache explicitly:

```python
index.build_fast_scan()
print(index.has_fast_scan)  # False when the CPU has no supported fast kernel
scores, ids = index.search_fast(vectors[:2], k=10)
scores, ids = index.search(vectors[:2], k=10)  # exact remains available
```

Both methods accept one vector or a matrix and return the same shapes and dtypes.
Fast search trades some score/ranking accuracy and extra memory for speed; it still
scans every document without reranking. ARM uses int8 dot products. x86 AVX-512F/BW
uses byte lookups for narrow tiers and int8 dot products for wider tiers.

Without a cache, `search_fast()` falls back to exact search. A nonempty `add()`
invalidates the cache; call `build_fast_scan()` again to restore the fast path.
The cache is derived in memory and is not saved: rebuild it after `load()`.
Use `has_fast_scan` to check whether the fast cache is active. Performance and
ranking agreement depend on the data and hardware; see the
[measurements](docs/glove200-fast-scan.md).

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

## ARM64 packed scans

On AArch64, packed-plane scoring automatically uses NEON grouped FP32 table
lookups. Row scoring and partial packed blocks retain the scalar reference.
Code and model formats are unchanged. Grouped sums can differ slightly from
coordinate-by-coordinate FP32 accumulation; this backend adds no quantization.
See the [ARM benchmark results](https://github.com/SeanPedersen/TieredQ/issues/3#issuecomment-6355240929)
for reproduction and limitations.

## Batch search throughput

Python matrix queries use native batched search with a shared index and OpenMP
workers; backends can reuse scan work across queries. Single-vector calls remain
serial. Set `OMP_NUM_THREADS` before starting Python to control the worker limit;
the useful count depends on CPU and memory bandwidth. Avoid multiplying that
limit by application-level request concurrency. Batched results match individual
searches using the same exact or fast mode.
