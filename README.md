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

## Exact or fast-only indexes

Choose the storage and scoring mode when constructing the index:

```python
exact = TieredQIndex(dimensions=100, code_bits=200)  # default FP32 scorer
fast = TieredQIndex(dimensions=100, code_bits=200, fast_mode=True)
fast.add(vectors)
scores, ids = fast.search(vectors[:2], k=10)
print(fast.fast_mode)       # True; read-only
print(fast.storage_bytes)   # allocated vector data, including padding/capacity
fast.save("fast.tq")
fast = TieredQIndex.load("fast.tq")  # preserves fast mode and is ready to search
```

`search()` honors the constructor's mode. Fast mode owns only the approximate
search representation: it does not retain the original packed data and needs no
`build_fast_scan()` call. Later appends encode into the same representation and
remain ready to search. The mode is fixed for the index's lifetime; saving and
loading preserves it. Both modes accept one query or a matrix and return the
same shapes and dtypes.

Fast mode trades some score/ranking accuracy for speed and still scans every
vector. Its memory footprint depends on the model and platform: x86 uses packed
byte lookups for tiers of at most four bits and expanded int8 levels for wider
tiers; ARM uses expanded int8 levels. A small `code_bits` budget therefore does
not guarantee a small fast representation. `bytes_per_vector` reports the
unpadded representation stride; `storage_bytes` reports allocated vector-data
bytes, including padding and append capacity but excluding shared models,
query projections, query state and top-k scratch. Initial encoding and file I/O
can use temporary buffers.

Fast-only files store the representation itself with a versioned payload.
Loading on another CPU preserves the representation and uses a portable scorer
when no compatible native kernel exists; performance can differ across CPUs.
Python emits a `RuntimeWarning` once per index when loading scalar-backed fast
data (or first adding data on a CPU without a native kernel). The warning
recommends rebuilding from the original vectors on this machine when a native
representation is available. Rebuilding cannot add SIMD support to unsupported
hardware. `uses_scalar_fast_scan` is a read-only property for checking the actual
fast backend; searches do not repeat the warning. Python's standard warning
filters control whether it is displayed or treated as an error.
Existing exact index files remain readable. See the
[measurements](docs/glove200-fast-scan.md) for precision and performance examples.

### Optional auxiliary cache on exact indexes

The previous opt-in cache API remains available:

```python
exact.add(vectors)
exact.build_fast_scan()
scores, ids = exact.search_fast(vectors[:2], k=10)
scores, ids = exact.search(vectors[:2], k=10)  # exact scorer
```

This keeps both representations. Without a cache, `search_fast()` falls back to
the exact scorer. A nonempty append invalidates this optional cache; rebuild it
before using fast search again. It is not saved with exact-mode indexes.
`has_fast_scan` reports cache availability. On a fast-only index,
`search_fast()` is an alias for configured search and `build_fast_scan()` is a no-op.

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
