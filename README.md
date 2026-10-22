# TieredQ

Standalone C++20 vector codec and append-only exhaustive in-memory index,
with training, encode/decode, scoring and persistence. No Python dependency.

Requires CMake >= 3.18, Clang/GCC, LAPACK/BLAS and OpenMP.
On macOS install Homebrew libomp; Accelerate supplies LAPACK.
On Debian/Ubuntu install liblapack-dev and libblas-dev.

    cmake -S . -B build -DTIEREDQ_TESTS=ON -DTIEREDQ_EXAMPLES=ON
    cmake --build build -j
    OMP_NUM_THREADS=2 ctest --test-dir build --output-on-failure
    OMP_NUM_THREADS=2 ./build/tieredq_example

The example reports training, encoding and search time and persists example.tq.
Install using cmake --install build --prefix /desired/prefix. Consumers use
find_package(TieredQ CONFIG REQUIRED) and link TieredQ::TieredQ.
FetchContent consumers link the same target (also available as dv_tieredq).
See [integration.md](integration.md) for contracts and private-repository builds.

Tests cover persistence with a golden model, malformed inputs,
row/block parity and index behavior. Grouped SIMD and scalar sums can differ.

Licensed under the [MIT License](LICENSE).
