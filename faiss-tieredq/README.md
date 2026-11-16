# faiss-tieredq

`IndexIVFTieredQ` is a Faiss IVF index whose inverted lists hold TieredQ fast-scan blocks (#55). It is an
out-of-tree plugin: a C++ library built against an installed Faiss (v1.16.0), plus a small Python module.

- **Encoder.** A TieredQ codec (fast target, no fine tiers, the IVF scan cap of
  `TieredQIVFIndex::automatic_scan_bits`) trained on a seeded sample of up to 32,768 training rows. That is the rule
  TieredQ's own Python build uses, so the same rows train the same codec. `by_residual = false`: codes do not depend on
  the list, so `merge_from`, `remove_ids`, `reconstruct` and `sa_encode` / `sa_decode` stay exact. `code_size` is
  `bytes_per_vector()`.
- **Lists.** `TieredQInvertedLists` keeps each list in one `TieredQFastBlocks` block laid out for its capacity: its
  rows rounded up to 16 after a bulk add, growing by half when appends fill it. Fixed 256-document tiles (as Faiss's
  fast-scan indexes use) would pad a 64K-list index of 8.84M rows to about twice its size. Faiss's generic list
  operations (removal, subsets) work through decoded rows (`get_single_code`).
- **Search.** `IndexIVF::search_preassigned` with a scanner that scores each probed list in one call of TieredQ's fast
  kernels. Inner product returns TieredQ's score (larger is better), L2 returns squared distances (smaller is better).
  Cosine is inner product over rows and queries that the codec normalizes; normalize the quantizer's inputs too, as
  usual in Faiss. TieredQ scanners are pooled between searches (`TieredQFastBlocks::reset`).
- **Coarse quantizer.** Any Faiss `Index`: `IndexFlat`, or `IndexHNSWFlat` for 64K+ lists. `add_core` takes
  precomputed lists, so another index's partition can be reproduced exactly.
- **Persistence.** `write_index_tieredq` / `read_index_tieredq`, since an out-of-tree type cannot hook into
  `faiss::read_index`. The quantizer is stored inside the file with `faiss::write_index`.

## Build

```sh
cmake -S faiss-tieredq -B build -DCMAKE_PREFIX_PATH=/path/to/faiss/install [-DFAISS_TIEREDQ_FAISS_TARGET=faiss_avx512]
cmake --build build && ctest --test-dir build
# Python module (needs the same Faiss install):
CMAKE_PREFIX_PATH=/path/to/faiss/install uv pip install ./faiss-tieredq
```

The plugin builds the TieredQ tree it ships in (`TIEREDQ_SOURCE_DIR`, default `..`).

```python
from faiss_tieredq import IndexIVFTieredQ
index = IndexIVFTieredQ(768, 2048, code_bits=1344, metric="ip")   # 1.75 bits per dimension
index.train(rows); index.add(rows)
distances, labels = index.search(queries, k=100, nprobe=16)
```

The Python module builds its own quantizer (`quantizer="flat"` or `"hnsw"`), since an out-of-tree index cannot take
objects from Faiss's SWIG module. `set_centroids` fixes the partition.

## Acceptance

`benchmarks/acceptance.cpp` builds `TieredQIVFIndex`, `IndexIVFTieredQ` and Faiss's `IndexIVFRaBitQ` and
`IndexIVFPQFastScan` on one Faiss k-means partition, the same assignment and the same training rows. It then checks
parity and serves each index alone on one core. Results: `benchmark/results/hartmut-faiss-plugin-20261116/`.
