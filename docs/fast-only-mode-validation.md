# Fast-only mode validation

The constructor API and storage semantics are documented in [README.md](../README.md#exact-or-fast-only-indexes).

Validated on Apple M1 (ARM64) and hartmut Intel i9-7980XE (x86-64):

- 41 Python tests and 2/2 C++ tests pass on each machine.
- Default indexes preserve exact scoring and legacy exact files remain readable.
- Fast-only search matches the corresponding existing fast-cache path; empty/failed appends preserve state, successful appends stay ready, and copies detach on mutation.
- Python warns once per scalar-backed index at load/first add, honors warning filters and warnings-as-errors, and stays silent for native/exact indexes. Appends and searches do not repeat the warning.
- Native fast-only files move in both directions between ARM and x86 with byte-identical scores and IDs for the tested 64/256-bit models. Appending and saving/loading on the receiving machine also pass. Unsupported native representations use the portable scorer while preserving their stored numerical representation.
- Version, kind, level range, array length, nonfinite metadata, centroid IDs, level data and trailing-byte corruption are rejected even with a recomputed payload checksum.

## Large-model storage check

The existing 800-bit GloVe200 model on hartmut, 1,183,514 vectors:

| Allocated vector arrays | Bytes |
|---|---:|
| Exact packed data | 124,269,600 |
| Exact data plus auxiliary fast cache | 253,297,696 |
| Fast-only data | 129,028,096 |

The fast-only row stride is 109 bytes; padding explains the difference from rows × stride. Shared models, query projections, input vectors, temporary encoding/file buffers and search scratch are excluded. These numbers describe retained vector arrays, not peak process RSS.

Single and batched queries at one core and 18 threads match the existing fast-cache results exactly. Saving/loading preserves mode, storage size and results. No new throughput claim is made by this API/storage validation. The fast representation remains model- and CPU-dependent: ARM currently expands numerical levels to int8, including for narrow tiers.

Raw evidence: [fast-only-storage-results.json](benchmarks/fast-only-storage-results.json).

Source snapshots, the large-check harness, cross-architecture files and validation scripts are preserved on hartmut under `~/experiments/TieredQ-fast-only-20261023/`. This adds a versioned fast-only index payload without changing the legacy exact container/model formats.
