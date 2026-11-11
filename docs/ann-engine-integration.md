# Integrating TieredQ IVF into Faiss and zvec (#6): plan

Status: draft (2026-11-11), against TieredQ b0148dd. Nothing is implemented and no PR is opened elsewhere. Faiss
signatures below are from the 1.x headers (`IndexIVF.h`, `InvertedLists.h`, `impl/CodePacker.h`); check them against
the v1.16.0 tag before writing code.

## What TieredQ already offers an engine

| Need | TieredQ API (b0148dd) |
|---|---|
| Train once on a sample | `TieredQ::train(config, sample, rows, seed)`; `TieredQIVFIndex::train_lists` / `set_lists` for a fixed partition |
| Encode / decode one vector | `TieredQ::encode` / `decode`: `bytes_per_vector()` row codes, list-independent (no residuals) |
| Score one code (asymmetric) | `TieredQ::query_weights(q)` then `TieredQ::score(query, rows, count)` on row-major codes |
| Scan a list with the fast kernels | `TieredQFastBlocks`: `pack(rows, count, block)`, `scanner(q).score(block, count, scores)`, `rows()` / `unpack()` to recover codes exactly; `format()` persisted per CPU family |
| Build from non-float storage | `TieredQIVFIndex::build(RowLoader, count)`, `sq8_row_loader` |
| Rescore candidates from SSD | `TieredQNestedIVFIndex` with `TieredQFileRows` (direct I/O, io_uring) or `rescore_batch` over rows the engine fetched itself |

The fast kernels are the reason to integrate. The plane path (`score_planes`) used by the October zvec prototype is
2–20× slower and is superseded (#6, comment of 2026-11-04).

## Faiss: `IndexIVFTieredQ`

Faiss's IVF is split into a coarse quantizer (any `Index`), an encoder (`train_encoder`, `encode_vectors`,
`sa_decode`), inverted lists, and an `InvertedListScanner`. The fast-scan indexes (`IndexIVFFastScan`, RaBitQ's fast
scan variant) store codes in fixed blocks of documents and pack them through a `CodePacker`. TieredQ fits that pattern:

1. **Encoder.** `by_residual = false`. `train_encoder` trains a `TieredQ` codec (fast target, metric from
   `metric_type`, `max_scan_bits` from `TieredQIVFIndex::automatic_scan_bits`). `code_size = bytes_per_vector()`.
   `encode_vectors` / `sa_decode` call `encode` / `decode`. Because codes do not depend on the list, `reconstruct`,
   `merge_from` and `remove_ids` stay correct.
2. **Lists.** A `BlockInvertedLists` with `n_per_block = 256` (one lookup tile) and `block_size` of a 256-document
   tile, plus a `CodePacker` (`pack_1` / `unpack_1`), so adding, merging and removing ids go through Faiss's own
   machinery.
3. **Scanner.** `set_query` prepares `query_weights` once per query. `scan_codes(n, codes, ...)` walks the list's
   tiles, calls `Scanner::score` per tile and keeps the top k in Faiss's heap. `distance_to_code` scores one row-major
   code with `TieredQ::score`. This is the single-vector path HNSW refinement and range search need.
4. **Metric.** Inner product passes through. For L2, Faiss wants ascending distances, so the scanner returns the negated
   L2-mapped score. Cosine is normalized inner product, as elsewhere in Faiss.
5. **Coarse quantizer.** It is the user's choice: `IndexFlat`, or `IndexHNSWFlat` for 64K+ lists. TieredQ's
   two-level router (#47) can come later, wrapped as a quantizer `Index`. Faiss IVF-HNSW is the baseline it beat in #52.
6. **Serialization.** `faiss::read_index` dispatches on fourcc codes in `index_read.cpp`, so an out-of-tree type needs
   its own read/write entry points, or an in-tree registration.

### Library changes TieredQ needs first

| Gap | Why | Change |
|---|---|---|
| Fixed-capacity tiles | `TieredQFastBlocks` lays out a whole run: its factor, correction and centroid arrays span all `count` documents before the codes, and the tail tile is shaped by `count`. A `CodePacker` must place one row into a fixed-size, self-contained tile | `TieredQFastBlocks::tile_bytes()`, `write_row(tile, slot, row)` and `score(tile, live, scores)`: a 256-slot tile with its own metadata, padded slots ignored |
| Score mapping | `Scanner` returns model-space scores. The L2 map (`TieredQ::score_map`) is private | Public `TieredQ::map_scores(query, scores, count)`, or a scanner option that returns metric scores |
| Pruned scans | Range-checkpoint pruning (`TieredQPruning`) is only reachable through `TieredQIVFIndex` | Optional `Scanner::score_top(tile, live, threshold, ...)`. At k = 100 pruning was neutral (#43), so this is a later speed-up |

The tile is the layout #43 measured as fastest (four 64-document chunks per table load). Per-tile metadata adds no
scanned bytes, only a few cache lines per 256 documents. Measure it against `TieredQIVFIndex` before upstreaming.

### Upstream or plugin

- **Plugin (`faiss-tieredq`), first.** A separate C++ library and Python module that subclasses `IndexIVF` against an
  installed Faiss. There is no review cycle and it can ship with TieredQ releases. The cost is its own serialization
  and SWIG/nanobind wrapper.
- **Upstream PR, after the plugin's numbers hold.** Faiss vendors its quantizers (RaBitQ landed natively as
  `IndexRaBitQ` / `IndexIVFRaBitQ`) and already depends on BLAS/LAPACK and OpenMP, which TieredQ training needs.
  The io_uring tier reader would stay out of it.
- **Nested tier.** Faiss's `IndexRefine` rescores a base index's top `k_factor × k` with a refine index's distance
  computer. A tier index whose `get_distance_computer()` scores an id's tier row (from RAM, or `TieredQFileRows` on
  SSD) gives the nested configuration without new search code. Phase 2.

### Acceptance (before any PR)

- Same partition (`set_lists` from the Faiss quantizer's centroids) and same nprobe: ids and scores equal
  `TieredQIVFIndex` up to tie order.
- VIBE arxiv k = 100 and Qwen 8.84M k = 10, single core: QPS within 10% of `TieredQIVFIndex` at equal recall, and
  measured RAM within 5% (tile metadata included).
- Against Faiss's own `IndexIVFRaBitQ` / `IndexIVFPQFastScan` on the same quantizer: recall per RAM at fixed recall,
  with QPS as a constraint (the #41 metric).

## zvec

The October prototype (`ivf_tieredq`, TieredQ 8440b12) scored lists with `score_planes`. Planned move:

1. Store each list's codes as one `TieredQFastBlocks` block (or as fixed tiles, once they exist) in zvec's segment
   storage, and persist `format()` with the segment.
2. Scan with `Scanner::score` and train the codec per collection (or per segment; see #19 §2 for drift and retraining).
3. Rebase onto current zvec. Add model formats 3/4 and `metric`.
4. Benchmark against zvec's `IVF_RABITQ` with accurate encoding (the open `faster_config` audit), counting RAM the
   #41 way.

This part is unverified against current zvec. Its source is not on this machine (`/Users/Studio-MacMini/projects/open-source/zvec`).

## Order and effort

1. TieredQ: fixed-capacity tiles, public score mapping, tests (byte-identical scores to `TieredQIVFIndex` on the same
   lists). About 1–2 days.
2. Faiss plugin: encoder, packer, scanner, I/O, Python wrapper, parity tests. About 2–3 days.
3. Benchmarks per the acceptance list (hartmut, about 1 h of runs).
4. zvec move onto the same tiles. About 2 days, after 1.
5. Upstream PR to Faiss only if 3 holds, and after asking.
