# TieredQ: Variance-Aware Bit Allocation

TieredQ is a new vector quantization algorithm that dynamically allocates bits per dimension, providing higher recall compression ratio (compared to RaBitQ or TurboQuant). TieredQ centers and rotates vectors into a PCA basis, then greedily allocates 0–8 bits per dimension to reduce quantization error (0–12 in the flat fast index, whose fine tiers scan only each code's top bits and rescore candidates exactly). Existing ANN indexes (HNSW / IVF) can use the codec directly as drop-in for RaBitQ or TurboQuant.

TieredQ core algorithm:
1. PCA
2. Variance-aware greedy bit allocation per dimension (principal component)
3. Scalar quantization per dimension with allocated bit budget

## Install

Install [TieredQ from PyPI](https://pypi.org/project/tieredq/):

```sh
uv add tieredq
# or: pip install tieredq
```

## Python

```python
import gensim.downloader as api
from tieredq import TieredQIndex

vectors = api.load("glove-wiki-gigaword-100").vectors  # (400000, 100), float32
index = TieredQIndex(dimensions=100, code_bits=200) # 2 bits per dim
index.add(vectors)  # first add trains (a seeded 32,768-row sample of larger batches); later adds only encode
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
- `mode="fast"` is the default and stores the native fast search representation. Models with tiers above 4 bits score
  through approximate byte tables (L2 models keep exact FP32 scoring); `exact_fast_scores=True` keeps bit-exact
  FP32 scoring at lower speed.
  Use `mode="full"` for compact packed rows and FP32 scoring of quantized vectors,
  with slower search. Both modes scan every vector; saving and loading preserve
  the selected representation. Read the selection from `index.mode`.
  The former `fast_mode` boolean argument/property has been removed; use `mode`.
- Fast ip/cosine indexes use fine tiers by default (#26, #31): ranks coded with up to `max_tier_bits=12` bits at the
  same bytes, whose top bits are an ordinary fast code at the automatic cap (6 bits, or ceil(bits/dim) when larger).
  The scan reads only those prefixes, then rescores max(32, 1.5·k) candidates exactly. Training uses them whenever the
  allocation codes a rank wider than the prefix; `fine_tiers=False` disables them, and `index.fast_prefix_bits` is the
  prefix width (0: none). 100k vectors, mean missed top-10 hits over 3 seeds, fine tiers vs none: Gemini-768 66 → 15,
  OpenAI-1536 44 → 6 at 8 bits/dim; 209 → 171 and 168 → 115 at 4 bits/dim; GloVe-200 162 → 87 at 8 bits/dim and
  unchanged at 2-4 (no rank goes wide). Batch time per query is +0-15% at 6-8 bits/dim and +30-40% at 3-4 bits/dim. On
  1M sets (OpenAI-1536, MS MARCO Cohere-1024, 6-8 bits/dim) top-10 misses fell 46-79% and search got 3-24% faster (likely
  as the scan reads fewer bits); encoding is 11-34% slower (wider codes). Only the flat `TieredQIndex` serves fine-tier
  models: train IVF, nested and refinement models with `TieredQConfig::fine_tiers = false` (the Python IVF and nested
  indexes do).
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

### IVF and nested search (Python)

```python
from tieredq import TieredQIVFIndex, TieredQNestedIVFIndex

ivf = TieredQIVFIndex(dimensions=d, code_bits=2 * d, lists=1024)  # lists=0: one list per codec centroid
ivf.build(vectors)  # trains the codec (seeded 32,768-row sample) and the lists, then indexes
scores, ids = ivf.search(queries, k=10, nprobe=32)

# Prefix codes in an in-RAM IVF; candidates rescored with a 4 bits/dim refinement tier on SSD.
index = TieredQNestedIVFIndex.build(vectors, prefix_bits=768, total_bits=4 * d, lists=256)
index.write_tier("tier.rows", vectors)  # or index.attach_tier(index.encode_tier(vectors)) to keep it in RAM
index.attach_tier("tier.rows", reads="direct", queue_depth=32)
scores, ids = index.search(queries, k=10, nprobe=64, candidates=50)
index.save("nested.tq")  # IVF, refinement and tier layout; attach the tier again after load()
```

- One `build` trains the prefix, the lists and the refinement from the same seeded sample, so they cannot be
  mismatched. Pass `sample=` to train on earlier data instead; its rows also fix the lists for later builds (as
  `TieredQIVFIndex.train_lists` does).
- `tier_rows="suffix"` stores a smaller tier and decodes each candidate's prefix from the lists (see the C++ notes).
- Tier rows kept in another store (a database, an object store): `ids = index.candidates(queries, 50, nprobe=64)`,
  fetch the `encode_tier` rows of those ids (shape `ids.shape + (index.tier_bytes_per_vector,)`; ids of
  `2**32 - 1` are padding), then `scores, ids = index.rescore(queries, ids, rows, k=10)`.
- `pruning=TieredQPruning(confidence=8)` opts IVF and nested scans into approximate pruning (see below).

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

`dv::TieredQ::train(config, sample, rows, seed)` trains on a contiguous sample. To train on part of a larger
array without copying it, `dv::TieredQ::train_sampled(config, data, count, sample_ids, sample_rows, seed)` reads
the named rows in place (the same model). About 8–16K training rows reach full recall on 1M-row sets; fewer than
~32 rows per centroid lose recall ([#19.2 results](benchmark/results/issue19-drift-20261031/README.md)).

### IVF (C++)

`dv::TieredQIVFIndex` is a fast-mode inverted-file index. With `lists = 0` the codec's k-means centroids
(at most 256) are the lists, at no extra storage. Any other count trains a separate coarse quantizer, whose
centroids are stored.

```cpp
dv::TieredQIVFIndex index(codec, /*lists=*/1024);
index.train_lists(sample, sample_rows);  // optional: fix the lists on a sample; builds then only assign rows
index.build(rows, count);
auto hits = index.search(query, /*k=*/10, /*nprobe=*/16);
// Opt-in approximate pruning for models with more than four tiers (about 2+ bits/dim):
auto fast = index.search(query, 10, 16, dv::TieredQPruning{/*confidence=*/8, /*prefix_tiers=*/4});
```

- **Pruning.** By default the scan prunes exactly: hits equal an unpruned scan. A positive `confidence`
  bounds the remaining tiers statistically instead, which is faster and approximate. On OpenAI 100k, confidence 8
  kept recall unchanged (±0.0001) at 1.7× the QPS at 4 bits/dim. With `prefix_tiers = 3` it gave 1.6–2.4× the QPS
  for 0.001–0.003 recall at 2 and 4 bits/dim. Confidence 4 or less loses clearly more recall.
  Platforms: the exact FP32 code scan prunes on AVX-512 only. The nibble byte-table scan (default fast models with
  tiers above 4 bits) prunes on AVX-512, AVX2 and NEON. The scalar fallback validates the setting and scans unpruned.
  By default (`prefix_tiers = 0`, automatic) a nibble scan prunes after all but the last 12.5% of its sections
  (37.5% with a positive confidence), or after the first four tiers where those leave more; an explicit
  `prefix_tiers` prunes after that many tiers. Single-core, 100k documents, 256 lists, on Apple M1 (NEON), EPYC Milan
  (AVX2) and i9-7980XE (AVX-512) with 384-, 768-, 1024- and 1536-d embeddings: exact pruning returned the unpruned
  hits in every run and was 1.0–1.3× faster than unpruned at nprobe 64–256 (about 1.0× at nprobe 16); confidence 8
  was 1.0–1.8× faster with unchanged recall@10. Flat single-query search prunes the same way, 1.0–1.2× faster except
  Gemini-768 at 2 bits/dim, 5–18% slower. Larger suffix shares lose with the exact bound, which loosens faster than it
  saves work.
  To prune through the exact FP32 scan instead, train with `TieredQConfig::target_mode = TieredQIndexMode::full`
  (Python `exact_fast_scores=True`). Measured at 4 bits/dim, before nibble scans were pruned: on OpenAI-1536 100k
  exact with confidence 8 was 4–11% faster than nibble scoring at equal or slightly higher recall, and on OpenAI 1M
  17–21% faster but 0.4 pt R@10 lower; on Gemini-768 nibble scoring was 20–38% faster, at 0.2 pt R@10 lower. At
  2 bits/dim nibble scoring was faster on both (#13).
- **Bit cap.** At about 2 bits/dim or less, training with `TieredQConfig::max_scan_bits = 4` keeps every tier on the
  nibble lookup kernel. That gave 1.7× RaBitQ's IVF QPS at equal recall and memory (OpenAI 100k), for about
  0.006 recall at full probe. Don't use it at 4 bits/dim, where it loses recall.
- **Sub-1-bit index + rerank.** With rerank, the in-RAM IVF code can be far below 1 bit/dim. On OpenAI-1M (256
  lists), a 512-bit prefix (627 index bits/vector) with an INT8 rerank tier reached R@10 0.95 / 0.99 at 2.8x / 3.1x
  the QPS of RaBitQ 1 bit (1,683 bits) + INT8.
- **Nested rerank codes.** `dv::TieredQRefinement` (`<tieredq/tieredq_refinement.h>`) trains a suffix that refines
  the prefix towards a finer model. The suffix is read by id for the scan's candidates and scored with the scan's query
  weights; nesting costs about no recall against a flat code of the full width:

  `dv::TieredQNestedIVFIndex` (`<tieredq/tieredq_nested.h>`) serves this end to end. The prefix codes stay in the
  in-RAM IVF lists. Each candidate's tier row is read by id from a `TieredQRowSource`: `TieredQMemoryRows` in RAM,
  `TieredQFileRows` with one `pread` per candidate on SSD (`TieredQFileReads::direct` bypasses the page cache), or your
  own store. By default a tier row holds the prefix
  row and the suffix (`TieredQTierRows::prefix_and_suffix`, one contiguous read). `TieredQTierRows::suffix` stores
  only the suffix and decodes each candidate's prefix from the IVF lists (`TieredQIVFIndex::document_rows`). That tier
  is smaller (651 vs 784 B/vector at a 1024-bit prefix, d = 1536) but costs about 3.4 us of decoding per candidate and
  a 4-byte id-to-document map in RAM.

  ```cpp
  auto refinement = dv::TieredQRefinement::train(codec, /*total_code_bits=*/4 * d, sample, rows, seed);
  dv::TieredQIVFIndex ivf(codec);
  ivf.build(data, n);
  dv::TieredQNestedIVFIndex index(std::move(ivf), std::move(refinement));
  dv::TieredQRowFileWriter writer("tier.rows", index.tier_bytes_per_vector());
  for (size_t first = 0; first < n; first += chunk) {        // ids in build order
    index.encode_tier(data + first * d, first, count, buffer);  // count = min(chunk, n - first)
    writer.append(buffer, count);
  }
  writer.finish();
  index.attach_tier(std::make_shared<dv::TieredQFileRows>("tier.rows"));
  auto hits = index.search(query, /*k=*/10, /*nprobe=*/32, /*candidates=*/50);
  ```
- **Flat nested search.** `dv::TieredQNestedIndex` (`<tieredq/tieredq_nested_flat.h>`) scans a narrow prefix for
  every document and rescores the top candidates with an in-RAM suffix tier (a one-list `TieredQNestedIVFIndex`). On
  OpenAI-1536 it is the fast flat path from about 3 bits/dim. A 512-bit prefix with a 3 bits/dim total tier reached
  RaBitQ's 4-bit recall at 2.2x (100K, 50 candidates) and 2.9x (1M, 100 candidates) its single-query QPS, with
  0.76-0.79x its stored bits. The plain flat index is at 0.54x there.

  ```cpp
  auto prefix = dv::TieredQ::train(dv::TieredQConfig::target_code_bits(d, 512), sample, rows, seed);
  auto refinement = dv::TieredQRefinement::train(prefix, /*total_code_bits=*/3 * d, sample, rows, seed);
  dv::TieredQNestedIndex index(std::move(prefix), std::move(refinement), data, n);
  auto hits = index.search(query, /*k=*/10, /*candidates=*/100);
  ```

  `serialize()` stores the IVF, the refinement and the tier layout; after loading, attach the tier again. For a
  tier in an external store, run `index().search_batch` for the candidates and `rescore_batch` with their fetched
  tier rows; it needs no attached tier.

  At 1M, 512 / 768 bit prefixes + a 4-bit refinement (about 700 B/vector) beat RaBitQ's own 5/6-bit codes by 1.2-2.4x
  QPS at R@10 0.95-0.98 with 28-43% fewer rerank bytes. Refinement recall tops out near 0.984; above that use INT8.
- **Results** against RaBitQ's IVF at equal measured memory are in
  [issue #19](https://github.com/SeanPedersen/TieredQ/issues/19), [#23](https://github.com/SeanPedersen/TieredQ/issues/23)
  and the [issue #6 eval](benchmark/experiments/issue6_ivf/ivf_eval.cpp)
  ([rerank results](benchmark/results/issue6-ivf-rerank-20261030/README.md)).

## Codec comparison benchmark

See [benchmark/README.md](benchmark/README.md) for the one-command CPU comparison
against official RaBitQ on SIFT1M, GloVe-200 and OpenAI3-1536, including measured
storage overhead, recall, throughput and latency.

Disclaimer: everything but the core idea and the README was created using both Opus 5.5 and GPT 6.1 Sol
