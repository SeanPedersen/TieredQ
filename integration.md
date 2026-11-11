# C++ integration contract

Include <tieredq/tieredq.h> and link TieredQ::TieredQ. Symbols live in the tq namespace. Headers under detail/ only complete private
class storage; their symbols are not a supported API. src/ holds private
training and SIMD implementation. Do not add it to normal consumer include
paths. Dispatch is automatic; explicit kernels support diagnostics.

## Codec and storage

TieredQ::train(config, sample, rows, seed) trains from contiguous row-major
finite FP32 data with config.dimensions columns. Input pointers are borrowed
only during calls. Configuration validates dimensions and budgets. Training
uses LAPACK and optional OpenMP. TIEREDQ_BLAS and
TIEREDQ_BLAS_STAGE select the optional training backend.

encode returns owned bytes or writes count * bytes_per_vector() bytes to
caller-owned storage. Each row contains code_bytes() packed code bytes and
5 metadata bytes: centroid ID, FP16 scale, FP16 projected correction.
decode writes count * dimensions floats. Do not overlap input and output.
Optional encode corrections receive the stored rounded correction.

query borrows one vector for the call and returns owned lookup buffers. Retain
them during scoring and use them only with their originating model. score
reads contiguous encoded rows and writes count floats. pack/unpack transpose
rows to/from byte planes of 16 documents. Plane buffers require
ceil(count/16) * 16 * bytes_per_vector() bytes, including padded lanes.
score_planes writes count scores; retain default query validation unless the
caller has already validated the query against the same model.

Scores are larger-is-better estimates under TieredQConfig::metric: inner
products (default), cosine similarity (rows and queries are normalized), or for
L2 the negative squared distance -|q - x|^2. Codec-level score()/score_planes()
score model space (model_dimensions() wide: normalized or lifted vectors); for
L2 only TieredQIndex maps those scores to distances. Engines can lift queries
once with lift_queries() and use the model_query* functions. Grouped/scalar
accumulations have established floating-point differences; compare tolerances
and validate top-k using the same scoring kernel. TieredQIndex::search scans
exhaustively, sorts descending score then ascending insertion ID for ties,
and returns at most min(k, size()) hits. IDs are uint32. add appends borrowed
FP32 rows and the index owns its resulting encoded storage.

## Fast and full index storage

Construct `tq::TieredQIndex(codec)` or Python
`TieredQIndex(dimensions=d, code_bits=b)` to retain the default fast
representation. Select `tq::TieredQIndexMode::full` in C++ or `mode="full"`
in Python to retain compact packed rows for full scoring. `search` and
`search_batch` honor this choice; fast-only indexes support appending and do not
require cache rebuilding. C++ explicit exact-kernel overrides are rejected on a
fast-only index. Mode selection is read-only (`mode()` returning `tq::TieredQIndexMode` in C++, `mode`
returning `"fast"` or `"full"` in Python).

`storage_bytes()` / Python `storage_bytes` accounts for allocated vector arrays,
including capacity and padding, but excludes model/query projection data and
search scratch. `bytes_per_vector()` / Python `bytes_per_vector` describes the
selected unpadded row representation. On ARM, the fast representation uses int8
levels even for narrow codebooks; x86 byte-lookup storage is used only for tiers
of four bits or less. Fast-only does not imply a smaller encoding than exact.

Exact files retain container kind 2. Fast-only files use kind 3 and a separately
versioned payload containing representation kind, level quantization range,
metadata arrays and packed lookup codes or int8 levels. No original packed rows
are stored. Query projection and cache geometry are reconstructed from the
persisted codec. Shapes, levels, centroid IDs, finiteness and trailing bytes are
validated before search. Cross-platform loads preserve the saved representation;
a portable scorer handles a representation unsupported by the local native
kernel. Python warns once per index on load/first add when scalar fast scoring
is selected; unsupported hardware gets a distinct warning that rebuilding will
not enable SIMD. C++ callers can query `uses_scalar_fast_scan()` and
`native_fast_scan_available()` and route diagnostics through their own logger.
The same backend selector drives scoring and these diagnostics. Exact-mode
auxiliary caches remain derived, unpersisted state.

Optional auxiliary caching remains available on full-mode indexes: call
`build_fast_scan()` then `search_fast` / `search_batch_fast`. This retains both
representations, leaves configured search exact, and requires rebuilding after a
nonempty append or load. Empty and failed appends preserve the optional cache.
On fast-only indexes, `build_fast_scan()` is a no-op and `search_fast` matches
configured search. `has_fast_scan` reports whether fast data is present.

## Fast blocks in engine storage

`TieredQFastBlocks` (`<tieredq/tieredq_blocks.h>`) serializes the fast representation of a run of encoded rows
into one caller-owned byte block (`block_bytes(count)`, `pack`) that an engine stores, e.g. per inverted list on
disk, and scans in place: `scanner(query)` prepares one query, `Scanner::score(block, count, scores)` writes
model-space inner-product estimates for the block's documents without copying it. `unpack` recovers the exact
`encode()` rows, and `rows(block, count, documents, n, out)` only those of selected documents (e.g. the prefix rows
a `TieredQRefinement` suffix tier rescores, without storing them twice). `scanner(weights)` takes the query's
`codec().query_weights()` instead, so a caller that also builds `TieredQRefinement::query(vector, weights)` projects
the query once. `format()` names the representation chosen for the model and CPU (nibble lookup, packed code
words or int8 levels); persist it and reopen stored blocks with `TieredQFastBlocks(codec, format)`. Other CPUs scan
a persisted format with the portable scorer. Blocks need no alignment (64-byte starts keep loads within cache
lines). A scanner is single-threaded; use one per thread and query. Nibble-lookup blocks end on a short tail block,
so small lists are not padded to 256 documents; pruned (range-checkpoint) scans do not apply to such views.

## Ownership and concurrency

Models and queries own their buffers. Const codec operations and index searches
may run concurrently with independent query, scratch and output buffers.
Never mutate queries during scoring. add, build_fast_scan, assignment, destruction
and other index mutations require external synchronization against readers/writers. Do not
change process-wide BLAS/OpenMP/backend settings during calls. Internal OpenMP
may add threads; set OMP_NUM_THREADS to control the budget.

save/load validate checksums and model/index layouts. TieredQIndex.load streams the
payload from the file into its final in-memory layout, so peak load memory stays near
the loaded index size; payload corruption is still reported as a checksum mismatch,
and a file is fully read before a load succeeds. DiskoVec owns routing,
disk extents, filtering, journals, reranking and engine-level zero markers.
Keep its zero-vector bypass based on input values, including under fast-math.

## Private GitHub dependency and CI

Pin FetchContent GIT_TAG to a full reviewed commit. Use
 git@github.com:SeanPedersen/TieredQ.git with an SSH identity authorized to
read the private repository. Override fetching for local development:

    cmake -S . -B build -DFETCHCONTENT_SOURCE_DIR_TIEREDQ=/absolute/path/TieredQ

For CI configure a read-only TieredQ deploy key as a secret, load it into
ssh-agent, and provide a verified GitHub known_hosts entry before configuring.
A checkout token for another repository does not grant access here. An
authorized credential helper with an SSH-to-HTTPS Git insteadOf mapping is
an alternative. Never embed tokens in CMake, URLs or build logs. Clean builds
and DiskoVec source distributions require authentication or a local override.

## Seeded training

Training uses std::mt19937_64, std::normal_distribution and uniform sampling
without replacement. A seed reproduces results within a fixed toolchain and
backend configuration. Standard-library distribution algorithms and BLAS may
vary across platforms; cross-toolchain trained-model byte identity is not a
contract. Persist models when sharing trained results between environments.

## Python API

TieredQIndex(dimensions, code_bits, *, mode="fast") is the only public class and the only
configuration surface. The first non-empty add() trains the codec on that
batch using min(256, rows // 2) centroids and seed 0, then stores the batch.
It needs at least 2 rows; empty adds are no-ops. Later add(rows) calls only
encode. len(index) reports the count, and search(queries, k=10) returns
(scores, ids) as float32 and uint32 arrays. A 1-D query returns 1-D results; a
2-D [queries, dimensions] array returns [queries, min(k, size)]. save(path)
and TieredQIndex.load(path) persist the trained index.

The former boolean constructor and `fast_mode` accessor are removed. Use `mode="full"`
instead of `fast_mode=False`, and `mode="fast"` instead of `fast_mode=True`. In C++,
replace boolean arguments with `tq::TieredQIndexMode::full` / `::fast`. Existing saved
indexes retain their file format and stored mode when loaded.

`search()` uses the selected storage mode; the default is fast scoring.
`mode="fast"` owns only fast data, persists the mode, and stays ready after
appends or load. The mode is read-only. On full-mode indexes,
`build_fast_scan()` prepares an optional auxiliary cache for `search_fast`.
The read-only `has_fast_scan`, `bytes_per_vector` and `storage_bytes` properties
report fast-data availability, unpadded representation stride and allocated
vector-data storage. Native calls release the GIL; writes hold the exclusive
Python index lock against concurrent readers.

The advanced codec surface (encode/decode/score, Config, Query) is C++ only.
Inputs must be CPU NumPy arrays of float32 with contiguous layout; read-only
inputs are accepted. Shape/type errors raise exceptions; inputs are borrowed
during calls and must not be mutated concurrently. Outputs own their buffers
and remain valid after the index is released. Native work releases the GIL.
The index serializes writes against readers and permits concurrent searches.

For development: uv sync --python 3.12 --dev --inexact, then uv run --no-sync pytest -q.
Installation builds the extension locally; no package is published to PyPI.

## Bit allocation

The allocator ranks each extra bit by PCA variance times the reduction in
expected Gaussian quantization error, subject to the configured bit caps.
Greedy allocation applies to every dimensionality. Optional rotations within
equal-bit tiers follow PCA and bit allocation.

## Engine containers and zvec IVF

`TieredQ::serialize()` returns owned model bytes, and
`TieredQ::deserialize(bytes)` validates and restores that model without temporary
files. These are the same model bytes used by the file API; engine containers
must checksum their complete model and payload. Neither method serializes index
rows. Persist the trained model instead of retraining it when reopening an index.

Model format 3 stores the PCA basis as IEEE binary16. Training rounds the basis once
and fits the projection statistics to the rounded values, so serving and encoding use
exactly what is persisted; that halves the basis on disk and in RAM. Format-2 models
(FP32 basis) still load, serve with identical results, and serialize back to identical
bytes. Older readers reject format 3 explicitly.

The zvec integration in its `src/core/algorithm/ivf_tieredq/` module uses this
API to embed the model alongside inverted lists. It registers native
`IvfTieredqBuilder` and `IvfTieredqStreamer` factories. Its initial scope is dense
FP32 inner-product search, up to 256 lists, and read-only search after building.
See that module's README for build flags, query controls, and focused tests.
