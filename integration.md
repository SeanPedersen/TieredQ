# C++ integration contract

Include <tieredq/tieredq.h> and link TieredQ::TieredQ. The dv namespace is
retained for source compatibility. Headers under detail/ only complete private
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

Scores estimate inner products; larger is better. Cosine callers normalize
inputs and queries. Scores are not squared L2 distances. Grouped/scalar
accumulations have established floating-point differences; compare tolerances
and validate top-k using the same scoring kernel. TieredQIndex::search scans
exhaustively, sorts descending score then ascending insertion ID for ties,
and returns at most min(k, size()) hits. IDs are uint32. add appends borrowed
FP32 rows and the index owns its resulting encoded storage.

## Exact and optional fast scans

`TieredQIndex::search` and `search_batch` retain the exact compressed FP32 scorer.
`build_fast_scan()` creates an optional, approximate auxiliary cache; use
`search_fast` or `search_batch_fast` to select it explicitly. Both modes evaluate
every document without reranking. Score and ranking differences are expected in
fast mode, and its memory cost depends on the model and CPU.

`has_fast_scan()` reports whether a supported cache is present. Fast search falls
back to exact search when it is absent. A nonempty successful `add()` invalidates
the cache; rebuild it after appending. Empty and failed adds preserve it. Cache
contents are derived and not persisted, so loaded indexes start without a fast
cache. Exact search remains available after building the fast cache.

## Ownership and concurrency

Models and queries own their buffers. Const codec operations and index searches
may run concurrently with independent query, scratch and output buffers.
Never mutate queries during scoring. add, build_fast_scan, assignment, destruction
and other index mutations require external synchronization against readers/writers. Do not
change process-wide BLAS/OpenMP/backend settings during calls. Internal OpenMP
may add threads; set OMP_NUM_THREADS to control the budget.

save/load validate checksums and model/index layouts. DiskoVec owns routing,
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

TieredQIndex(dimensions, code_bits) is the only public class and the only
configuration surface. The first non-empty add() trains the codec on that
batch using min(256, rows // 2) centroids and seed 0, then stores the batch.
It needs at least 2 rows; empty adds are no-ops. Later add(rows) calls only
encode. len(index) reports the count, and search(queries, k=10) returns
(scores, ids) as float32 and uint32 arrays. A 1-D query returns 1-D results; a
2-D [queries, dimensions] array returns [queries, min(k, size)]. save(path)
and TieredQIndex.load(path) persist the trained index.

`search()` stays exact by default. `build_fast_scan()` prepares approximate
search, the read-only `has_fast_scan` property reports availability, and
`search_fast(queries, k=10)` uses the same input/output shapes as `search()`.
Matrix inputs use the native fast batch path. Without a supported cache it falls
back to exact search. Rebuild after a nonempty append or a load; cache construction
and append hold the exclusive Python index lock against concurrent readers.

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

The zvec integration in its `src/core/algorithm/ivf_tieredq/` module uses this
API to embed the model alongside inverted lists. It registers native
`IvfTieredqBuilder` and `IvfTieredqStreamer` factories. Its initial scope is dense
FP32 inner-product search, up to 256 lists, and read-only search after building.
See that module's README for build flags, query controls, and focused tests.
