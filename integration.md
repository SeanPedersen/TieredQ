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

## Ownership and concurrency

Models and queries own their buffers. Const codec operations and index searches
may run concurrently with independent query, scratch and output buffers.
Never mutate queries during scoring. add, assignment, destruction and index
mutation require external synchronization against readers/writers. Do not
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
