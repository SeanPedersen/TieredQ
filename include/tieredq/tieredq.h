// Runtime-sized TieredQ. Link dv_tieredq (LAPACK/BLAS); no Python dependency.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <utility>
#include "tieredq/detail/tieredq_scan_plan.h"
#include "tieredq/detail/tieredq_elias_fano.h"
#include "tieredq/detail/tieredq_shared_array.h"

namespace dv {
namespace tieredq_detail { struct FastFine; class FastRowReader; class StageProfile; struct FastScanCache; struct FastQueryBasis; struct EncoderData; class LazyEncoder;
                           struct TrainingRows;
                           struct CoarseQuantizer; struct CoarseRouter; struct Writer; }
enum class TieredQIndexMode { fast, full };
enum class TieredQKernel { automatic, scalar, avx2, avx512, neon };
// Similarity an index serves; scores are always larger-is-better. Cosine normalizes rows and queries. L2 lifts rows
// and queries into one more dimension (whitened for the query distribution, as calibrated on the training sample),
// where inner products rank by distance; index scores are estimates of -|q - x|^2.
enum class TieredQMetric { inner_product, cosine, l2 };
struct TieredQConfig {
  uint32_t dimensions = 512;
  uint32_t code_bits = 512;
  uint32_t num_centroids = 256;
  bool rotate_tiers = true;
  // Training-only cap on the bits per coordinate the fast scan reads (the full tier width unless fine tiers are on);
  // persisted models record only the resulting tiers. 0 (automatic) is 6 for fast-target inner product and cosine, or
  // code_bits / dimensions rounded up when larger, and 8 otherwise.
  uint32_t max_scan_bits = 0;
  // Training-only fine tiers (#26), for fast-target inner product and cosine with a scan cap of at least 6 bits: ranks
  // may take up to max_tier_bits in total, and the fast scan reads only each code's top max_scan_bits (its prefix)
  // before rescoring its candidates exactly, at the same stored bytes. Models get them whenever the allocation codes a
  // rank wider than the prefix; other targets and metrics ignore the flag. The flat TieredQIndex and
  // TieredQIVFIndex serve fine-tier models; other indexes reject them, so train their models with fine_tiers off.
  bool fine_tiers = true;
  uint32_t max_tier_bits = 12;
  // Training-only: only the top-N PCA ranks may exceed 6 bits (UINT32_MAX disables the restriction).
  uint32_t wide_dims = UINT32_MAX;
  // Training-only: the index mode whose storage layout bit allocation optimises for. Fast models with all tiers
  // <= 4 bits use the x86 nibble lookup: 3-bit tiers take an additive codebook whose codes split into 2- and 1-bit
  // nibble parts, and code_bits bounds padded nibble storage (or the padded size of 2/4-bit tiers at code_bits, if
  // larger), so the realised config().code_bits may differ.
  // Full mode packs codes densely and keeps the Lloyd-Max 3-bit codebook. Either model serves in either mode.
  TieredQIndexMode target_mode = TieredQIndexMode::fast;
  TieredQMetric metric = TieredQMetric::inner_product;
  // Training-only: allocate bits on PCA variance^allocation_power. 0 picks the metric's default in either mode:
  // for inner product and cosine, whose queries follow the data, 1.5 from 1 bit per dimension up (#9), falling to 1
  // at 0.5 bits per dimension and below (#22); 1 (MSE) for L2, whose lift whitens queries.
  double allocation_power = 0;
  static TieredQConfig target_bytes(uint32_t dimensions, uint32_t bytes);
  static TieredQConfig target_code_bits(uint32_t dimensions, uint32_t bits);
  static TieredQConfig target_bytes_per_vector(uint32_t dimensions, uint32_t bytes) {
    return target_bytes(dimensions, bytes);
  }
  void validate() const;
};
struct TieredQTier {
  uint32_t start, dimensions, bits, scale_group, bit_offset; // persisted scale_group is reserved and must be zero
};
// One FP32 scan lookup over count whole coordinates starting at rank first (tieredq_scan_groups.h).
struct TieredQScanGroup {
  static constexpr uint32_t wide = UINT32_MAX;  // table value of a coordinate scored from its shared codebook
  uint32_t bit_offset, bits, first, count, table;
};
struct TieredQQuery {
  // Float LUT entries already include sigma. One table per coded coordinate.
  std::vector<float> lut, centroid_scores;
  // q.c / |c|^2 per centroid: scaling a row's centroid correction by its centroid's weight removes exactly the
  // error component along c, leaving (q - weight * c).(r - r_hat).
  std::vector<float> correction_weights;
  std::vector<size_t> offsets;
  // Generated coordinate weights let index scans reuse shared wide codebooks.
  // The public row scorer continues to consume lut directly.
  std::vector<float> weights;
  // FP32 grouped scan plan and its float tables; queries without them use the per-coordinate kernels.
  std::vector<TieredQScanGroup> scan_groups;
  std::vector<float> scan_tables;
};
struct TieredQHit { float score; uint32_t id; };
// Right-hand GEMM operand in K-major panels of PANEL columns (zero padded).
struct TieredQPanels {
  static constexpr size_t PANEL = 48;
  size_t depth = 0, columns = 0;
  std::vector<float> data;
  size_t padded_columns() const { return (columns + PANEL - 1) / PANEL * PANEL; }
};
class TieredQIndex;
class TieredQRefinement;
class TieredQNestedIVFIndex;
class TieredQFastBlocks;
class TieredQ {
 public:
  static TieredQ train(TieredQConfig config, const float* sample, size_t rows, uint64_t seed = 0);
  // Trains on the rows sample_ids[0..sample_rows) of data (count rows) in that order without copying them: the same
  // model as train() on those rows gathered into one array.
  static TieredQ train_sampled(TieredQConfig config, const float* data, size_t count, const uint32_t* sample_ids,
                               size_t sample_rows, uint64_t seed = 0);
  const TieredQConfig& config() const { return config_; }
  const std::vector<TieredQTier>& tiers() const { return tiers_; }
  // Width of the fast-scan prefix of fine-tier models (#26): tiers wider than it refine a prefix code of this width.
  // 0 for models without fine tiers.
  uint32_t fast_prefix_bits() const { return fast_prefix_bits_; }
  size_t coded_dimensions() const { return sigma_.size(); }
  size_t code_bytes() const { return (size_t(config_.code_bits) + 7) / 8; }
  size_t metadata_bytes_per_vector() const;
  size_t bytes_per_vector() const { return code_bytes() + metadata_bytes_per_vector(); }
  size_t model_bytes() const;
  // Resident model memory: serving state, plus encoder state once encode() has built it.
  size_t memory_bytes() const;
  uint32_t model_id() const;
  void decode(const uint8_t* payload, size_t count, float* rows) const;
  // Plane buffers contain ceil(count/16) complete blocks of 16 lanes.
  // Automatic packed scoring selects NEON on AArch64; explicit NEON row scoring uses scalar.
  void pack(const uint8_t* rows, size_t count, uint8_t* planes) const;
  void unpack(const uint8_t* planes, size_t count, uint8_t* rows) const;
  // Scores always consume the correction embedded in each row.
  void score_planes(const uint8_t* planes, size_t count, const TieredQQuery& query,
                    float* scores, TieredQKernel kernel = TieredQKernel::automatic,
                    bool validate_query = true) const;
  // Batched score_planes: one pass over the plane block serves up to four queries.
  // Query i writes scores[i * score_stride + doc]; score_stride must be at least count.
  void score_planes_batch(const uint8_t* planes, size_t count, const TieredQQuery* queries,
                          size_t query_count, float* scores, size_t score_stride,
                          TieredQKernel kernel = TieredQKernel::automatic,
                          bool validate_query = true) const;
  // Rows are contiguous finite float32 vectors of config().dimensions. Codec-level scores estimate inner products
  // in model space: the metric's normalized or lifted vectors (model_dimensions() wide); TieredQIndex maps L2 scores
  // to negative squared distances.
  std::vector<uint8_t> encode(const float* rows, size_t count) const;
  // Writes count * bytes_per_vector() bytes; parallel over row blocks outside OpenMP regions.
  // Optional output receives the stored, rounded centroid correction for each row.
  // Metadata uses round-to-nearest-even binary16 and rejects overflow/nonfinite values.
  void encode(const float* rows, size_t count, uint8_t* out, float* corrections = nullptr) const;
  TieredQQuery query(const float* vector) const;
  // Projection and centroid terms without LUTs or scan groups, for weight-based fast scans.
  // Use query() to prepare input for score() and score_planes().
  TieredQQuery query_weights(const float* vector) const;
  // Batched query_weights over contiguous rows; results equal per-row calls exactly.
  std::vector<TieredQQuery> query_weights(const float* vectors, size_t rows) const;
  void score(const uint8_t* payload, size_t count, const TieredQQuery& query,
             float* scores, TieredQKernel kernel = TieredQKernel::automatic) const;
  // Model space: the metric's transform of input vectors. Engines that prepare many queries can lift them once
  // and call the model_* variants, which take model_dimensions() floats per vector.
  size_t model_dimensions() const;
  void lift_queries(const float* queries, size_t count, float* out) const;
  TieredQQuery model_query(const float* vector) const;
  TieredQQuery model_query_weights(const float* vector) const;
  std::vector<TieredQQuery> model_query_weights(const float* vectors, size_t rows) const;
  // Format 3 stores the PCA basis as IEEE binary16; format-2 files (FP32 basis) still load and serve unchanged.
  // Format 4 adds the metric and its lift; inner-product models keep writing format 3.
  static constexpr uint32_t metric_model_format = 4;
  static constexpr uint32_t model_format = 3;
  static constexpr uint32_t fp32_basis_model_format = 2;
  static constexpr size_t metadata_bytes = 5;
  static bool kernel_available(TieredQKernel kernel);
  void save(const std::string& path) const;
  static TieredQ load(const std::string& path);
  // Owned model bytes for embedding in an engine's storage container.
  std::vector<uint8_t> serialize() const;
  static TieredQ deserialize(const std::vector<uint8_t>& data);
 private:
  friend class TieredQIndex;
  friend class TieredQIVFIndex;
  friend struct tieredq_detail::FastQueryBasis;
  friend class TieredQFastBlocks;
  friend class TieredQRefinement;
  friend class TieredQNestedIVFIndex;
  TieredQ() = default;
  // The fast-scan view of a fine-tier model: the same model serving prefix tiers (none wider than its prefix width,
  // code_bits = their bits) without fine tiers; it shares the model's large arrays, and derived state is rebuilt as
  // on load.
  TieredQ prefix_view(std::vector<TieredQTier> tiers) const;
  static TieredQ train_model(TieredQConfig config, const tieredq_detail::TrainingRows& sample, size_t rows, bool additive_wide_tiers, uint64_t seed);
  TieredQConfig config_;
  std::vector<TieredQTier> tiers_;
  uint32_t fast_prefix_bits_ = 0;
  // The large arrays are shared by copies of the model (e.g. a fine-tier prefix view).
  tieredq_detail::SharedArray<float> centroids_, basis_;
  std::vector<float> mean_, sigma_;
  // Binary16 basis of format-3 models in 16-rank panels (tieredq_basis_panels.h; files keep d x m row order);
  // exactly one of basis_ (row-major) and basis_half_ holds the d x m basis.
  tieredq_detail::SharedArray<uint16_t> basis_half_;
  // memory_bytes() less the arrays this model shares with `owner` (counted there).
  size_t memory_bytes_beyond(const TieredQ& owner) const;
  // Basis entry at row-major index input * m + rank, in either encoding.
  float basis_value(size_t index) const;
  std::vector<std::vector<float>> levels_;
  // L2 lift: input scale and the lower Cholesky factor of the lifted query second moment (row-major; the base map).
  float lift_scale_ = 1;
  std::vector<float> lift_base_map_;
  // Derived (not serialized): inverse transpose of the base map, which whitens lifted queries, and both maps as
  // GEMM panels.
  std::vector<float> lift_query_map_;
  TieredQPanels lift_base_panels_, lift_query_panels_;
  // Derived serving state (not serialized): float half norms supply query correction weights.
  std::vector<float> centroid_half_norms_;
  // Derived scan geometry (not serialized): window and shift per grouped FP32 lookup, built from tiers_.
  std::vector<tieredq_detail::ScanGroupPlan> scan_plan_;
  // Encoder-only state (fused panels, FP64 centroid projections), built on first encode().
  std::shared_ptr<tieredq_detail::LazyEncoder> encoder_;
  void prepare_serving();
  tieredq_detail::EncoderData build_encoder() const;
  void encode_block(const float* rows, size_t count, uint8_t* out, float* corrections, tieredq_detail::StageProfile* profile = nullptr) const;
  void encode_rows(const float* rows, size_t count, uint8_t* out, float* corrections) const;
  // Up to eight validated rows; shares basis traffic and vectorizes centroid dots across rows.
  void project_query_tile(const float* vectors, size_t rows, TieredQQuery* out) const;
  void validate_model() const;
  // Writes count model-space vectors of model_dimensions() floats.
  void lift(const float* vectors, size_t count, bool queries, float* out) const;
  void prepare_lift();
  // Maps a model-space score of a query to the metric's score (identity except for L2).
  std::pair<double,double> score_map(const float* query) const;
};
class TieredQIndex {
 public:
  // Fast mode owns only its approximate search representation; the mode is persisted.
  explicit TieredQIndex(TieredQ codec, TieredQIndexMode mode = TieredQIndexMode::fast);
  ~TieredQIndex();
  TieredQIndex(const TieredQIndex&) = default;
  TieredQIndex& operator=(const TieredQIndex&) = default;
  TieredQIndex(TieredQIndex&& other) noexcept
      : codec_(std::move(other.codec_)), fast_mode_(other.fast_mode_), payload_(std::move(other.payload_)),
        fast_(std::move(other.fast_)), fine_tiers_(std::move(other.fine_tiers_)), fine_(std::move(other.fine_)),
        count_(std::exchange(other.count_,0)) {}
  TieredQIndex& operator=(TieredQIndex&& other) noexcept {
    if (this!=&other) {
      codec_=std::move(other.codec_); fast_mode_=other.fast_mode_; payload_=std::move(other.payload_);
      fast_=std::move(other.fast_); fine_tiers_=std::move(other.fine_tiers_); fine_=std::move(other.fine_);
      count_=std::exchange(other.count_,0);
    }
    return *this;
  }
  const TieredQ& codec() const { return codec_; }
  size_t size() const { return count_; }
  TieredQIndexMode mode() const { return fast_mode_ ? TieredQIndexMode::fast : TieredQIndexMode::full; }
  // Actual fast-data dispatch; false when no fast representation is present.
  bool uses_scalar_fast_scan() const;
  static bool native_fast_scan_available();
  size_t payload_bytes() const;
  // Allocated vector data and fast-cache state, including its derived query projection/codebooks and a fine-tier
  // model's prefix view (beyond the arrays it shares with codec()). Excludes codec memory, allocator bookkeeping,
  // query state and top-k scratch.
  size_t storage_bytes() const;
  size_t bytes_per_vector() const;
  void add(const float* rows, size_t count);
  std::vector<TieredQHit> search(const float* query, size_t k,
                                TieredQKernel kernel = TieredQKernel::automatic) const;
  // Optional auxiliary cache for exact-mode indexes; fast-only indexes need no rebuild.
  // Models containing tiers wider than 4 bits preserve code IDs for FP32 scoring. Other models
  // use int8 levels on ARM and compact lookups on x86 when all tiers fit four bits.
  // Successful nonempty appends invalidate only an exact-mode auxiliary cache.
  // No supported native kernel leaves exact-mode indexes without this optional cache.
  void build_fast_scan();
  bool has_fast_scan() const { return fast_!=nullptr; }
  // Full scan using fast data, or exact fallback when an exact-mode cache is absent.
  std::vector<TieredQHit> search_fast(const float* query, size_t k) const;
  // Batched search over rows x dimensions queries. Results are rows x min(k, size()) in
  // row-major order; the caller owns both output arrays. Runs OpenMP workers internally.
  void search_batch(const float* queries, size_t rows, size_t k,
                    float* scores, uint32_t* ids) const;
  // Same contract as search_batch, using fast data. A portable scorer handles saved
  // representations unsupported by local native kernels; absent exact-mode caches fall back.
  void search_batch_fast(const float* queries, size_t rows, size_t k,
                         float* scores, uint32_t* ids) const;
  void save(const std::string& path) const;
  static TieredQIndex load(const std::string& path);
 private:
  // Model-space searches; the public searches lift queries once and map the scores they return.
  std::vector<TieredQHit> search_model(const float* query, size_t k, TieredQKernel kernel) const;
  std::vector<TieredQHit> search_fast_model(const float* query, size_t k) const;
  void search_batch_model(const float* queries, size_t rows, size_t k, float* scores, uint32_t* ids) const;
  void search_batch_fast_model(const float* queries, size_t rows, size_t k, float* scores, uint32_t* ids) const;
  void finish_scores(const float* query, float* scores, size_t count) const;
  // Appends rows already encoded by codec() (bytes_per_vector() each).
  void add_encoded(const uint8_t* rows, size_t count);
  // Fills an empty fast-mode index from a serialized fast payload (tieredq_index_read.h).
  template<class Reader> void read_fast_payload(Reader& reader, size_t rows);
  friend class TieredQIVFIndex;
  friend class TieredQFastBlocks;
  std::shared_ptr<tieredq_detail::FastScanCache> make_fast_storage(
      const uint8_t* rows, size_t count, bool lookup, int level_max) const;
  std::shared_ptr<tieredq_detail::FastScanCache> make_fast_storage(const uint8_t* rows, size_t count) const;
  TieredQ codec_;
  bool fast_mode_ = false;
  // Exact-mode planes across 16 documents; empty for fast-only indexes.
  std::vector<uint8_t> payload_;
  std::shared_ptr<tieredq_detail::FastScanCache> fast_;
  // Fine-tier fast models (#26): the prefix view the scan uses, and every document's fine bits.
  std::shared_ptr<const tieredq_detail::FastFine> fine_tiers_;
  std::vector<uint8_t> fine_;
  size_t count_ = 0;
  const TieredQ& scan_codec() const;
  // Exact scores of scanned candidates; rows[i] is hits[i]'s prefix row (fine_read_slack readable bytes past each).
  void score_fine(const TieredQQuery& weights, TieredQHit* hits, size_t count, const uint8_t* const* rows) const;
  static void keep_best(std::vector<TieredQHit>& hits, size_t k);
  // Rescores every query's scanned candidates exactly and keeps each one's best k: decodes the distinct candidates of
  // a chunk of queries once, in document order (hits.id are fast-storage documents).
  void rescore_candidates(const TieredQQuery* weights, std::vector<std::vector<TieredQHit>>& candidates,
                          size_t k) const;
  // Room for count documents ahead of an empty index's appends.
  void reserve_rows(size_t count);
  // Fast payload bytes (storage, then fine rows) as save() and TieredQIVFIndex persist them.
  std::vector<uint8_t> fast_payload() const;
  // Query weights for fine-tier scans and rescoring, from the scan's own projection where it has one.
  std::vector<TieredQQuery> fine_query_weights(const float* queries, size_t rows) const;
};
// Fast-mode inverted-file index. With lists == 0 each of the codec's k-means centroids (at most 256) forms one list
// at no extra storage; otherwise build trains a separate k-means coarse quantizer with that many lists (seeded sample
// of at least 32,768 or 64 rows per list, or the sample given to train_lists) and stores its centroids. Coarse lists
// live in model space for inner product and cosine, and in input space for L2 (plain L2 k-means: the L2 lift's model
// space is dominated by row norms; L2 indexes saved before payload version 4 keep their model-space lists). Lists are
// stored contiguously in one fast representation, so every list scan uses the flat index's kernels; each list starts
// on a 32-document boundary (padded with copies of its last row, which are never returned). Search ranks lists by
// distance to the query in the coarse quantizer's space and scans the nearest nprobe nonempty lists. Ids are input
// row order. Build replaces the content.
// Range-checkpoint pruning of list scans (exact FP32 code scans on AVX-512; nibble byte-table scans on AVX-512, AVX2
// and NEON; other scans ignore it): after a prefix of its code a document skips the rest when its score bound cannot
// reach the running top k. With confidence 0 (default) the bound covers every suffix code, so hits equal the unpruned
// scan. A positive confidence bounds the suffix at that many standard deviations of its score for unit-Gaussian
// coordinates: approximate and faster. On OpenAI 100k at 4 bits/dim, confidence 8 kept recall at every nprobe.
// prefix_tiers automatic_prefix (default) sizes the prefix per scan: nibble scans leave a fixed share of their work to
// the suffix (larger with a positive confidence), FP32 code scans use the first four tiers. Other values prune after
// that many tiers when the model has more; prefix_tiers 3 prunes more for a small recall loss.
struct TieredQPruning {
  static constexpr uint32_t automatic_prefix = 0;
  double confidence = 0;
  uint32_t prefix_tiers = automatic_prefix;
};
// Coarse list training and assignment of TieredQIVFIndex builds (#48); build-time only, not persisted. iterations
// counts the Lloyd iterations of list training. assignment_supers > 0 assigns rows (in every training iteration and in
// the build) to their nearest list among the members of their assignment_probes nearest supers, a k-means of the
// lists' centroids into assignment_supers groups: on Qwen 1024-D (500K rows, 32K lists) 256 supers and 16 probes gave
// 99.0% exact labels at +0.1% objective, 8.5x faster than exact assignment. Rows whose nearest list lies in a probed
// super get the exact label up to floating-point near-ties (both score the same rounded centroids; exact assignment
// sums in FP32 BLAS, restricted scoring in the binary16 panel kernel). 0 assigns exactly, via BLAS sgemm. Lists that
// build() trains itself use a seeded sample of max(32,768, sample_rows_per_list x lists) rows (train_lists() takes
// its own sample). These member defaults are the exact recipe; automatic() is what an index uses unless
// set_coarse_training() overrides it.
struct TieredQCoarseTraining {
  uint32_t iterations = 20;
  uint32_t assignment_supers = 0;
  uint32_t assignment_probes = 16;
  uint32_t sample_rows_per_list = 64;
  // From fast_lists lists on: 4 iterations on 40 rows per list (Faiss's budget) with 256 x 16 restricted assignment,
  // the 8.84M x 1024 shipping build (64K lists in 232 s against 1,621 s for Faiss); the exact recipe below. On Qwen 1M x
  // 1024 (#52) fast training kept recall@10 within 0.002 of the exact recipe from recall 0.93 up at 8K-32K lists and
  // trained them 25-43x faster (8K: 136 -> 5.5 s); at 4K it lost up to 0.008 (nprobe 64), so smaller counts stay exact.
  static constexpr uint32_t fast_lists = 8192;
  static TieredQCoarseTraining automatic(size_t lists);
  void validate() const;
};
// Wall seconds of the last build() per phase: list training (0 for given lists), list assignment (0 for codec lists),
// encoding and storing the documents (codec lists: the encoding that finds their lists too), and the router.
struct TieredQBuildStats {
  double train_s = 0, assign_s = 0, encode_s = 0, router_s = 0;
};
// Two-level compressed coarse router (#47) for searches over many coarse lists, where ranking every
// centroid exactly is memory-bound (about 9 ms per query at 64K x 1024). The nonempty lists' centroids are grouped
// into supers (k-means, about one per lists_per_super centroids), stored as codes of a small L2 TieredQ IVF index
// (bits_per_dimension, 4-bit scan cap): a search ranks the supers exactly, scores the codes of the probed supers'
// member centroids, then re-ranks the best rerank of them exactly on the same rounded centroids the exact ranking uses
// and keeps the compressed order beyond them. The number of probed supers grows with nprobe: enough supers to hold
// expansion member centroids per wanted list, at least min_super_probes.
struct TieredQRouterConfig {
  uint32_t lists_per_super = 128;
  double bits_per_dimension = 2;
  uint32_t rerank = 64;           // tuned on Qwen 8.84M x 1024 at 64K lists: 256 cost 0.1-0.3 ms, not recall, above 0.945
  double expansion = 64;          // tuned on Qwen 8.84M x 1024 at 64K-128K lists (#47): more costs time, not recall
  uint32_t min_super_probes = 32;
  uint64_t seed = 0;
  // Indexes with at least this many coarse lists build the router with this config unless remove_router() opts out.
  // On Qwen 1M x 1024 (#52) the router matched exact ranking's recall from 8K lists on and was faster (8K: 0.25 ms of
  // 1.4-3 ms per query; 32K: 2.8 vs 6.1 ms at recall 0.957); at 4K, where every super is probed, it was 0.1 ms slower.
  static constexpr uint32_t automatic_lists = 8192;
  void validate() const;
};
// Per-search choice of coarse ranking. automatic uses the router whenever one is built, for single queries and batches
// alike: on Qwen 8.84M x 1024 at 64K lists, 32-thread batches through the router reached 1.4x the QPS of the exact GEMM
// ranking at equal nprobe (#51). router forces the router; exact ranks every centroid through the GEMM. super_probes > 0
// overrides the router's probed supers.
struct TieredQRouting {
  enum class Mode { automatic, router, exact };
  Mode mode = Mode::automatic;
  uint32_t super_probes = 0;
};
class TieredQIVFIndex {
 public:
  // lists 0: one list per codec centroid. Coarse indexes train their lists as TieredQCoarseTraining::automatic(lists)
  // says, and from TieredQRouterConfig::automatic_lists lists on build the router at its defaults; explicit
  // set_coarse_training(), build_router() and remove_router() override either.
  explicit TieredQIVFIndex(TieredQ codec, uint32_t lists = 0, uint64_t seed = 0);
  // List count for rows vectors: the power of two nearest rows / r, or 0 (codec lists) where that is at most the
  // codec's 256 centroids. r falls log-linearly from 256 rows per list at 1M rows to 135 at 8.84M (held outside), the
  // best measured counts on Qwen x 1024 (#52): 4K lists at 1M (ahead of 8K, 16K and 32K, router or not) and 64K at 8.84M.
  static uint32_t automatic_lists(size_t rows);
  // Scan cap for an IVF codec trained with config (its max_scan_bits when set): 4 at up to 2 bits per dimension for
  // fast targets without fine tiers, where every tier then scans on the nibble kernels (1.7x RaBitQ's IVF QPS at equal
  // recall and memory on OpenAI 100k, about 0.006 recall at full probe); 0 (the codec's automatic cap) otherwise, as
  // the 4-bit cap loses recall at 4 bits per dimension.
  static uint32_t automatic_scan_bits(const TieredQConfig& config);
  const TieredQ& codec() const { return documents_.codec(); }
  // Coarse indexes only: trains the lists on sample (rows x config().dimensions, at least one row per list) instead
  // of on a sample of build()'s rows, so a partition learned on earlier data serves later builds unchanged. Loaded
  // indexes with trained lists keep them the same way.
  void train_lists(const float* sample, size_t rows);
  // Coarse indexes only: fixes the lists to given centroids (lists() x coarse_dimensions() floats, row-major) instead
  // of training them, e.g. a partition trained elsewhere; build() then only assigns. Centroids are in the coarse
  // space: input vectors for L2 and inner product, normalized vectors for cosine.
  void set_lists(const float* centroids);
  // Coarse indexes only: list training for later train_lists() and build() calls, overriding the automatic one.
  void set_coarse_training(TieredQCoarseTraining training);
  // The training later builds use: the one set, else automatic(lists()) (coarse indexes only).
  TieredQCoarseTraining coarse_training() const;
  // Floats per coarse centroid: the input dimensions, plus the lift's one for L2 indexes saved before version 4.
  size_t coarse_dimensions() const { return input_space_coarse_ ? codec().config().dimensions : codec().model_dimensions(); }
  void build(const float* rows, size_t count);
  // Fills out (count x config().dimensions floats, row-major) with the input rows ids[0..count), for builds whose rows
  // are not held as one float array (e.g. decoded from a compact store). Called from the building thread only.
  using RowLoader = std::function<void(const uint32_t* ids, size_t count, float* out)>;
  // The same build pulling rows through a loader, a bounded chunk at a time; equals build() on the loaded rows.
  void build(const RowLoader& rows, size_t count);
  // Coarse indexes only: builds the two-level router over the current lists (now when built, otherwise at the next
  // build) and rebuilds it after every later build(), set_lists() or train_lists() and build. Loaded indexes keep theirs.
  void build_router(TieredQRouterConfig config = {});
  // Drops the router and its config, so searches rank every centroid exactly and later builds add none.
  void remove_router();
  bool has_router() const { return bool(router_); }
  // Router codes, its super index and the row-major centroid copy for the exact re-rank; included in storage_bytes().
  size_t router_bytes() const;
  const TieredQBuildStats& build_stats() const { return build_stats_; }
  size_t size() const { return size_; }
  size_t lists() const { return list_rows_.size(); }
  size_t list_size(size_t list) const { return list_rows_.at(list); }
  // The list of every input row (row order), e.g. to share the partition with another index.
  std::vector<uint32_t> list_assignment() const;
  // Fast representation (including list padding), list offsets and the Elias-Fano id lists.
  size_t storage_bytes() const;
  std::vector<TieredQHit> search(const float* query, size_t k, size_t nprobe, TieredQPruning pruning = {},
                                 TieredQRouting routing = {}) const;
  // Results are rows x min(k, size()) in row-major order; slots past the probed rows hold id UINT32_MAX and
  // score -inf. Queries are split across OpenMP workers; a single query scans on the calling thread.
  void search_batch(const float* queries, size_t rows, size_t k, size_t nprobe, float* scores, uint32_t* ids,
                    TieredQPruning pruning = {}, TieredQRouting routing = {}) const;
  // The same searches with the queries' codec().query_weights() prepared by the caller, which reuses them (e.g. a
  // TieredQRefinement rescoring the hits from the same weights) instead of projecting each query again.
  std::vector<TieredQHit> search(const float* query, const TieredQQuery& weights, size_t k, size_t nprobe,
                                 TieredQPruning pruning = {}, TieredQRouting routing = {}) const;
  void search_batch(const float* queries, const TieredQQuery* weights, size_t rows, size_t k, size_t nprobe,
                    float* scores, uint32_t* ids, TieredQPruning pruning = {}, TieredQRouting routing = {}) const;
  // The nprobe lists a search of query would scan, nearest first (empty lists skipped), ranked as routing says.
  std::vector<uint32_t> probe_lists(const float* query, size_t nprobe, TieredQRouting routing = {}) const;
  // Stored document of every input row id (id order), to address document_rows().
  std::vector<uint32_t> id_documents() const;
  // codec().encode() rows (bytes_per_vector() each) of stored documents, recovered exactly from the list storage, so
  // an index keeps no row-major copy of its codes (e.g. for TieredQRefinement rescoring of its hits).
  void document_rows(const uint32_t* documents, size_t count, uint8_t* rows) const;
  // Self-contained bytes (model, lists and fast payload) for an engine's own storage container.
  std::vector<uint8_t> serialize() const;
  static TieredQIVFIndex deserialize(const std::vector<uint8_t>& bytes);
  void save(const std::string& path) const;
  static TieredQIVFIndex load(const std::string& path);
 private:
  std::vector<TieredQHit> search_hits(const float* query, const TieredQQuery* weights, size_t k, size_t nprobe,
                                      TieredQPruning pruning, TieredQRouting routing) const;
  void search_rows(const float* queries, const TieredQQuery* weights, size_t rows, size_t k, size_t nprobe,
                   float* scores, uint32_t* ids, TieredQPruning pruning, TieredQRouting routing) const;
  void refresh_router();
  const float* model_rows(const float* rows, size_t first, size_t count, std::vector<float>& buffer) const;
  std::vector<uint32_t> coarse_lists(const RowLoader& rows, size_t count);
  void write_lists(tieredq_detail::Writer& writer) const;
  template<class Reader> static TieredQIVFIndex read_lists(TieredQ model, Reader& reader);
  // Version 2 payloads store every padded document's id raw; build() wrote them ascending per list.
  template<class Reader> void read_raw_ids(Reader& reader, size_t padded);
  template<class Reader> void read_router(Reader& reader);
  struct Place { uint32_t list, position; };
  Place place_of(size_t document) const;
  bool live(size_t document) const;  // not list padding
  // Input rows of hits on stored documents.
  std::vector<uint32_t> hit_rows(const std::vector<TieredQHit>& hits) const;
  size_t stored_documents() const;
  TieredQIndex documents_;
  std::vector<uint32_t> list_begin_, list_rows_; // first padded document and live rows per list
  tieredq_detail::EliasFanoLists ids_;           // input rows of every list, ascending
  size_t size_ = 0;
  uint32_t coarse_lists_ = 0;                    // 0: lists are the codec's centroids
  uint64_t seed_ = 0;
  bool fixed_lists_ = false;                     // train_lists() trained coarse_; build() only assigns
  std::optional<TieredQCoarseTraining> coarse_training_;  // unset: automatic(lists)
  TieredQBuildStats build_stats_;
  // Coarse centroids live in input space (new L2 indexes: plain L2 k-means, assignment and ranking on the input
  // vectors) instead of the codec's model space (inner product, cosine, and L2 indexes saved before payload version 4,
  // whose lifted space is dominated by row norms).
  bool input_space_coarse_ = false;
  // Rows [first, first + count) in the coarse quantizer's space.
  const float* coarse_rows(const float* rows, size_t first, size_t count, std::vector<float>& buffer) const;
  std::shared_ptr<const tieredq_detail::CoarseQuantizer> coarse_;
  std::shared_ptr<const TieredQRouterConfig> router_config_;  // set by build_router()
  std::shared_ptr<const tieredq_detail::CoarseRouter> router_;
};
} // namespace dv
