// Runtime-sized TieredQ. Link dv_tieredq (LAPACK/BLAS); no Python dependency.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "tieredq/detail/tieredq_scan_plan.h"
#include "tieredq/detail/tieredq_elias_fano.h"
#include "tieredq/detail/tieredq_shared_array.h"

namespace dv {
namespace tieredq_detail { struct FastFine; class FastRowReader; class StageProfile; struct FastScanCache; struct FastQueryBasis; struct EncoderData; class LazyEncoder;
                           struct TrainingRows;
                           struct CoarseQuantizer; struct Writer; }
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
// at no extra storage; otherwise build trains a separate model-space k-means coarse quantizer with that many lists
// (seeded sample of at least 32,768 or 64 rows per list, or the sample given to train_lists) and stores its centroids. Lists are stored contiguously in
// one fast representation, so every list scan uses the flat index's kernels; each list starts on a 32-document
// boundary (padded with copies of its last row, which are never returned). Search ranks lists by model-space
// distance to the query and scans the nearest nprobe nonempty lists. Ids are input row order. Build replaces the
// content.
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
class TieredQIVFIndex {
 public:
  explicit TieredQIVFIndex(TieredQ codec, uint32_t lists = 0, uint64_t seed = 0);
  const TieredQ& codec() const { return documents_.codec(); }
  // Coarse indexes only: trains the lists on sample (rows x config().dimensions, at least one row per list) instead
  // of on a sample of build()'s rows, so a partition learned on earlier data serves later builds unchanged. Loaded
  // indexes with trained lists keep them the same way.
  void train_lists(const float* sample, size_t rows);
  void build(const float* rows, size_t count);
  size_t size() const { return size_; }
  size_t lists() const { return list_rows_.size(); }
  size_t list_size(size_t list) const { return list_rows_.at(list); }
  // The list of every input row (row order), e.g. to share the partition with another index.
  std::vector<uint32_t> list_assignment() const;
  // Fast representation (including list padding), list offsets and the Elias-Fano id lists.
  size_t storage_bytes() const;
  std::vector<TieredQHit> search(const float* query, size_t k, size_t nprobe, TieredQPruning pruning = {}) const;
  // Results are rows x min(k, size()) in row-major order; slots past the probed rows hold id UINT32_MAX and
  // score -inf. Queries are split across OpenMP workers; a single query scans on the calling thread.
  void search_batch(const float* queries, size_t rows, size_t k, size_t nprobe, float* scores, uint32_t* ids,
                    TieredQPruning pruning = {}) const;
  // The same searches with the queries' codec().query_weights() prepared by the caller, which reuses them (e.g. a
  // TieredQRefinement rescoring the hits from the same weights) instead of projecting each query again.
  std::vector<TieredQHit> search(const float* query, const TieredQQuery& weights, size_t k, size_t nprobe,
                                 TieredQPruning pruning = {}) const;
  void search_batch(const float* queries, const TieredQQuery* weights, size_t rows, size_t k, size_t nprobe,
                    float* scores, uint32_t* ids, TieredQPruning pruning = {}) const;
  // Stored document of every input row id (id order), to address document_rows().
  std::vector<uint32_t> id_documents() const;
  // codec().encode() rows (bytes_per_vector() each) of stored documents, recovered exactly from the list storage, so
  // an index keeps no row-major copy of its codes (e.g. for TieredQRefinement rescoring of its hits).
  void document_rows(const uint32_t* documents, size_t count, uint8_t* rows) const;
  // The per-document terms of those rows' scores (centroid id, factor, correction), one cache line each instead of
  // a row's codes, e.g. for additive TieredQRefinement::score_from_prefix of this index's hits.
  void document_terms(const uint32_t* documents, size_t count, uint8_t* centroids, float* factors,
                      float* corrections) const;
  // Self-contained bytes (model, lists and fast payload) for an engine's own storage container.
  std::vector<uint8_t> serialize() const;
  static TieredQIVFIndex deserialize(const std::vector<uint8_t>& bytes);
  void save(const std::string& path) const;
  static TieredQIVFIndex load(const std::string& path);
 private:
  std::vector<TieredQHit> search_hits(const float* query, const TieredQQuery* weights, size_t k, size_t nprobe,
                                      TieredQPruning pruning) const;
  void search_rows(const float* queries, const TieredQQuery* weights, size_t rows, size_t k, size_t nprobe,
                   float* scores, uint32_t* ids, TieredQPruning pruning) const;
  const float* model_rows(const float* rows, size_t first, size_t count, std::vector<float>& buffer) const;
  std::vector<uint32_t> coarse_lists(const float* rows, size_t count);
  void write_lists(tieredq_detail::Writer& writer) const;
  template<class Reader> static TieredQIVFIndex read_lists(TieredQ model, Reader& reader);
  // Version 2 payloads store every padded document's id raw; build() wrote them ascending per list.
  template<class Reader> void read_raw_ids(Reader& reader, size_t padded);
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
  std::shared_ptr<const tieredq_detail::CoarseQuantizer> coarse_;
};
} // namespace dv
